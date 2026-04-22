import json
import os
from datetime import datetime

from config import OUTPUT_DIR, SCORE_THRESHOLD, TOP_SALES_PERCENTILE
from loader import load_crm, load_conversations, load_chatpage, identify_top_sales
from joiner import join_data
from feature import extract_features
from reward import compute_reward
from segment import get_segment
from scoring import compute_score
from prediction import predict_revenue
from ground_truth import label_outcomes, apply_outcome_weights, compute_ground_truth_stats
from tone_style import extract_style_profile
from chunker import extract_segments
from embedding import embed_batch
from qdrant_ops import create_collection, push

SYSTEM_PROMPT = (
    "Bạn là nhân viên sales spa chuyên chốt khách. "
    "Luôn: hỏi thêm thông tin, dẫn khách tới hành động, "
    "có CTA (SĐT / lịch / Zalo), không kết thúc hội thoại."
)


def run():
    print("=" * 60)
    print("🚀 DuongSpa AI Sales Training Pipeline v3.1")
    print("=" * 60)

    # --- 1. Load data ---
    print("\n📂 [1/10] Loading data...")
    crm = load_crm()
    conv = load_conversations()
    chatpage = load_chatpage()
    print(f"   CRM: {len(crm)} customers")
    print(f"   Conversations: {len(conv)} messages")
    print(f"   Chatpage staff: {len(chatpage)} staff members")

    # --- 2. Identify top sales ---
    print("\n⭐ [2/10] Identifying top sales staff...")
    top_sales = identify_top_sales(chatpage, percentile=TOP_SALES_PERCENTILE)
    print(f"   Top sales ({TOP_SALES_PERCENTILE*100:.0f}%): {top_sales}")

    # --- 3. Join ---
    print("\n🔗 [3/10] Joining CRM + Conversations...")
    df = join_data(conv, crm)
    print(f"   Joined: {len(df)} rows")
    print(f"   Conversations with CRM match: {df['paid_value'].gt(0).sum()}")

    # Mark conversations assigned to top sales
    if "assigned_staff" in df.columns and top_sales:
        df["is_top_sales_conv"] = df["assigned_staff"].isin(top_sales)
        top_conv_count = df[df["is_top_sales_conv"]]["conversation_id"].nunique()
        print(f"   Conversations from top sales: {top_conv_count}")
    else:
        df["is_top_sales_conv"] = False

    # --- 4. Extract features ---
    print("\n🔬 [4/10] Extracting features...")
    df = extract_features(df)

    # --- 5. Compute reward & scoring ---
    print("\n📊 [5/10] Computing reward & scoring...")
    df["reward"] = df.apply(compute_reward, axis=1)
    df["segment"] = df["paid_value"].apply(get_segment)
    df["score"] = df.apply(compute_score, axis=1)
    df["pred_prob"], df["expected_rev"] = zip(*df.apply(predict_revenue, axis=1))

    # --- 6. Ground Truth labeling + outcome weighting (v3.1) ---
    print("\n🏷️  [6/10] Ground Truth labeling & outcome weighting...")
    df = label_outcomes(df)
    df = apply_outcome_weights(df)
    gt_stats = compute_ground_truth_stats(df)
    print(f"   Outcome distribution: {gt_stats['outcome_distribution']}")
    print(f"   Ground truth coverage: {gt_stats['ground_truth_coverage']}%")
    print(f"   Gold samples (won): {gt_stats['gold_count']}")

    # --- 7. Tone & Style Extraction (v3.1) ---
    print("\n🎨 [7/10] Extracting Tone & Style from top sales...")
    style_profile = extract_style_profile(df, top_sales)
    print(f"   Analyzed {style_profile.get('sample_count', 0)} assistant messages")
    print(f"   Emoji rate: {style_profile.get('emoji_usage_rate', 0):.1%}")
    print(f"   Avg msg length: {style_profile.get('avg_msg_length', 0):.0f} chars")
    print(f"   CTA phrases found: {len(style_profile.get('cta_phrases', []))}")

    # --- 8. Filter high-quality conversations ---
    print(f"\n🎯 [8/10] Filtering (weighted_score > {SCORE_THRESHOLD})...")
    total_convs = df["conversation_id"].nunique()
    df = df[df["weighted_score"] > SCORE_THRESHOLD]
    filtered_convs = df["conversation_id"].nunique()
    print(f"   Before: {total_convs} conversations")
    print(f"   After:  {filtered_convs} conversations ({filtered_convs/max(total_convs,1)*100:.1f}%)")

    # --- 9. Extract segments (with state detection) ---
    print("\n✂️  [9/10] Extracting segments (with state detection)...")
    segments = extract_segments(df)
    print(f"   Generated: {len(segments)} segments")

    seg_dist = {}
    for s in segments:
        seg_dist[s["segment"]] = seg_dist.get(s["segment"], 0) + 1
    print(f"   Segment distribution: {seg_dist}")

    state_dist = {}
    for s in segments:
        st = s.get("dominant_state", "unknown")
        state_dist[st] = state_dist.get(st, 0) + 1
    print(f"   State distribution: {state_dist}")

    outcome_dist = {}
    for s in segments:
        ol = s.get("outcome_label", "unknown")
        outcome_dist[ol] = outcome_dist.get(ol, 0) + 1
    print(f"   Outcome distribution: {outcome_dist}")    # --- 10. Embed & Export & Push ---
    print(f"\n🧠 [10/10] Embedding {len(segments)} segments & exporting...")
    from config import EMBED_ENABLED, EMBED_PROVIDER
    print(f"   Embed enabled: {EMBED_ENABLED} | provider: {EMBED_PROVIDER}")

    # Dung embed_batch de giam so luong API calls (100 texts / request)
    texts = [seg["text"] for seg in segments]
    vectors = embed_batch(texts)
    if (len(vectors) % 50) == 0 or len(vectors) == len(segments):
        print(f"   Embedded {len(vectors)}/{len(segments)} segments")

    points = []
    for i, (seg, vector) in enumerate(zip(segments, vectors)):
        payload = {k: v for k, v in seg.items() if k not in ("messages", "state_sequence")}
        points.append({"id": i, "vector": vector, "payload": payload})

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 10a. Qdrant JSONL
    qdrant_path = os.path.join(OUTPUT_DIR, f"qdrant_points_{timestamp}.jsonl")
    with open(qdrant_path, "w", encoding="utf-8") as f:
        for point in points:
            f.write(json.dumps(point, ensure_ascii=False, default=str) + "\n")
    print(f"   ✅ Qdrant JSONL: {qdrant_path} ({len(points)} points)")

    # 10b. Fine-tune JSONL — sorted by weighted_score (gold first)
    finetune_path = os.path.join(OUTPUT_DIR, f"finetune_openai_{timestamp}.jsonl")
    ft_count = 0
    with open(finetune_path, "w", encoding="utf-8") as f:
        sorted_segs = sorted(segments, key=lambda s: s.get("weighted_score", 0), reverse=True)
        for seg in sorted_segs:
            msgs = seg.get("messages", [])
            if len(msgs) < 2:
                continue
            record = {"messages": [{"role": "system", "content": SYSTEM_PROMPT}, *msgs]}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            ft_count += 1
    print(f"   ✅ Fine-tune JSONL: {finetune_path} ({ft_count} samples)")

    # 10c. Q/A pairs
    qa_path = os.path.join(OUTPUT_DIR, f"qa_pairs_{timestamp}.json")
    all_qa = []
    for seg in segments:
        for qa in seg.get("qa_pairs", []):
            all_qa.append({
                "text": qa["question"], "response": qa["answer"],
                "segment": seg["segment"],
                "service_interest": seg.get("service_interest", ""),
                "score": seg["score"],
                "outcome_label": seg.get("outcome_label", ""),
                "dominant_state": seg.get("dominant_state", ""),
            })
    with open(qa_path, "w", encoding="utf-8") as f:
        json.dump(all_qa, f, ensure_ascii=False, indent=2)
    print(f"   ✅ Q/A pairs: {qa_path} ({len(all_qa)} pairs)")

    # 10d. Style profile
    style_path = os.path.join(OUTPUT_DIR, f"style_profile_{timestamp}.json")
    with open(style_path, "w", encoding="utf-8") as f:
        json.dump(style_profile, f, ensure_ascii=False, indent=2, default=str)
    print(f"   ✅ Style profile: {style_path}")

    # 10e. Report
    report_path = os.path.join(OUTPUT_DIR, f"report_{timestamp}.json")
    report = {
        "version": "3.1",
        "timestamp": timestamp,
        "total_messages": len(conv),
        "total_crm_customers": len(crm),
        "total_chatpage_staff": len(chatpage),
        "top_sales_staff": top_sales,
        "filtered_conversations": filtered_convs,
        "total_segments": len(segments),
        "total_finetune_samples": ft_count,
        "total_qa_pairs": len(all_qa),
        "segment_distribution": seg_dist,
        "state_distribution": state_dist,
        "outcome_distribution": outcome_dist,
        "ground_truth_stats": gt_stats,
        "style_profile_summary": {
            "sample_count": style_profile.get("sample_count", 0),
            "emoji_rate": style_profile.get("emoji_usage_rate", 0),
            "avg_msg_length": style_profile.get("avg_msg_length", 0),
        },
        "score_threshold": SCORE_THRESHOLD,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"   ✅ Report: {report_path}")

    # 10f. Push to Qdrant
    try:
        create_collection()
        push(points)
        print(f"   ✅ Pushed {len(points)} points to Qdrant")
    except Exception as e:
        print(f"   ⚠️  Qdrant push skipped (not running?): {e}")

    print("\n" + "=" * 60)
    print("✅ Pipeline v3.1 completed successfully!")
    print(f"   Segments: {len(segments)} | Fine-tune: {ft_count} | Q/A: {len(all_qa)}")
    print(f"   Gold (won): {outcome_dist.get('won', 0)} | Top sales: {len(top_sales)}")
    print("=" * 60)


if __name__ == "__main__":
    run()
