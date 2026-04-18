import json
import os
from datetime import datetime

from config import OUTPUT_DIR, SCORE_THRESHOLD
from loader import load_crm, load_conversations
from joiner import join_data
from feature import extract_features
from reward import compute_reward
from segment import get_segment
from scoring import compute_score
from prediction import predict_revenue
from chunker import extract_segments
from embedding import embed
from qdrant_client import create_collection, push

SYSTEM_PROMPT = (
    "Bạn là nhân viên sales spa chuyên chốt khách. "
    "Luôn: hỏi thêm thông tin, dẫn khách tới hành động, "
    "có CTA (SĐT / lịch / Zalo), không kết thúc hội thoại."
)


def run():
    print("=" * 60)
    print("🚀 DuongSpa AI Sales Training Pipeline")
    print("=" * 60)

    # --- 1. Load data ---
    print("\n📂 [1/8] Loading data...")
    crm = load_crm()
    conv = load_conversations()
    print(f"   CRM: {len(crm)} customers")
    print(f"   Conversations: {len(conv)} messages")

    # --- 2. Join ---
    print("\n🔗 [2/8] Joining CRM + Conversations...")
    df = join_data(conv, crm)
    print(f"   Joined: {len(df)} rows")
    print(f"   Conversations with CRM match: {df['paid_value'].gt(0).sum()}")

    # --- 3. Extract features ---
    print("\n🔬 [3/8] Extracting features...")
    df = extract_features(df)

    # --- 4. Compute reward & scoring ---
    print("\n📊 [4/8] Computing reward & scoring...")
    df["reward"] = df.apply(compute_reward, axis=1)
    df["segment"] = df["paid_value"].apply(get_segment)
    df["score"] = df.apply(compute_score, axis=1)
    df["pred_prob"], df["expected_rev"] = zip(*df.apply(predict_revenue, axis=1))

    # --- 5. Filter high-quality conversations ---
    print(f"\n🎯 [5/8] Filtering (score > {SCORE_THRESHOLD})...")
    total_convs = df["conversation_id"].nunique()
    df = df[df["score"] > SCORE_THRESHOLD]
    filtered_convs = df["conversation_id"].nunique()
    print(f"   Before: {total_convs} conversations")
    print(f"   After:  {filtered_convs} conversations ({filtered_convs/max(total_convs,1)*100:.1f}%)")

    # --- 6. Extract segments ---
    print("\n✂️  [6/8] Extracting segments...")
    segments = extract_segments(df)
    print(f"   Generated: {len(segments)} segments")

    # Segment distribution
    seg_dist = {}
    for s in segments:
        seg_dist[s["segment"]] = seg_dist.get(s["segment"], 0) + 1
    print(f"   Distribution: {seg_dist}")

    # --- 7. Embed & build points ---
    print(f"\n🧠 [7/8] Embedding {len(segments)} segments...")
    points = []
    for i, seg in enumerate(segments):
        vector = embed(seg["text"])
        # Payload cho Qdrant (không bao gồm messages để giảm size)
        payload = {k: v for k, v in seg.items() if k != "messages"}
        points.append({
            "id": i,
            "vector": vector,
            "payload": payload,
        })
        if (i + 1) % 50 == 0:
            print(f"   Embedded {i + 1}/{len(segments)}...")

    # --- 8. Export & Push ---
    print("\n💾 [8/8] Exporting & pushing to Qdrant...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 8a. Export Qdrant JSONL (vectors + payload)
    qdrant_path = os.path.join(OUTPUT_DIR, f"qdrant_points_{timestamp}.jsonl")
    with open(qdrant_path, "w", encoding="utf-8") as f:
        for point in points:
            f.write(json.dumps(point, ensure_ascii=False, default=str) + "\n")
    print(f"   ✅ Qdrant JSONL: {qdrant_path} ({len(points)} points)")

    # 8b. Export Fine-tune JSONL (OpenAI format)
    finetune_path = os.path.join(OUTPUT_DIR, f"finetune_openai_{timestamp}.jsonl")
    ft_count = 0
    with open(finetune_path, "w", encoding="utf-8") as f:
        for seg in segments:
            msgs = seg.get("messages", [])
            if len(msgs) < 2:
                continue
            record = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    *msgs,
                ]
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            ft_count += 1
    print(f"   ✅ Fine-tune JSONL: {finetune_path} ({ft_count} samples)")

    # 8c. Export Q/A pairs cho Qdrant RAG
    qa_path = os.path.join(OUTPUT_DIR, f"qa_pairs_{timestamp}.json")
    all_qa = []
    for seg in segments:
        for qa in seg.get("qa_pairs", []):
            all_qa.append({
                "text": qa["question"],
                "response": qa["answer"],
                "segment": seg["segment"],
                "service_interest": seg.get("service_interest", ""),
                "score": seg["score"],
            })
    with open(qa_path, "w", encoding="utf-8") as f:
        json.dump(all_qa, f, ensure_ascii=False, indent=2)
    print(f"   ✅ Q/A pairs: {qa_path} ({len(all_qa)} pairs)")

    # 8d. Export summary report
    report_path = os.path.join(OUTPUT_DIR, f"report_{timestamp}.json")
    report = {
        "timestamp": timestamp,
        "total_messages": len(conv),
        "total_crm_customers": len(crm),
        "filtered_conversations": filtered_convs,
        "total_segments": len(segments),
        "total_finetune_samples": ft_count,
        "total_qa_pairs": len(all_qa),
        "segment_distribution": seg_dist,
        "score_threshold": SCORE_THRESHOLD,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"   ✅ Report: {report_path}")

    # 8e. Push to Qdrant
    create_collection()
    push(points)
    print(f"   ✅ Pushed {len(points)} points to Qdrant")

    print("\n" + "=" * 60)
    print("✅ Pipeline completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    run()
