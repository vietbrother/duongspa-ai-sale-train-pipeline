
import os

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from config import QDRANT_URL, COLLECTION_NAME, TOP_K
from embedding import embed
from state_engine import detect_state_single

_api_key = os.environ.get("QDRANT_API_KEY", "")
_kwargs = {"url": QDRANT_URL}
if _api_key:
    _kwargs["api_key"] = _api_key

client = QdrantClient(**_kwargs)

# Intent keywords mapping
INTENT_KEYWORDS = {
    "ask_price": ["giá", "bao nhiêu", "phí", "chi phí", "tiền"],
    "concern": ["đau", "an toàn", "tác dụng phụ", "có đau", "nguy hiểm", "sợ"],
    "location": ["ở đâu", "địa chỉ", "chi nhánh"],
    "promotion": ["khuyến mãi", "giảm giá", "ưu đãi", "sale"],
    "booking": ["đặt lịch", "hẹn", "book", "khi nào"],
    "service_info": ["liệu trình", "dịch vụ", "gồm gì", "bao lâu"],
}


def intent_detect(query: str) -> str:
    """Phát hiện intent từ query bằng keyword matching."""
    query_lower = query.lower()
    for intent, keywords in INTENT_KEYWORDS.items():
        if any(kw in query_lower for kw in keywords):
            return intent
    return "general"


def hybrid_score(hit, intent: str, query_state: str = None) -> float:
    """Tính hybrid score v3.1: semantic + quality + state + outcome.

    Final score =
      0.30 × similarity
    + 0.25 × quality_score (from CRM outcome)
    + 0.15 × intent_boost
    + 0.10 × state_match_boost
    + 0.10 × time_decay (placeholder)
    + 0.10 × segment_boost
    """
    p = hit.payload
    similarity = hit.score  # Cosine similarity (0-1)

    # Quality score from CRM outcome (weighted_score normalized)
    conv_score = p.get("weighted_score", p.get("score", 0))
    quality = min(conv_score / 200, 1.0)

    # Intent boost (placeholder: 1.0 for all — can be tuned)
    intent_boost = 1.0

    # State match boost (v3.1)
    state_boost = 0.0
    if query_state and p.get("dominant_state") == query_state:
        state_boost = 1.0
    elif query_state and p.get("final_state") == query_state:
        state_boost = 0.5

    # Segment boost
    segment_boost_map = {"VIP": 1.0, "HIGH": 0.8, "MID": 0.6, "LOW": 0.3}
    seg_boost = segment_boost_map.get(p.get("segment", "LOW"), 0.3)

    # Outcome weight boost (v3.1)
    outcome_w = p.get("outcome_weight", 1.0)

    # CTA boost
    cta = 0.15 if p.get("has_closing_cta") else 0.0

    final = (
        0.30 * similarity
        + 0.25 * quality * outcome_w
        + 0.15 * intent_boost
        + 0.10 * state_boost
        + 0.10 * seg_boost
        + 0.10 * 0.5  # time_decay placeholder
        + cta
    )
    return final


def search(query: str, segment: str = None, top_k: int = None) -> list[dict]:
    """Search Qdrant với hybrid scoring.

    Args:
        query: Câu hỏi người dùng
        segment: Filter theo segment (LOW/MID/HIGH/VIP)
        top_k: Số kết quả trả về

    Returns:
        List các context dict {text, score, segment, ...}
    """
    top_k = top_k or TOP_K
    vec = embed(query)

    # Build filter nếu có segment
    query_filter = None
    if segment:
        query_filter = Filter(
            must=[FieldCondition(key="segment", match=MatchValue(value=segment))]
        )

    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=vec,
        query_filter=query_filter,
        limit=top_k * 4,  # Lấy nhiều hơn để re-rank
        with_payload=True,
    )

    if not results:
        return []

    intent = intent_detect(query)
    query_state = detect_state_single(query, role="user")
    scored = [(hybrid_score(r, intent, query_state), r) for r in results]
    scored.sort(reverse=True, key=lambda x: x[0])

    return [
        {
            "text": r.payload.get("text", ""),
            "score": final_score,
            "segment": r.payload.get("segment", ""),
            "service_interest": r.payload.get("service_interest", ""),
            "has_cta": r.payload.get("has_closing_cta", False),
            "dominant_state": r.payload.get("dominant_state", ""),
            "outcome_label": r.payload.get("outcome_label", ""),
            "outcome_weight": r.payload.get("outcome_weight", 1.0),
        }
        for final_score, r in scored[:top_k]
    ]
