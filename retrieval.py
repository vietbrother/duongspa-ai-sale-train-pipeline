
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from config import QDRANT_URL, COLLECTION_NAME, TOP_K
from embedding import embed

client = QdrantClient(url=QDRANT_URL)

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


def hybrid_score(hit, intent: str) -> float:
    """Tính hybrid score kết hợp semantic + quality signals."""
    p = hit.payload
    score = hit.score  # Cosine similarity (0-1)

    # Quality boost: conversation có score cao → câu trả lời tốt hơn
    conv_score = p.get("score", 0)
    score += (conv_score / 200) * 0.3  # Normalize score ~0-1 rồi boost 30%

    # CTA boost: ưu tiên segments có CTA
    if p.get("has_closing_cta"):
        score *= 1.15

    # Segment boost: ưu tiên theo paid value
    segment_boost = {"VIP": 1.2, "HIGH": 1.15, "MID": 1.1, "LOW": 1.0}
    score *= segment_boost.get(p.get("segment", "LOW"), 1.0)

    return score


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
    scored = [(hybrid_score(r, intent), r) for r in results]
    scored.sort(reverse=True, key=lambda x: x[0])

    return [
        {
            "text": r.payload.get("text", ""),
            "score": final_score,
            "segment": r.payload.get("segment", ""),
            "service_interest": r.payload.get("service_interest", ""),
            "has_cta": r.payload.get("has_closing_cta", False),
        }
        for final_score, r in scored[:top_k]
    ]
