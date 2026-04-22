"""
test_search_openai.py — Test semantic search voi OpenAI embedding

Chay:
    cd duongspa-ai-sale-train-pipeline
    python test/test_search_openai.py

Yeu cau:
    - OPENAI_API_KEY hop le trong .env hoac bien moi truong
    - QDRANT_URL dang chay va da co du lieu (chay main.py truoc)
    - EMBED_PROVIDER=openai, VECTOR_SIZE=1536
"""

import os
import sys

# Them thu muc goc vao path de import config, embedding, retrieval
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# Force OpenAI provider
os.environ["EMBED_PROVIDER"] = "openai"
os.environ["EMBED_ENABLED"] = "true"

from qdrant_client import QdrantClient
from config import QDRANT_URL, COLLECTION_NAME, VECTOR_SIZE
from embedding import embed
from retrieval import search, intent_detect

# ─────────────────────────────────────────────
# Cau hinh
# ─────────────────────────────────────────────
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")
TOP_K = 3

# Cac cau hoi mau khach hang thuong hoi
TEST_QUERIES = [
    {
        "query": "Da mình bị mụn nhiều, liệu trình nào phù hợp?",
        "desc": "Khach hoi ve dich vu tri mun",
        "segment": None,
    },
    {
        "query": "Giá liệu trình trị nám bao nhiêu tiền chị?",
        "desc": "Khach hoi gia dich vu",
        "segment": None,
    },
    {
        "query": "Mình sợ đau lắm, làm có đau không?",
        "desc": "Khach lo ngai ve dau",
        "segment": None,
    },
    {
        "query": "Cho mình đặt lịch hẹn ngày mai được không?",
        "desc": "Khach muon dat lich",
        "segment": None,
    },
    {
        "query": "Spa ở đâu vậy chị? Có chi nhánh quận 7 không?",
        "desc": "Khach hoi dia chi",
        "segment": None,
    },
    {
        "query": "Khuyến mãi tháng này có gì không?",
        "desc": "Khach hoi khuyen mai",
        "segment": None,
    },
    {
        "query": "Mình muốn làm đẹp toàn diện, có gói nào không?",
        "desc": "Khach VIP muon goi toan dien",
        "segment": "VIP",
    },
]

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def sep(title=""):
    print("\n" + "═" * 60)
    if title:
        print(f"  {title}")
        print("═" * 60)


def check_qdrant():
    """Kiem tra ket noi Qdrant va collection."""
    sep("Kiem tra ket noi Qdrant")
    kwargs = {"url": QDRANT_URL}
    if QDRANT_API_KEY:
        kwargs["api_key"] = QDRANT_API_KEY
    client = QdrantClient(**kwargs)
    info = client.get_collection(COLLECTION_NAME)
    count = client.count(COLLECTION_NAME, exact=True).count
    print(f"  URL        : {QDRANT_URL}")
    print(f"  Collection : {COLLECTION_NAME}")
    print(f"  Points     : {count}")
    print(f"  Vector size: {info.config.params.vectors.size}")
    print(f"  Status     : {info.status}")
    if count == 0:
        print("\n  [WARN] Collection rong! Hay chay main.py truoc.")
        sys.exit(1)
    return count


def check_embed_provider():
    """Kiem tra OpenAI embedding co hoat dong khong."""
    sep("Kiem tra OpenAI Embedding")
    sample = "xin chào, tôi muốn tư vấn dịch vụ"
    print(f"  Provider   : openai")
    print(f"  Vector size: {VECTOR_SIZE}")
    print(f"  Sample text: {sample!r}")
    try:
        vec = embed(sample)
        print(f"  Vector len : {len(vec)}")
        print(f"  Vec[:5]    : {[round(v, 4) for v in vec[:5]]}")
        if len(vec) != VECTOR_SIZE:
            print(f"  [WARN] Vector size {len(vec)} != VECTOR_SIZE {VECTOR_SIZE}!")
        else:
            print("  [OK] Embedding hoat dong binh thuong")
    except Exception as e:
        print(f"  [ERROR] {e}")
        sys.exit(1)


def run_search_tests():
    """Chay cac bai test search voi cau hoi khach hang."""
    sep("Test Semantic Search — OpenAI")
    passed = 0
    for i, tc in enumerate(TEST_QUERIES, 1):
        query = tc["query"]
        segment = tc.get("segment")
        print(f"\n[{i}/{len(TEST_QUERIES)}] {tc['desc']}")
        print(f"  Query   : {query!r}")
        if segment:
            print(f"  Filter  : segment={segment}")

        intent = intent_detect(query)
        print(f"  Intent  : {intent}")

        try:
            results = search(query, segment=segment, top_k=TOP_K)
            if not results:
                print("  [WARN] Khong tim thay ket qua!")
                continue

            print(f"  Results : {len(results)} hits")
            for j, r in enumerate(results, 1):
                print(f"    [{j}] score={r['score']:.3f} | seg={r['segment']:4s} "
                      f"| state={r['dominant_state']:20s} | outcome={r['outcome_label']:12s} "
                      f"| cta={r['has_cta']}")
                # In doan text mau (50 chars dau)
                preview = r['text'][:80].replace('\n', ' ')
                print(f"         text: {preview!r}...")

            # Pass neu co it nhat 1 ket qua co score > 0
            if results[0]["score"] > 0:
                print("  [PASS]")
                passed += 1
            else:
                print("  [WARN] Score = 0, kiem tra lai du lieu")

        except Exception as e:
            print(f"  [ERROR] {e}")

    print()
    sep(f"Ket qua: {passed}/{len(TEST_QUERIES)} tests passed")
    return passed


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  DuongSpa — Semantic Search Test (OpenAI)")
    print("=" * 60)

    check_qdrant()
    check_embed_provider()
    passed = run_search_tests()

    sys.exit(0 if passed == len(TEST_QUERIES) else 1)
