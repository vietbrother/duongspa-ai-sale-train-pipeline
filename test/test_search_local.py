"""
test_search_local.py — Test semantic search voi local embedding (sentence-transformers)

Chay:
    cd duongspa-ai-sale-train-pipeline
    python test/test_search_local.py

Yeu cau:
    - pip install sentence-transformers torch
    - QDRANT_URL dang chay va da co du lieu voi VECTOR_SIZE=384
    - EMBED_PROVIDER=local, VECTOR_SIZE=384

Luu y:
    - Du lieu phai duoc embed bang cung model local khi chay main.py
    - Neu du lieu embed bang openai (1536 dims) -> khong the dung local test nay
    - Neu muon test cheo provider, dung test_search_openai.py
"""

import os
import sys

# Them thu muc goc vao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# Force local provider
os.environ["EMBED_PROVIDER"] = "local"
os.environ["EMBED_ENABLED"] = "true"

# Local model mac dinh — co the override qua bien moi truong
LOCAL_MODEL = os.environ.get(
    "EMBED_LOCAL_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
os.environ["EMBED_LOCAL_MODEL"] = LOCAL_MODEL

# VECTOR_SIZE phai = 384 cho model paraphrase-multilingual-MiniLM-L12-v2
os.environ["VECTOR_SIZE"] = os.environ.get("VECTOR_SIZE", "384")

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


def check_local_model():
    """Kiem tra sentence-transformers co cai chua."""
    sep("Kiem tra Local Embedding Model")
    print(f"  Provider   : local")
    print(f"  Model      : {LOCAL_MODEL}")
    print(f"  VECTOR_SIZE: {VECTOR_SIZE}")
    try:
        from sentence_transformers import SentenceTransformer
        print("  sentence-transformers: [OK]")
    except ImportError:
        print("  [ERROR] sentence-transformers chua cai!")
        print("  Chay: pip install sentence-transformers torch")
        sys.exit(1)


def check_qdrant():
    """Kiem tra ket noi Qdrant va collection."""
    sep("Kiem tra ket noi Qdrant")
    kwargs = {"url": QDRANT_URL}
    if QDRANT_API_KEY:
        kwargs["api_key"] = QDRANT_API_KEY
    client = QdrantClient(**kwargs)
    info = client.get_collection(COLLECTION_NAME)
    count = client.count(COLLECTION_NAME, exact=True).count
    remote_size = info.config.params.vectors.size
    print(f"  URL        : {QDRANT_URL}")
    print(f"  Collection : {COLLECTION_NAME}")
    print(f"  Points     : {count}")
    print(f"  Vector size: {remote_size} (Qdrant)")
    print(f"  VECTOR_SIZE: {VECTOR_SIZE} (config)")

    if count == 0:
        print("\n  [WARN] Collection rong! Hay chay main.py truoc.")
        sys.exit(1)

    if remote_size != int(VECTOR_SIZE):
        print(f"\n  [ERROR] Vector size khong khop!")
        print(f"  Qdrant={remote_size} != VECTOR_SIZE={VECTOR_SIZE}")
        print(f"  Hay recreate collection voi VECTOR_SIZE={remote_size}")
        print(f"  hoac re-embed du lieu voi VECTOR_SIZE={VECTOR_SIZE}")
        sys.exit(1)

    print(f"  Status     : {info.status}")
    return count


def check_embed_provider():
    """Kiem tra local embedding co hoat dong khong."""
    sep("Kiem tra Local Embedding (warm-up)")
    sample = "xin chào, tôi muốn tư vấn dịch vụ"
    print(f"  Sample text: {sample!r}")
    print("  Dang load model lan dau (co the mat 30-60s)...")
    try:
        vec = embed(sample)
        print(f"  Vector len : {len(vec)}")
        print(f"  Vec[:5]    : {[round(v, 4) for v in vec[:5]]}")
        if len(vec) != int(VECTOR_SIZE):
            print(f"  [ERROR] Vector size {len(vec)} != VECTOR_SIZE {VECTOR_SIZE}")
            print(f"  Model nay tao vector {len(vec)} dims. Set VECTOR_SIZE={len(vec)}")
            sys.exit(1)
        print("  [OK] Embedding hoat dong binh thuong")
    except Exception as e:
        print(f"  [ERROR] {e}")
        sys.exit(1)


def run_search_tests():
    """Chay cac bai test search voi cau hoi khach hang."""
    sep("Test Semantic Search — Local Model")
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
                preview = r['text'][:80].replace('\n', ' ')
                print(f"         text: {preview!r}...")

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


def run_interactive():
    """Che do nhap tay — nhap cau hoi tu ban phim."""
    sep("Interactive Mode — nhap cau hoi tuy y (Ctrl+C de thoat)")
    print("  Provider: local |", LOCAL_MODEL)
    while True:
        try:
            query = input("\nNhap cau hoi khach hang: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nThoat.")
            break
        if not query:
            continue

        intent = intent_detect(query)
        print(f"  Intent detect: {intent}")

        results = search(query, top_k=TOP_K)
        if not results:
            print("  Khong tim thay ket qua.")
            continue

        for j, r in enumerate(results, 1):
            print(f"  [{j}] score={r['score']:.3f} | seg={r['segment']:4s} "
                  f"| state={r['dominant_state']} | outcome={r['outcome_label']}")
            print(f"       {r['text'][:120].replace(chr(10), ' ')!r}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test search voi local embedding")
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Che do nhap tay (nhap cau hoi tu ban phim)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  DuongSpa — Semantic Search Test (Local Model)")
    print("=" * 60)

    check_local_model()
    check_qdrant()
    check_embed_provider()

    if args.interactive:
        run_interactive()
    else:
        passed = run_search_tests()
        sys.exit(0 if passed == len(TEST_QUERIES) else 1)
