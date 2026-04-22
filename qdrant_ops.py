import logging
import os

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from config import COLLECTION_NAME, QDRANT_URL, VECTOR_SIZE

logger = logging.getLogger(__name__)

# API key cho Qdrant (set QDRANT_API_KEY trong .env neu Qdrant bat auth)
_API_KEY = os.environ.get("QDRANT_API_KEY", "")
_client_kwargs = {"url": QDRANT_URL}
if _API_KEY:
    _client_kwargs["api_key"] = _API_KEY

client = QdrantClient(**_client_kwargs)


def _check_connection():
    """Kiem tra ket noi Qdrant va in thong tin debug."""
    try:
        cols = client.get_collections().collections
        logger.debug(f"Qdrant OK. Collections: {[c.name for c in cols]}")
    except Exception as e:
        raise ConnectionError(
            f"Khong the ket noi Qdrant tai '{QDRANT_URL}'.\n"
            f"  Qdrant dang chay? (docker ps | grep qdrant)\n"
            f"  QDRANT_API_KEY: {'da set' if _API_KEY else 'CHUA SET - kiem tra .env'}\n"
            f"  Loi: {e}"
        )


def create_collection():
    """Tao (hoac recreate) collection trong Qdrant voi payload indexes.
    
    Thay recreate_collection (deprecated) bang delete + create.
    """
    _check_connection()

    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        print(f"   Collection '{COLLECTION_NAME}' da ton tai -> xoa va tao lai")
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    print(f"   Created '{COLLECTION_NAME}' (size={VECTOR_SIZE}, cosine)")

    index_fields = [
        ("segment",          "keyword"),
        ("service_interest", "keyword"),
        ("dominant_state",   "keyword"),
        ("final_state",      "keyword"),
        ("outcome_label",    "keyword"),
        ("score",            "float"),
        ("weighted_score",   "float"),
        ("outcome_weight",   "float"),
        ("has_closing_cta",  "bool"),
    ]
    for field, field_type in index_fields:
        client.create_payload_index(COLLECTION_NAME, field, field_type)

    info = client.get_collection(COLLECTION_NAME)
    print(f"   Collection '{COLLECTION_NAME}' ready | status={info.status} | indexes={len(index_fields)} (v3.1)")


def push(points):
    """Upsert points vao Qdrant, batch 100 points moi lan."""
    if not points:
        print("   [WARN] Khong co points de push!")
        return

    _check_connection()

    batch_size = 100
    qdrant_points = [
        PointStruct(id=p["id"], vector=p["vector"], payload=p["payload"])
        for p in points
    ]

    total_batches = (len(qdrant_points) + batch_size - 1) // batch_size
    for i in range(0, len(qdrant_points), batch_size):
        batch = qdrant_points[i:i + batch_size]
        batch_num = i // batch_size + 1
        result = client.upsert(collection_name=COLLECTION_NAME, points=batch, wait=True)
        logger.debug(f"Batch {batch_num}/{total_batches}: status={result.status}")

    # Xac nhan so diem thuc te trong collection
    count = client.count(COLLECTION_NAME, exact=True).count
    print(f"   Upserted {len(qdrant_points)} points in {total_batches} batches")
    print(f"   Verified: {count} total points in '{COLLECTION_NAME}'")
    if count != len(qdrant_points):
        logger.warning(f"[WARN] So luong khong khop: upserted={len(qdrant_points)}, actual={count}")
