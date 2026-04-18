from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

from config import QDRANT_URL, COLLECTION_NAME, VECTOR_SIZE

client = QdrantClient(url=QDRANT_URL)


def create_collection():
    """Tạo (hoặc recreate) collection trong Qdrant với payload indexes."""
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    # Tạo index trên payload fields thường dùng để filter
    client.create_payload_index(COLLECTION_NAME, "segment", "keyword")
    client.create_payload_index(COLLECTION_NAME, "service_interest", "keyword")
    client.create_payload_index(COLLECTION_NAME, "score", "float")
    client.create_payload_index(COLLECTION_NAME, "has_closing_cta", "bool")
    print(f"   Collection '{COLLECTION_NAME}' created with payload indexes")


def push(points):
    """Upsert points vào Qdrant, batch 100 points mỗi lần."""
    batch_size = 100
    qdrant_points = [
        PointStruct(
            id=p["id"],
            vector=p["vector"],
            payload=p["payload"],
        )
        for p in points
    ]

    for i in range(0, len(qdrant_points), batch_size):
        batch = qdrant_points[i:i + batch_size]
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=batch,
            wait=True,
        )
    print(f"   Upserted {len(qdrant_points)} points in {(len(qdrant_points)-1)//batch_size + 1} batches")
