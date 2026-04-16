from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

client = QdrantClient(url="http://localhost:6333")

def create_collection():
    client.recreate_collection(
        collection_name="conversation_memory",
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )

def push(points):
    client.upsert(
        collection_name="conversation_memory",
        points=points
    )
