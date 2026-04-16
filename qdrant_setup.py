
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

client = QdrantClient(url="http://localhost:6333")

def setup():
    client.recreate_collection(
        collection_name="conversation_memory",
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
    )
    client.create_payload_index("conversation_memory", "segment", "keyword")
    client.create_payload_index("conversation_memory", "service", "keyword")
    client.create_payload_index("conversation_memory", "score", "float")

if __name__ == "__main__":
    setup()
