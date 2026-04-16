QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "conversation_memory"

EMBEDDING_MODEL = "text-embedding-3-small"

TOP_SCORE_THRESHOLD = 70

SEGMENT_RULES = {
    "LOW": (0, 2_000_000),
    "MID": (2_000_000, 10_000_000),
    "HIGH": (10_000_000, 30_000_000),
    "VIP": (30_000_000, float("inf"))
}
