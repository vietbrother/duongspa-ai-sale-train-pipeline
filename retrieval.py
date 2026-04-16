
from qdrant_client import QdrantClient
from embedding import embed
from datetime import datetime

client = QdrantClient(url="http://localhost:6333")

def intent_detect(q):
    if "giá" in q: return "ask_price"
    if "đau" in q: return "concern"
    return "general"

def time_decay(created):
    if not created: return 1
    days = (datetime.now() - datetime.fromisoformat(created)).days
    return 1/(1+days/30)

def hybrid_score(hit, intent):
    p = hit.payload
    score = hit.score
    score += p.get("reward",0)*0.3
    if p.get("intent")==intent: score*=1.3
    score *= time_decay(p.get("created_at"))
    return score

def search(query, service=None):
    vec = embed(query)
    results = client.search(
        collection_name="conversation_memory",
        query_vector=vec,
        limit=20
    )
    intent = intent_detect(query)
    scored = [(hybrid_score(r,intent), r) for r in results]
    scored.sort(reverse=True, key=lambda x:x[0])
    return [r.payload["text"] for _,r in scored[:5]]
