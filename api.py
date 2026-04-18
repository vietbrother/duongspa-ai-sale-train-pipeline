
from fastapi import FastAPI, Query
from retrieval import search
from prompt_builder import build_prompt

app = FastAPI(title="DuongSpa AI Sales Pipeline API")


@app.get("/search")
def search_endpoint(
    q: str = Query(..., description="Search query"),
    segment: str = Query("MID", description="Customer segment: LOW/MID/HIGH/VIP"),
    top_k: int = Query(5, description="Number of results"),
):
    """Search similar conversations từ Qdrant."""
    results = search(q, segment=segment, top_k=top_k)
    return {"query": q, "segment": segment, "results": results}


@app.get("/chat")
def chat(
    q: str = Query(..., description="User question"),
    segment: str = Query("MID", description="Customer segment"),
    model: str = Query("openai", description="LLM provider"),
):
    """Build prompt cho chatbot response."""
    ctx = search(q, segment=segment)
    prompt = build_prompt(segment=segment, context=ctx, query=q, model=model)
    return {"context": ctx, "prompt": prompt}
