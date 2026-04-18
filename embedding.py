
from openai import OpenAI

client = OpenAI()

MODEL = "text-embedding-3-small"


def embed(text: str) -> list[float]:
    """Embed text thành vector sử dụng OpenAI embedding model.

    Truncate text quá dài để tránh lỗi token limit.
    """
    text = str(text).strip()
    if not text:
        text = "empty"

    # text-embedding-3-small hỗ trợ tối đa 8191 tokens (~32k chars)
    if len(text) > 30000:
        text = text[:30000]

    response = client.embeddings.create(
        model=MODEL,
        input=text,
    )
    return response.data[0].embedding


def embed_batch(texts: list[str], batch_size: int = 100) -> list[list[float]]:
    """Embed nhiều texts cùng lúc để giảm API calls."""
    all_vectors = []
    for i in range(0, len(texts), batch_size):
        batch = [str(t).strip() or "empty" for t in texts[i:i + batch_size]]
        batch = [t[:30000] for t in batch]
        response = client.embeddings.create(model=MODEL, input=batch)
        all_vectors.extend([d.embedding for d in response.data])
    return all_vectors
