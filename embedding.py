
import logging
import time

from config import EMBED_ENABLED, EMBED_LOCAL_MODEL, EMBED_PROVIDER, VECTOR_SIZE

logger = logging.getLogger(__name__)

_OPENAI_MODEL = "text-embedding-3-small"
_openai_client = None
_local_model = None

# Delay (seconds) giua cac batch de tranh Rate Limit (429).
# Tang len neu van gap loi, vi du EMBED_BATCH_DELAY=2.0
import os as _os
EMBED_BATCH_DELAY = float(_os.environ.get("EMBED_BATCH_DELAY", "0.5"))
# So lan retry khi gap RateLimitError / 429
EMBED_MAX_RETRIES = int(_os.environ.get("EMBED_MAX_RETRIES", "3"))
# He so backoff: lan 1 = delay*2, lan 2 = delay*4, ...
EMBED_BACKOFF_FACTOR = float(_os.environ.get("EMBED_BACKOFF_FACTOR", "2.0"))


# ---------------------------------------------------------------------------
# Provider helpers
# ---------------------------------------------------------------------------

def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI()
    return _openai_client


def _get_local_model():
    global _local_model
    if _local_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _local_model = SentenceTransformer(EMBED_LOCAL_MODEL)
            logger.info(f"Loaded local embedding model: {EMBED_LOCAL_MODEL}")
        except ImportError:
            raise RuntimeError(
                "sentence-transformers chua duoc cai. Chay: pip install sentence-transformers"
            )
    return _local_model


def _zero_vector() -> list:
    return [0.0] * VECTOR_SIZE


def _normalize(text: str, max_chars: int = 30000) -> str:
    text = str(text).strip()
    return (text or "empty")[:max_chars]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def embed(text: str) -> list:
    """Embed mot text thanh vector.

    - EMBED_ENABLED=false  -> zero vector (khong goi LLM).
    - Loi API / key sai    -> log warning + zero vector (khong crash pipeline).
    - EMBED_PROVIDER=local -> dung sentence-transformers (chay offline).
    """
    if not EMBED_ENABLED:
        return _zero_vector()

    text = _normalize(text)
    try:
        if EMBED_PROVIDER == "local":
            return _get_local_model().encode(text).tolist()
        else:
            client = _get_openai_client()
            response = client.embeddings.create(model=_OPENAI_MODEL, input=text)
            return response.data[0].embedding
    except Exception as e:
        logger.warning(f"[embed] Loi embedding provider={EMBED_PROVIDER}: {e}. Dung zero vector.")
        return _zero_vector()


def embed_batch(texts: list, batch_size: int = 100) -> list:
    """Embed nhieu texts cung luc de giam API calls.

    - EMBED_ENABLED=false -> tra ve danh sach zero vector.
    - Rate limit (429)    -> tu dong retry voi exponential backoff.
    - Delay giua batch    -> EMBED_BATCH_DELAY (mac dinh 0.5s).
    - Loi khong the retry -> fallback tung item, item loi dung zero vector.
    """
    if not EMBED_ENABLED:
        return [_zero_vector() for _ in texts]

    normalized = [_normalize(t) for t in texts]
    try:
        if EMBED_PROVIDER == "local":
            model = _get_local_model()
            result = []
            bs = min(batch_size, 64)
            for i in range(0, len(normalized), bs):
                result.extend(model.encode(normalized[i:i + bs]).tolist())
            return result
        else:
            client = _get_openai_client()
            result = []
            total_batches = (len(normalized) + batch_size - 1) // batch_size
            for batch_idx, i in enumerate(range(0, len(normalized), batch_size)):
                batch = normalized[i:i + batch_size]
                # --- retry voi exponential backoff khi gap rate limit ---
                for attempt in range(1, EMBED_MAX_RETRIES + 1):
                    try:
                        response = client.embeddings.create(model=_OPENAI_MODEL, input=batch)
                        result.extend([d.embedding for d in response.data])
                        break
                    except Exception as e:
                        is_rate_limit = (
                            "429" in str(e)
                            or "rate_limit" in str(e).lower()
                            or "RateLimitError" in type(e).__name__
                        )
                        if attempt < EMBED_MAX_RETRIES and is_rate_limit:
                            wait = EMBED_BATCH_DELAY * (EMBED_BACKOFF_FACTOR ** attempt)
                            logger.warning(
                                f"[embed_batch] Rate limit batch {batch_idx+1}/{total_batches} "
                                f"(attempt {attempt}/{EMBED_MAX_RETRIES}). Cho {wait:.1f}s..."
                            )
                            time.sleep(wait)
                        else:
                            raise
                # --- delay co dinh giua cac batch (tranh rate limit chu dong) ---
                if EMBED_BATCH_DELAY > 0 and (i + batch_size) < len(normalized):
                    logger.debug(
                        f"[embed_batch] Batch {batch_idx+1}/{total_batches} done. "
                        f"Delay {EMBED_BATCH_DELAY}s..."
                    )
                    time.sleep(EMBED_BATCH_DELAY)
            return result
    except Exception as e:
        logger.warning(
            f"[embed_batch] Loi batch embedding provider={EMBED_PROVIDER}: {e}. "
            "Fallback sang embed tung item..."
        )
        return [embed(t) for t in normalized]
