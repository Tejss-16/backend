import hashlib
import pandas as pd

# ─────────────────────────────────────────────
# 8. LRU RESULT CACHE  (was section 7)
# ─────────────────────────────────────────────

def _df_fingerprint(df: pd.DataFrame) -> str:
    """
    Full-content fingerprint using pandas' own row hashing.
    Called ONCE at upload time and stored in DataStore.
    Never called again during request handling.
    """
    if len(df) == 0:
        return hashlib.md5(b"empty").hexdigest()

    sample_df = df if len(df) <= 500_000 else df.sample(10_000, random_state=0)
    row_hashes = pd.util.hash_pandas_object(sample_df, index=False)
    content_hash = format(int(row_hashes.sum()) & 0xFFFF_FFFF_FFFF_FFFF, "016x")
    meta = f"{df.shape[0]}x{df.shape[1]}|{','.join(df.columns)}"
    return hashlib.md5(f"{meta}|{content_hash}".encode()).hexdigest()


def _cache_key_from_fingerprint(fingerprint: str, query: str) -> str:
    """
    Build a cache key from a pre-computed fingerprint.
    Zero DataFrame access — just two string ops and an MD5.
    This is the primary cache key path used during request handling.
    """
    return hashlib.md5(
        f"{fingerprint}|{query.strip().lower()}".encode()
    ).hexdigest()


def _cache_key(df: pd.DataFrame, query: str) -> str:
    """
    Fallback: compute fingerprint inline.
    Only used when dataset_id is unavailable (e.g. tests, legacy callers).
    In normal request handling, use _cache_key_from_fingerprint instead.
    """
    return _cache_key_from_fingerprint(_df_fingerprint(df), query)


class _LRUCache:
    def __init__(self, max_size: int = 64):
        self._store: dict = {}
        self._max = max_size

    def get(self, key: str):
        return self._store.get(key)

    def set(self, key: str, value) -> None:
        if len(self._store) >= self._max:
            del self._store[next(iter(self._store))]
        self._store[key] = value


_result_cache = _LRUCache(max_size=64)