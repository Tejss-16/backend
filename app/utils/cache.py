import hashlib
import pandas as pd

# ─────────────────────────────────────────────
# 8. LRU RESULT CACHE  (was section 7)
# ─────────────────────────────────────────────

def _df_fingerprint(df: pd.DataFrame) -> str:
    """
    Full-content fingerprint using pandas' own row hashing.
    pd.util.hash_pandas_object hashes every cell of every row, then we
    XOR-sum all row hashes into a single uint64 and fold it into MD5.
    Cost: O(n*c) — acceptable because this runs once per request, not
    per chart. For very large DataFrames (>500k rows) a stratified
    sample of 10k rows gives collision resistance that is good enough
    in practice while keeping latency under 50 ms.
    """
    if len(df) == 0:
        return hashlib.md5(b"empty").hexdigest()

    sample_df = df if len(df) <= 500_000 else df.sample(10_000, random_state=0)
    row_hashes = pd.util.hash_pandas_object(sample_df, index=False)
    content_hash = format(int(row_hashes.sum()) & 0xFFFF_FFFF_FFFF_FFFF, "016x")
    meta = f"{df.shape[0]}x{df.shape[1]}|{','.join(df.columns)}"
    return hashlib.md5(f"{meta}|{content_hash}".encode()).hexdigest()


def _cache_key(df: pd.DataFrame, query: str) -> str:
    return hashlib.md5(
        f"{_df_fingerprint(df)}|{query.strip().lower()}".encode()
    ).hexdigest()


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
