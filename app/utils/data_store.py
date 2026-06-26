#app/utils/data_store.py

import logging
import math
import re

import pandas as pd
import uuid

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Lazy import helpers — column_utils lives in the same package.
# We import lazily to avoid a circular-import if column_utils ever imports
# data_store in the future, and to keep startup overhead zero.
# ─────────────────────────────────────────────────────────────────────────────

def _get_meaningful_numeric_cols(df: pd.DataFrame) -> list[str]:
    from app.utils.column_utils import _meaningful_numeric_cols
    return _meaningful_numeric_cols(df)


def _get_is_id_col(name: str) -> bool:
    from app.utils.column_utils import _is_id_col
    return _is_id_col(name)


def _df_fingerprint(df: pd.DataFrame) -> str:
    from app.utils.cache import _df_fingerprint as _fp
    return _fp(df)


# ─────────────────────────────────────────────────────────────────────────────
# DATASET METADATA CACHE
# Stores all expensive per-dataset analytics keyed by dataset_id.
# Populated once on DataStore.save(); never recomputed unless the dataset
# is overwritten (same dataset_id, new DataFrame).
# ─────────────────────────────────────────────────────────────────────────────

_DATE_NAME_HINTS = (
    "date", "time", "period", "order_date", "ship", "created", "updated"
)
_REVENUE_HINTS = ("revenue", "sales", "amount", "income", "gross")
_PROFIT_HINTS  = ("profit", "margin", "net")

# Column name substrings that mark a column as a non-KPI physical/operational
# attribute — never meaningful as a business scorecard or chart metric.
# Must stay in sync with llm_client._NON_KPI_SUBSTRINGS and scorecard._NON_KPI_SUBSTRINGS.
_NON_KPI_SUBSTRINGS = frozenset({
    "area", "size", "weight", "height", "width", "length", "depth",
    "latitude", "longitude", "lat", "lon", "lng", "zip", "postal",
    "phone", "fax", "email", "url", "address", "description",
    "notes", "comment", "remark", "flag", "status_code",
})
_ID_SUBSTRINGS = frozenset({"id", "key", "code"})


def _is_kpi_column(col_name: str) -> bool:
    """Return False for physical/operational attributes and pure identifiers."""
    lower = col_name.lower()
    if any(h in lower for h in _ID_SUBSTRINGS):
        return False
    if any(h in lower for h in _NON_KPI_SUBSTRINGS):
        return False
    return True

def _fmt_meta_val(v: float) -> str:
    """Compact number formatter for the dataset profile string."""
    try:
        if not math.isfinite(v):
            return "N/A"
        abs_v = abs(v)
        if abs_v >= 1_000_000_000:
            return f"{v / 1_000_000_000:.2f}B"
        if abs_v >= 1_000_000:
            return f"{v / 1_000_000:.2f}M"
        if abs_v >= 1_000:
            return f"{v / 1_000:.1f}K"
        return f"{v:.4g}"
    except Exception:
        return str(v)


def _find_date_col(df: pd.DataFrame) -> str | None:
    """Lightweight date-column finder used only during metadata build."""
    import warnings as _warnings
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
    for col in df.columns:
        if any(h in col.lower() for h in _DATE_NAME_HINTS) and df[col].dtype == object:
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore", UserWarning)
                if pd.to_datetime(df[col], errors="coerce").notna().mean() >= 0.5:
                    return col
    for col in df.select_dtypes(include="object").columns:
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", UserWarning)
            if pd.to_datetime(df[col], errors="coerce").notna().mean() >= 0.7:
                return col
    return None


def _compute_dataset_metadata(df: pd.DataFrame) -> dict:
    """
    Compute all expensive per-dataset analytics in a single pass.
    Called exactly once per (dataset_id, DataFrame) pair — never again
    until the dataset is replaced via DataStore.save().

    Returns a dict with keys:
      profile_str   – the full [DATASET PROFILE] string sent to the LLM
      num_cols      – list[str]  meaningful numeric column names
      cat_cols      – list[str]  categorical column names (cardinality ≤ 30)
      dt_cols       – list[str]  date/time column names
      date_col      – str | None  primary date column
      primary_metric– str | None  most likely revenue/sales column
      profit_metric – str | None  most likely profit/margin column
      primary_cat   – str | None  first categorical column with ≤ 20 unique values
      cardinality   – dict[str, int]  {col: nunique} for all categorical cols
    """
    t_start = __import__("time").perf_counter()

    # ── Column classifications ────────────────────────────────────────────────
    num_cols  = _get_meaningful_numeric_cols(df)
    date_col  = _find_date_col(df)

    all_cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if date_col and date_col in all_cat_cols:
        all_cat_cols = [c for c in all_cat_cols if c != date_col]

    # Compute cardinalities once for all categorical columns
    cardinality: dict[str, int] = {}
    for col in all_cat_cols:
        try:
            cardinality[col] = int(df[col].nunique())
        except Exception:
            cardinality[col] = 0

    cat_cols    = [c for c in all_cat_cols if cardinality.get(c, 0) <= 30]
    dt_cols     = (
        ([date_col] if date_col else []) +
        [
            c for c in df.columns
            if c != date_col and (
                "date" in c.lower() or "time" in c.lower()
                or str(df[c].dtype).startswith("datetime")
            )
        ]
    )

    # Primary business metrics — identified by column-name heuristics
    primary_metric = next(
        (c for c in num_cols if any(h in c.lower() for h in _REVENUE_HINTS)), None
    ) or (num_cols[0] if num_cols else None)

    profit_metric = next(
        (c for c in num_cols if any(h in c.lower() for h in _PROFIT_HINTS)), None
    )

    primary_cat = next(
        (c for c in cat_cols if cardinality.get(c, 0) <= 20), None
    )

    # ── Build the LLM profile string ─────────────────────────────────────────
    lines = ["[DATASET PROFILE]"]
    lines.append(f"Rows: {len(df):,}")

    # Date range
    if date_col:
        try:
            parsed = (
                df[date_col]
                if pd.api.types.is_datetime64_any_dtype(df[date_col])
                else pd.to_datetime(df[date_col], errors="coerce")
            )
            yr_min = parsed.dropna().dt.year.min()
            yr_max = parsed.dropna().dt.year.max()
            if yr_min and yr_max:
                lines.append(f"Date range ({date_col}): {int(yr_min)} – {int(yr_max)}")
        except Exception:
            pass

    # Numeric metrics — show sum, mean, min, max for each column.
    # Also flag columns where mean is significantly lower than max: these are
    # likely per-row rates (unit price, discount %) rather than pre-summed totals,
    # so the LLM should treat their raw .sum() with caution.
    if num_cols:
        lines.append("Numeric metrics (KPI=use for scorecards/charts, ATTR=skip for scorecards):")
        for col in num_cols[:12]:
            try:
                s     = df[col].dropna()
                total = s.sum()
                mean  = s.mean()
                tag   = "KPI" if _is_kpi_column(col) else "ATTR"
                lines.append(
                    f"  [{tag}] {col}: sum={_fmt_meta_val(total)}, "
                    f"mean={_fmt_meta_val(mean)}, "
                    f"min={_fmt_meta_val(float(s.min()))}, "
                    f"max={_fmt_meta_val(float(s.max()))}"
                )
            except Exception:
                pass

    # Categorical columns
    if all_cat_cols:
        lines.append("Categorical columns:")
        for col in all_cat_cols[:10]:
            try:
                n_unique = cardinality.get(col, df[col].nunique())
                top_vals = df[col].value_counts().head(5).index.tolist()
                top_str  = ", ".join(f'"{v}"' for v in top_vals)
                lines.append(f"  {col}: {n_unique} unique — top: {top_str}")
            except Exception:
                pass

    profile_str = "\n".join(lines)

    elapsed = __import__("time").perf_counter() - t_start
    logger.info(
        "DatasetMetadataCache: computed metadata in %.3fs  "
        "(rows=%d, num_cols=%d, cat_cols=%d)",
        elapsed, len(df), len(num_cols), len(all_cat_cols),
    )

    return {
        "profile_str":     profile_str,
        "profile_version": 2,           # bump when profile format changes — invalidates stale cache
        "num_cols":        num_cols,
        "cat_cols":        cat_cols,
        "dt_cols":         dt_cols,
        "date_col":        date_col,
        "primary_metric":  primary_metric,
        "profit_metric":   profit_metric,
        "primary_cat":     primary_cat,
        "cardinality":     cardinality,
    }


class DatasetMetadataCache:
    """
    Stores computed dataset metadata (column classifications, cardinalities,
    aggregates, and the LLM profile string) keyed by dataset_id.

    Invalidated automatically when DataStore.save() is called with the same
    dataset_id — i.e. the moment the underlying DataFrame changes, the stale
    metadata is evicted and will be recomputed on next access.
    """

    def __init__(self) -> None:
        self._store: dict[str, dict] = {}

    def get(self, dataset_id: str) -> dict | None:
        return self._store.get(dataset_id)

    def set(self, dataset_id: str, metadata: dict) -> None:
        self._store[dataset_id] = metadata

    def invalidate(self, dataset_id: str) -> None:
        self._store.pop(dataset_id, None)

    _CURRENT_PROFILE_VERSION = 2   # must match _compute_dataset_metadata

    def get_or_compute(self, dataset_id: str, df: pd.DataFrame) -> dict:
        """
        Return cached metadata if present and current, otherwise compute,
        cache, and return.

        Checks `profile_version` so that a cache entry produced by an older
        code version (e.g. before KPI/ATTR tagging was added) is automatically
        recomputed rather than served stale.
        """
        cached = self._store.get(dataset_id)
        if cached is not None:
            if cached.get("profile_version") == self._CURRENT_PROFILE_VERSION:
                logger.debug(
                    "DatasetMetadataCache: cache hit for dataset_id=%s", dataset_id
                )
                return cached
            # Stale version — fall through to recompute
            logger.info(
                "DatasetMetadataCache: stale profile_version for dataset_id=%s "
                "(got %r, want %r) — recomputing",
                dataset_id,
                cached.get("profile_version"),
                self._CURRENT_PROFILE_VERSION,
            )
        else:
            logger.info(
                "DatasetMetadataCache: cache miss for dataset_id=%s — computing",
                dataset_id,
            )
        metadata = _compute_dataset_metadata(df)
        self._store[dataset_id] = metadata
        return metadata


# ─────────────────────────────────────────────────────────────────────────────
# DataStore
# ─────────────────────────────────────────────────────────────────────────────

class DataStore:
    def __init__(self):
        self._store: dict[str, pd.DataFrame] = {}
        self._fingerprints: dict[str, str]   = {}

    def save(self, df: pd.DataFrame, dataset_id: str = None) -> str:
        """
        Save a DataFrame to the in-memory store.

        Computes and stores the content fingerprint once here — never again
        during request handling.  All subsequent cache-key lookups call
        get_fingerprint() instead of re-hashing the DataFrame.

        Side-effect: invalidates any stale metadata for this dataset_id.
        """
        if dataset_id is None:
            dataset_id = str(uuid.uuid4())
        self._store[dataset_id]        = df
        self._fingerprints[dataset_id] = _df_fingerprint(df)
        # Evict stale metadata — will be recomputed on next access
        dataset_metadata_cache.invalidate(dataset_id)
        return dataset_id

    def get(self, dataset_id: str) -> pd.DataFrame | None:
        return self._store.get(dataset_id)

    def get_fingerprint(self, dataset_id: str) -> str | None:
        """Return the pre-computed content fingerprint, or None if unknown."""
        return self._fingerprints.get(dataset_id)


# singleton instances
data_store = DataStore()
dataset_metadata_cache = DatasetMetadataCache()