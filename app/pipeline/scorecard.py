# app/pipeline/scorecard.py
#
# FIX: Raised _SCORECARD_MAX from 4 → 8 to allow richer dashboards.
# FIX: Added _is_kpi_column guard so non-business columns (StoreArea, Latitude,
#      etc.) are rejected at build time even if the LLM chose them.

import logging
import math
import pandas as pd

logger = logging.getLogger(__name__)

_SCORECARD_MAX = 6   # was 4

# Column name substrings that flag a column as a non-KPI physical/operational
# attribute — never meaningful as a business scorecard metric.
# Must stay in sync with llm_client._NON_KPI_SUBSTRINGS.
_NON_KPI_SUBSTRINGS = frozenset({
    "area", "size", "weight", "height", "width", "length", "depth",
    "latitude", "longitude", "lat", "lon", "lng", "zip", "postal",
    "phone", "fax", "email", "url", "address", "description",
    "notes", "comment", "remark", "flag", "status_code",
})

# Pure identifier substrings — always skip for scorecards.
_ID_SUBSTRINGS = frozenset({"id", "key", "code"})


def _is_kpi_column(col_name: str) -> bool:
    """
    Hard gate: return False for columns that are structurally unfit as KPI
    scorecards regardless of what the LLM requested.

    Rejects:
      - Pure identifier columns (id, key, code)
      - Physical / operational attributes (area, size, lat/lon, zip, phone…)

    Keeps everything else — the LLM and _infer_aggregation handle the rest.
    """
    lower = col_name.lower()
    if any(h in lower for h in _ID_SUBSTRINGS):
        return False
    if any(h in lower for h in _NON_KPI_SUBSTRINGS):
        return False
    return True


def _fmt_value(v: float) -> str:
    """Human-readable: 1_234_567 → '1.23M', 12_345 → '12.3K', else rounded."""
    if not math.isfinite(v):
        return "N/A"
    abs_v = abs(v)
    if abs_v >= 1_000_000:
        return f"{v / 1_000_000:.2f}M"
    if abs_v >= 1_000:
        return f"{v / 1_000:.1f}K"
    if v == int(v):
        return str(int(v))
    return f"{v:.4g}"


class ScorecardBuilder:
    """
    Executes LLM-specified scorecard configs against the actual DataFrame.
    The LLM decides what to show; this class only does the pandas math
    and formats the result.

    Accepts an optional DataFrameStats instance.  When provided, sum/mean/
    min/max aggregations are served from pre-computed values — no extra
    DataFrame scan per scorecard.
    """

    def __init__(self, df: pd.DataFrame, stats=None):
        self._df    = df
        self._stats = stats   # DataFrameStats | None

    def build_from_llm(self, scorecard_configs: list) -> list[dict]:
        if not scorecard_configs:
            logger.debug("ScorecardBuilder: no LLM scorecard configs — returning empty")
            return []

        cards = []
        for cfg in scorecard_configs[:_SCORECARD_MAX]:
            card = self._build_one(cfg)
            if card:
                cards.append(card)

        logger.info("ScorecardBuilder: built %d scorecard(s) from %d LLM configs",
                    len(cards), len(scorecard_configs))
        return cards

    def _build_one(self, cfg) -> dict | None:
        col      = cfg.column
        agg      = cfg.aggregation
        label    = cfg.label or f"{agg.capitalize()} {col}"
        subtitle = getattr(cfg, "subtitle", "") or ""

        # ── Hard KPI guard ────────────────────────────────────────────────────
        # Reject non-business columns (StoreArea, Latitude, ZipCode, etc.)
        # even if the LLM somehow chose them, before doing any DataFrame work.
        if not _is_kpi_column(col):
            logger.warning(
                "Scorecard skip: column %r is not a business KPI "
                "(matched non-KPI pattern). LLM should not have chosen this.",
                col,
            )
            return None

        if col not in self._df.columns:
            logger.warning("Scorecard skip: column %r not in DataFrame", col)
            return None

        # Fast path: use pre-computed stats when available (no extra scan)
        if self._stats is not None and agg in ("sum", "mean", "min", "max"):
            raw_or_none = self._stats.agg(col, agg)
            if raw_or_none is not None:
                return {
                    "label":       label,
                    "value":       _fmt_value(raw_or_none),
                    "raw":         raw_or_none,
                    "column":      col,
                    "aggregation": agg,
                    "subtitle":    subtitle,
                }
            # Column not in stats (e.g. non-numeric) — fall through to
            # the scan-based path which will log the appropriate warning.

        # Fallback path: compute directly (no stats provided, or agg=="count")
        series = self._df[col].dropna()
        if series.empty:
            logger.warning("Scorecard skip: column %r is all-null", col)
            return None

        if not pd.api.types.is_numeric_dtype(series):
            logger.warning("Scorecard skip: column %r is not numeric (dtype=%s)", col, series.dtype)
            return None

        try:
            if agg == "sum":
                raw = float(series.sum())
            elif agg == "mean":
                raw = float(series.mean())
            elif agg == "count":
                raw = float(series.count())
            elif agg == "min":
                raw = float(series.min())
            elif agg == "max":
                raw = float(series.max())
            else:
                logger.warning("Scorecard skip: unknown aggregation %r", agg)
                return None
        except Exception as exc:
            logger.warning("Scorecard compute failed for %r/%r: %s", col, agg, exc)
            return None

        return {
            "label":       label,
            "value":       _fmt_value(raw),
            "raw":         raw,
            "column":      col,
            "aggregation": agg,
            "subtitle":    subtitle,
        }