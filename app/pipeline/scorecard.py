# app/pipeline/scorecard.py
#
# FIX: Raised _SCORECARD_MAX from 4 → 8 to allow richer dashboards.

import logging
import math
import pandas as pd

logger = logging.getLogger(__name__)

_SCORECARD_MAX = 8   # was 4


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
    """

    def __init__(self, df: pd.DataFrame):
        self._df = df

    def build_from_llm(self, scorecard_configs: list) -> list[dict]:
        if not scorecard_configs:
            logger.info("ScorecardBuilder: no LLM scorecard configs — returning empty")
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
        col   = cfg.column
        agg   = cfg.aggregation
        label = cfg.label or f"{agg.capitalize()} {col}"

        if col not in self._df.columns:
            logger.warning("Scorecard skip: column %r not in DataFrame", col)
            return None

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
        }