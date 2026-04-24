import logging
logger = logging.getLogger(__name__)
import pandas as pd

# ─────────────────────────────────────────────
# 7. SCORECARD BUILDER  (deterministic — no LLM)
# ─────────────────────────────────────────────

_SCORECARD_MAX = 5   # hard cap per your spec

_AGG_PRIORITY: list[tuple[str, str, str]] = [
    # (query_hint, pandas_agg, label_template)
    # Checked in order; first match wins per column.
    ("revenue",  "sum",  "Total {col}"),
    ("sales",    "sum",  "Total {col}"),
    ("cost",     "sum",  "Total {col}"),
    ("profit",   "sum",  "Total {col}"),
    ("qty",      "sum",  "Total {col}"),
    ("quantity", "sum",  "Total {col}"),
    ("amount",   "sum",  "Total {col}"),
    ("price",    "mean", "Avg {col}"),
    ("rate",     "mean", "Avg {col}"),
    ("score",    "mean", "Avg {col}"),
    ("discount", "mean", "Avg {col}"),
    ("count",    "count","Count of {col}"),
]
_DEFAULT_AGG = ("sum", "Total {col}")


def _pick_agg(col: str) -> tuple[str, str]:
    """
    Choose aggregation + label for a column by name heuristic.
    Returns (pandas_agg, label_template).
    """
    col_lower = col.lower()
    for hint, agg, label in _AGG_PRIORITY:
        if hint in col_lower:
            return agg, label
    return _DEFAULT_AGG


def _fmt_value(v: float) -> str:
    """Human-readable number: 1 234 567 -> '1.23M', 12345 -> '12.3K', else rounded."""
    abs_v = abs(v)
    if abs_v >= 1_000_000:
        return f"{v / 1_000_000:.2f}M"
    if abs_v >= 1_000:
        return f"{v / 1_000:.1f}K"
    if v == int(v):
        return str(int(v))
    return f"{v:.2f}"


class ScorecardBuilder:
    """
    Builds 3–5 KPI scorecards deterministically from numeric columns.
    No LLM involved — fast, token-free, always correct.

    Output format per card:
        {"label": str, "value": str, "raw": float, "column": str, "aggregation": str}
    """

    def __init__(self, df: pd.DataFrame):
        self._df = df
        self._num_cols: list[str] = df.select_dtypes(include="number").columns.tolist()

    # ── public ────────────────────────────────────────────────────────

    def build(self) -> list[dict]:
        """Return a list of scorecard dicts (3–5 items, rendered first in UI)."""
        if not self._num_cols:
            logger.info("ScorecardBuilder: no numeric columns — returning empty")
            return []

        candidates = self._rank_columns()
        cards = []
        for col in candidates[:_SCORECARD_MAX]:
            card = self._build_one(col)
            if card:
                cards.append(card)

        logger.info("ScorecardBuilder: produced %d scorecard(s)", len(cards))
        return cards

    # ── private ───────────────────────────────────────────────────────

    def _rank_columns(self) -> list[str]:
        """
        Rank numeric columns by usefulness:
          1. Columns whose names match a known KPI hint (revenue, sales …) come first.
          2. Tie-break: higher variance = more interesting.
        Drops columns that are effectively IDs (all-unique integers).
        """
        df     = self._df
        ranked = []
        for col in self._num_cols:
            n_unique = df[col].nunique()
            n_rows   = len(df)
            # skip likely ID columns (>95 % unique integers)
            if n_unique / max(n_rows, 1) > 0.95 and pd.api.types.is_integer_dtype(df[col]):
                continue
            hint_score = sum(
                1 for hint, *_ in _AGG_PRIORITY if hint in col.lower()
            )
            variance = float(df[col].var(ddof=0)) if n_unique > 1 else 0.0
            ranked.append((col, hint_score, variance))

        # sort: higher hint_score first, then higher variance
        ranked.sort(key=lambda t: (t[1], t[2]), reverse=True)
        return [col for col, *_ in ranked]

    def _build_one(self, col: str) -> dict | None:
        series = self._df[col].dropna()
        if series.empty:
            return None

        agg, label_tpl = _pick_agg(col)

        try:
            if agg == "sum":
                raw = float(series.sum())
            elif agg == "mean":
                raw = float(series.mean())
            else:           # "count"
                raw = float(series.count())
        except Exception as exc:
            logger.warning("Scorecard failed for column %r: %s", col, exc)
            return None

        label = label_tpl.format(col=col)
        return {
            "label":       label,
            "value":       _fmt_value(raw),
            "raw":         raw,
            "column":      col,
            "aggregation": agg,
        }
