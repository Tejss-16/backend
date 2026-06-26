# app/utils/df_stats.py
#
# Single-pass DataFrame statistics computed once per request.
# Passed to ChartConfigNormalizer, ScorecardBuilder, and ChartBuilder
# so that nunique(), sum(), mean(), min(), max(), and select_dtypes()
# are never repeated across modules for the same DataFrame.

import logging
import math
import pandas as pd
from app.utils.column_utils import _meaningful_numeric_cols

logger = logging.getLogger(__name__)


class DataFrameStats:
    """
    Holds pre-computed per-column statistics for one DataFrame.

    Computed ONCE per ChartGenerator instantiation (or once per
    filtered-data variant when a date filter is applied).

    Attributes
    ----------
    num_cols : list[str]
        Meaningful numeric column names (IDs excluded).
    columns  : set[str]
        All column names.
    cardinality : dict[str, int]
        {col: nunique()} for every column that has been queried.
        Populated lazily on first access via card().
    col_sums, col_means, col_mins, col_maxes : dict[str, float]
        Pre-computed aggregates for all meaningful numeric columns.
    """

    __slots__ = (
        "_df", "num_cols", "columns",
        "_cardinality",
        "col_sums", "col_means", "col_mins", "col_maxes",
    )

    def __init__(self, df: pd.DataFrame):
        self._df       = df
        self.columns   = set(df.columns)
        self.num_cols  = _meaningful_numeric_cols(df)

        # Pre-compute aggregates for all numeric columns in one pass each.
        # These are O(n) but run once; accessing .sum() on a pre-sliced
        # numeric sub-DataFrame is faster than per-column calls later.
        self._cardinality: dict[str, int] = {}

        col_sums:  dict[str, float] = {}
        col_means: dict[str, float] = {}
        col_mins:  dict[str, float] = {}
        col_maxes: dict[str, float] = {}

        if self.num_cols:
            num_df = df[self.num_cols].dropna(how="all")
            for col in self.num_cols:
                s = num_df[col].dropna()
                if s.empty:
                    continue
                try:
                    col_sums[col]  = float(s.sum())
                    col_means[col] = float(s.mean())
                    col_mins[col]  = float(s.min())
                    col_maxes[col] = float(s.max())
                except Exception:
                    pass

        self.col_sums  = col_sums
        self.col_means = col_means
        self.col_mins  = col_mins
        self.col_maxes = col_maxes

        logger.debug(
            "DataFrameStats: %d rows, %d numeric cols, aggregates pre-computed",
            len(df), len(self.num_cols),
        )

    # ── Cardinality (lazy — only computed for columns actually used) ──────────

    def card(self, col: str) -> int:
        """Return nunique() for col, computing and caching on first call."""
        if col not in self._cardinality:
            try:
                self._cardinality[col] = int(self._df[col].nunique())
            except Exception:
                self._cardinality[col] = 0
        return self._cardinality[col]

    # ── Convenience accessors ─────────────────────────────────────────────────

    def agg(self, col: str, func: str) -> float | None:
        """
        Return a pre-computed aggregate for a numeric column.
        func: "sum" | "mean" | "min" | "max"
        Returns None if the column is not numeric or has no data.
        """
        lookup = {
            "sum":  self.col_sums,
            "mean": self.col_means,
            "min":  self.col_mins,
            "max":  self.col_maxes,
        }
        store = lookup.get(func)
        if store is None:
            return None
        return store.get(col)

    def series(self, col: str) -> pd.Series:
        """Return the non-null series for col (no copy — read-only use only)."""
        return self._df[col].dropna()