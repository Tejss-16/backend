# FIX: Updated type-specific guards to handle new chart types:
#      area, stacked_bar, grouped_bar, heatmap, bubble, funnel, treemap, waterfall

import logging
logger = logging.getLogger(__name__)
import pandas as pd
from app.schemas.chart_schema import ChartConfigSchema

# ─────────────────────────────────────────────
# 3. CONFIG NORMALIZER  (business rules, immutable output)
# ─────────────────────────────────────────────

# Types that behave like bar/line for normalisation purposes
_BAR_LIKE    = {"bar", "stacked_bar", "grouped_bar", "funnel", "treemap", "waterfall", "area"}
_SCATTER_LIKE = {"scatter", "bubble"}

class ChartConfigNormalizer:
    """
    Pure normalizer — never mutates the source DataFrame.
    Synthetic columns are passed as a Series in cfg["_synthetic"]
    so DataTransformer can join them on demand.
    """

    def __init__(self, df: pd.DataFrame):
        self._df         = df
        self._columns    = set(df.columns)
        self._num_cols   = set(df.select_dtypes(include="number").columns)
        self._cardinality: dict[str, int] = {}

    def _card(self, col: str) -> int:
        if col not in self._cardinality:
            self._cardinality[col] = self._df[col].nunique()
        return self._cardinality[col]

    def normalize(self, schema: ChartConfigSchema) -> dict | None:
        chart_type = schema.type
        x          = schema.x
        y          = schema.y
        color      = schema.color
        agg        = schema.aggregation

        if x not in self._columns:
            logger.debug("Dropping chart: x=%r not in columns", x)
            return None

        # ── pie without y: build synthetic count Series ───────────────────────
        synthetic: pd.Series | None = None
        synthetic_name: str | None  = None

        if chart_type == "pie" and y is None:
            synthetic_name = f"_count_{x}"
            synthetic = pd.Series(1, index=self._df.index, name=synthetic_name, dtype="int64")
            y   = synthetic_name
            agg = "count"
            logger.debug("Pie chart: synthetic count column %r for x=%r", y, x)

        # effective column set = real columns + any synthetic
        effective_cols = self._columns | ({synthetic_name} if synthetic_name else set())
        effective_nums = self._num_cols | ({synthetic_name} if synthetic_name else set())

        # ── per-type guards ───────────────────────────────────────────────────

        if chart_type == "box":
            if y not in self._num_cols:
                return None

        if chart_type == "histogram":
            if x not in self._num_cols:
                return None
        elif chart_type == "heatmap":
            # heatmap needs y as a category column and z (stored in schema.z) as numeric
            z_col = schema.z if hasattr(schema, "z") else None
            if y is None or y not in effective_cols:
                logger.debug("Dropping heatmap: y=%r not in columns", y)
                return None
            if z_col and z_col not in self._num_cols:
                logger.debug("Dropping heatmap: z=%r is not numeric", z_col)
                return None
        elif chart_type == "bubble":
            # bubble needs y as numeric
            if y is None or y not in effective_nums:
                logger.debug("Dropping bubble: y=%r not numeric", y)
                return None
            if x not in self._num_cols:
                logger.debug("Dropping bubble: x=%r not numeric", x)
                return None
        elif chart_type not in ("histogram",):
            if y is None or y not in effective_cols:
                logger.debug("Dropping chart: y=%r not in columns", y)
                return None

        if chart_type in _SCATTER_LIKE:
            if self._card(x) < 5 or (y and self._card(y) < 5):
                return None

        if chart_type in (_BAR_LIKE | {"pie"}) and y and y not in effective_nums:
            return None

        if color and (color not in self._columns or self._card(color) > 6):
            color = None

        if chart_type in (_BAR_LIKE | {"line", "pie", "area"}) and agg == "none":
            agg = "sum"

        # ── build normalised config dict ──────────────────────────────────────
        result = {
            "type":             chart_type,
            "x":                x,
            "y":                y,
            "color":            color,
            "aggregation":      agg,
            "time_granularity": schema.time_granularity,
            "layout_size":      schema.layout_size,
            "title":            schema.title,
            "limit_top":        chart_type in ("bar", "pie", "stacked_bar", "grouped_bar", "treemap", "funnel")
                                and self._card(x) > min(30, max(10, len(self._df) // 5)),
            "_synthetic":       synthetic,
        }

        # Pass z and size through for chart types that use them
        if hasattr(schema, "z") and schema.z:
            result["z"] = schema.z
        if hasattr(schema, "size") and schema.size:
            result["size"] = schema.size

        return result
