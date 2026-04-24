import logging
logger = logging.getLogger(__name__)
import pandas as pd
from app.schemas.chart_schema import ChartConfigSchema

# ─────────────────────────────────────────────
# 3. CONFIG NORMALIZER  (business rules, immutable output)
# ─────────────────────────────────────────────

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

        # ── pie without y: build synthetic count Series, never touch self._df ──
        synthetic: pd.Series | None = None
        synthetic_name: str | None  = None

        if chart_type == "pie" and y is None:
            synthetic_name = f"_count_{x}"
            # A Series of 1s — transformer will groupby-sum it into value_counts
            synthetic = pd.Series(1, index=self._df.index, name=synthetic_name, dtype="int64")
            y   = synthetic_name
            agg = "count"
            logger.debug("Pie chart: synthetic count column %r for x=%r (no mutation)", y, x)
        # ──────────────────────────────────────────────────────────────────────

        # effective column set = real columns + any synthetic
        effective_cols = self._columns | ({synthetic_name} if synthetic_name else set())
        effective_nums = self._num_cols | ({synthetic_name} if synthetic_name else set())

        if chart_type != "histogram" and (y is None or y not in effective_cols):
            logger.debug("Dropping chart: y=%r not in columns", y)
            return None

        if chart_type == "histogram" and x not in self._num_cols:
            return None
        if chart_type == "scatter" and (self._card(x) < 5 or (y and self._card(y) < 5)):
            return None
        if chart_type in ("bar", "pie") and y and y not in effective_nums:
            return None

        if color and (color not in self._columns or self._card(color) > 6):
            color = None

        if chart_type in ("bar", "pie", "line") and agg == "none":
            agg = "sum"

        return {
            "type":             chart_type,
            "x":                x,
            "y":                y,
            "color":            color,
            "aggregation":      agg,
            "time_granularity": schema.time_granularity,
            "layout_size":      schema.layout_size,
            "title":            schema.title,
            "limit_top":        chart_type in ("bar", "pie") and self._card(x) > 20,
            # synthetic Series passed through config — never written to source df
            "_synthetic":       synthetic,
        }
