# FIX: Updated type-specific guards to handle new chart types:
#      area, stacked_bar, grouped_bar, heatmap, bubble, funnel, treemap, waterfall

import logging
logger = logging.getLogger(__name__)
import pandas as pd
from app.schemas.chart_schema import ChartConfigSchema
from app.utils.column_utils import _meaningful_numeric_cols, _is_id_col

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
        self._df          = df
        self._columns     = set(df.columns)
        # FIX 1: use _meaningful_numeric_cols so ID columns are excluded globallyFID
        self._num_cols    = set(_meaningful_numeric_cols(df))
        self._cardinality: dict[str, int] = {}
        logger.debug(
            "ChartConfigNormalizerFixed: meaningful numeric cols = %s",
            sorted(self._num_cols),
        )

    def _card(self, col: str) -> int:
        if col not in self._cardinality:
            self._cardinality[col] = self._df[col].nunique()
        return self._cardinality[col]
    
    def _resolve_heatmap_z(
        self,
        z_col: str | None,
        x: str,
        y: str,
        color: str | None,
    ) -> str | None:
         # If the LLM gave us a z that passes validation, use it.
        if z_col and z_col in self._num_cols:
            return z_col
 
        # Auto-select: prefer columns that are not the axis or color columns.
        candidates = [
            c for c in self._num_cols
            if c not in {x, y, color}
        ]
        # Wider fallback: any meaningful numeric column except axes.
        if not candidates:
            candidates = [c for c in self._num_cols if c not in {x, y}]
        # Last resort: any meaningful numeric column at all.
        if not candidates:
            candidates = list(self._num_cols)
 
        if not candidates:
            # FIX 2: hard failure — no ID column sneaks through as a fake z.
            logger.warning(
                "Dropping heatmap: no meaningful numeric z column available "
                "(all numeric columns are identifiers or axes)"
            )
            return None
 
        chosen = candidates[0]
        logger.info("Heatmap: auto-selected z=%r from candidates %s", chosen, candidates)
        return chosen

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
        
        # In ChartConfigNormalizer.normalize(), replace the heatmap block:
        elif chart_type == "heatmap":
            z_col = schema.z if hasattr(schema, "z") else None

            # x must be a column that exists
            if x not in self._columns:
                logger.debug("Dropping heatmap: x=%r not in columns", x)
                return None
            # y must exist
            if y is None or y not in self._columns:
                logger.debug("Dropping heatmap: y=%r not in columns", y)
                return None
            if self._card(x) > 50:
                logger.debug("Dropping heatmap: x=%r has too many unique values (%d)", x, self._card(x))
                return None
            if self._card(y) > 50:
                logger.debug("Dropping heatmap: y=%r has too many unique values (%d)", y, self._card(y))
                return None
            # In the heatmap elif block, after the cardinality checks:
            if _is_id_col(x):
                logger.debug("Dropping heatmap: axis=x col=%r flagged as ID", x)
                return None
            if _is_id_col(y):
                logger.debug("Dropping heatmap: axis=y col=%r flagged as ID", y)
                return None
            # ❗ HARDENING: block numeric axes for heatmap
            if x in self._num_cols:
                logger.debug("Dropping heatmap: x=%r is numeric (invalid axis)", x)
                return None

            if y in self._num_cols:
                logger.debug("Dropping heatmap: y=%r is numeric (invalid axis)", y)
                return None

            # z must be numeric; if absent, try to auto-select one
            z_col = self._resolve_heatmap_z(z_col, x, y, color)
            if z_col is None:
                return None
            if z_col not in self._num_cols:
                logger.debug("Dropping heatmap: z=%r is not a valid numeric metric", z_col)
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
        if chart_type == "heatmap":
            result["z"] = z_col
        else:
            result["z"] = getattr(schema, "z", None)  # None is fine, builder handles it
        result["size"] = getattr(schema, "size", None)

        return result