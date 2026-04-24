from typing import Literal, Optional
from pydantic import BaseModel, field_validator, model_validator, ValidationError
import logging
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# 1. PYDANTIC SCHEMA  (strict validation of LLM output)
# ─────────────────────────────────────────────

ChartType       = Literal["bar", "line", "scatter", "histogram", "pie", "heatmap", "box"]
AggType         = Literal["sum", "mean", "count", "none"]
GranularityType = Literal["day", "week", "month", "year", "none"]
LayoutSize      = Literal["small", "medium", "large"]


class ChartConfigSchema(BaseModel):
    type:             ChartType
    x:                str
    y:                Optional[str]   = None
    color:            Optional[str]   = None
    aggregation:      AggType         = "none"
    time_granularity: GranularityType = "none"
    layout_size:      LayoutSize      = "medium"
    title:            str             = ""

    @field_validator("x", "y", "color", mode="before")
    @classmethod
    def strip_nullish(cls, v):
        """Coerce null-like strings ('null', 'none', '') to None."""
        if isinstance(v, str) and v.strip().lower() in ("null", "none", ""):
            return None
        return v

    @model_validator(mode="after")
    def histogram_has_no_y(self) -> "ChartConfigSchema":
        if self.type == "histogram":
            self.y = None
        return self

    @model_validator(mode="after")
    def non_histogram_needs_y(self) -> "ChartConfigSchema":
        if self.type != "histogram" and not self.y:
            if self.type == "pie":
                # LLM sent pie with only x — treat as value_counts, normalizer synthesises y
                self.aggregation = "count"
                # y intentionally left None; normalizer will handle it
            else:
                raise ValueError(f"Chart type '{self.type}' requires a y column")
        return self


TableType = Literal["pivot", "summary"]
AggType2  = Literal["sum", "mean", "count", "min", "max"]


class TableConfigSchema(BaseModel):
    type:        TableType
    title:       str            = ""
    index:       Optional[str]  = None          # pivot row grouping column
    columns:     Optional[str]  = None          # pivot column grouping column
    values:      str            = ""            # column to aggregate (required)
    aggregation: AggType2       = "sum"

    @field_validator("values", mode="before")
    @classmethod
    def values_must_be_string(cls, v):
        """Accept single-element lists from LLM (e.g. ["Sales"] -> "Sales")."""
        if isinstance(v, list):
            if len(v) == 1:
                return str(v[0])
            raise ValueError(f"'values' must be a single column name, got list: {v}")
        return v

    @model_validator(mode="after")
    def pivot_needs_index(self) -> "TableConfigSchema":
        if self.type == "pivot" and not self.index:
            raise ValueError("pivot table requires an 'index' column")
        return self

    @model_validator(mode="after")
    def values_not_empty(self) -> "TableConfigSchema":
        if not self.values:
            raise ValueError("'values' column is required")
        return self


class LLMResponseSchema(BaseModel):
    charts: list[ChartConfigSchema]
    tables: list[TableConfigSchema] = []

    @field_validator("charts", mode="before")
    @classmethod
    def drop_invalid_charts(cls, raw_charts):
        """
        Validate each chart individually.
        Drop invalid ones with a warning instead of failing the whole response.
        Raise only if EVERY chart is invalid (nothing left to render).
        """
        if not isinstance(raw_charts, list):
            raise ValueError("'charts' must be a list")

        valid = []
        for i, raw in enumerate(raw_charts):
            try:
                valid.append(ChartConfigSchema.model_validate(raw))
            except ValidationError as exc:
                first_err = exc.errors()[0].get("msg", str(exc))
                logger.warning("Dropping chart[%d] (%s): %s", i, raw.get("type", "?"), first_err)

        if not valid:
            raise ValueError("All charts failed validation — no renderable charts returned")

        return valid

    @field_validator("tables", mode="before")
    @classmethod
    def drop_invalid_tables(cls, raw_tables):
        """Drop invalid table configs individually — never fail the whole response."""
        if not isinstance(raw_tables, list):
            return []
        valid = []
        for i, raw in enumerate(raw_tables):
            try:
                valid.append(TableConfigSchema.model_validate(raw))
            except ValidationError as exc:
                first_err = exc.errors()[0].get("msg", str(exc))
                logger.warning("Dropping table[%d] (%s): %s", i, raw.get("type", "?"), first_err)
        return valid
