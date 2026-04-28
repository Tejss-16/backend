from typing import Literal, Optional
from pydantic import BaseModel, field_validator, model_validator, ValidationError
import logging
logger = logging.getLogger(__name__)

ChartType = Literal[
    "bar", "line", "area", "scatter", "histogram", "pie", "box",
    "heatmap", "bubble", "funnel", "treemap", "waterfall",
    "stacked_bar", "grouped_bar",
]
AggType         = Literal["sum", "mean", "count", "none"]
GranularityType = Literal["day", "week", "month", "year", "none"]
LayoutSize      = Literal["small", "medium", "large"]


class ChartConfigSchema(BaseModel):
    type:             ChartType
    x:                str
    y:                Optional[str]   = None
    size:             Optional[str]   = None
    z:                Optional[str]   = None
    color:            Optional[str]   = None
    aggregation:      AggType         = "none"
    time_granularity: GranularityType = "none"
    layout_size:      LayoutSize      = "medium"
    title:            str             = ""

    @field_validator("x", "y", "color", "size", "z", mode="before")
    @classmethod
    def strip_nullish(cls, v):
        if isinstance(v, str) and v.strip().lower() in ("null", "none", ""):
            return None
        return v

    @model_validator(mode="after")
    def histogram_clears_y(self) -> "ChartConfigSchema":
        if self.type == "histogram":
            self.y = None
        return self

    @model_validator(mode="after")
    def box_auto_fills_y(self) -> "ChartConfigSchema":
        if self.type == "box" and not self.y:
            self.y = self.x
            logger.debug("Box chart: auto-filled y=%r from x", self.x)
        return self

    @model_validator(mode="after")
    def non_histogram_needs_y(self) -> "ChartConfigSchema":
        no_y_types = {"histogram", "box"}
        if self.type not in no_y_types and not self.y:
            if self.type == "pie":
                self.aggregation = "count"
            else:
                raise ValueError(f"Chart type '{self.type}' requires a y column")
        return self


TableType = Literal["pivot", "summary"]
AggType2  = Literal["sum", "mean", "count", "min", "max"]


class TableConfigSchema(BaseModel):
    type:        TableType
    title:       str            = ""
    index:       Optional[str]  = None
    columns:     Optional[str]  = None
    values:      str            = ""
    aggregation: AggType2       = "sum"

    @field_validator("values", mode="before")
    @classmethod
    def values_must_be_string(cls, v):
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


# ── Scorecard schema ──────────────────────────────────────────────────────────

ScorecardAgg = Literal["sum", "mean", "count", "min", "max"]

class ScorecardConfigSchema(BaseModel):
    column:      str
    aggregation: ScorecardAgg = "sum"
    label:       str          = ""

    @field_validator("column", mode="before")
    @classmethod
    def column_not_nullish(cls, v):
        # FIX: renamed from strip_nullish to avoid confusion — raises, not strips
        if isinstance(v, str) and v.strip().lower() in ("null", "none", ""):
            raise ValueError("scorecard column must not be null")
        return v

    @field_validator("label", mode="before")
    @classmethod
    def default_label(cls, v):
        return v or ""


# ── LLMResponseSchema ─────────────────────────────────────────────────────────

class LLMResponseSchema(BaseModel):
    charts:     list[ChartConfigSchema]
    tables:     list[TableConfigSchema]     = []
    scorecards: list[ScorecardConfigSchema] = []

    @field_validator("charts", mode="before")
    @classmethod
    def drop_invalid_charts(cls, raw_charts):
        # FIX BUG 1: was stub (pass) — restored full original body
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
        # FIX BUG 1: was stub (pass) — restored full original body
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

    @field_validator("scorecards", mode="before")
    @classmethod
    def drop_invalid_scorecards(cls, raw):
        if not isinstance(raw, list):
            return []
        valid = []
        for i, item in enumerate(raw):
            try:
                valid.append(ScorecardConfigSchema.model_validate(item))
            except ValidationError as exc:
                first_err = exc.errors()[0].get("msg", str(exc))
                logger.warning("Dropping scorecard[%d]: %s", i, first_err)
        return valid