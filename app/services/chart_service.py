# app/services/chart_service.py

import asyncio
import concurrent.futures
import logging
import math
import os
import re
import time
from datetime import datetime
from app.utils.column_utils import _meaningful_numeric_cols


import pandas as pd

logger = logging.getLogger(__name__)

EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    # CPU-bound pandas work is GIL-constrained — more threads cause context-switch
    # overhead with no parallelism gain.  Keep a small pool: enough to run
    # _process_sync and _apply_date_filter concurrently with async I/O, but
    # no larger.  2–4 workers is optimal for this workload.
    max_workers=min(4, os.cpu_count() or 2),
    thread_name_prefix="chart-worker",
)

from app.pipeline.llm_client import LLMClient
from app.pipeline.normalizer import ChartConfigNormalizer
from app.pipeline.transformer import DataTransformer
from app.pipeline.chart_builder import ChartBuilder
from app.pipeline.table_builder import TableBuilder
from app.pipeline.scorecard import ScorecardBuilder
from app.schemas.chart_schema import LLMResponseSchema
from app.utils.cache import _cache_key, _cache_key_from_fingerprint, _result_cache
from app.utils.task_manager import is_cancelled
from app.utils.data_store import dataset_metadata_cache, data_store
from app.pipeline.llm_client import _infer_aggregation
from app.utils.df_stats import DataFrameStats


# ─────────────────────────────────────────────────────────────────────────────
# QUERY PLAN  — computed once per request, passed through the entire pipeline
# ─────────────────────────────────────────────────────────────────────────────

from dataclasses import dataclass, field

@dataclass
class QueryPlan:
    """
    All query-parsing results computed exactly once from corrected_query.
    Eliminates repeated calls to _query_mode, _detect_requested_chart_types,
    _extract_quantity, _should_show_tables, _should_show_scorecards which
    each re-run regex over the same string.
    """
    mode:            str
    requested_types: list
    quantity:        int
    show_tables:     bool
    show_scorecards: bool
    table_quantity:  int  = 1
    sc_quantity:     int  = 6


def _set_loop_executor(loop: asyncio.AbstractEventLoop | None = None) -> None:
    (loop or asyncio.get_event_loop()).set_default_executor(EXECUTOR)


_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

# Maximum charts returned per query mode.
# Tune these constants to control dashboard density without touching logic.
CHART_MAX_EXPLORATORY = 8   # open-ended: "dashboard", "analyze", "overview"
CHART_MAX_ALL_OF_TYPE = 8   # wildcard:   "all possible bar charts"
CHART_MAX_MULTI       = 6    # named set:  "histogram and pie and scatter"
# "specific" mode always returns exactly 1 — no cap needed

TABLE_MAX     = 2   # max tables in any single response
SCORECARD_MAX = 6   # max scorecards in any single response (mirrors scorecard.py _SCORECARD_MAX)

def _over_limit_msg(kind: str, requested: int, limit: int) -> str:
    """Human-readable message shown when user requests more than the allowed limit."""
    return (
        f"Showing {limit} {kind} (you requested {requested}, but the limit is {limit}). "
        f"If you need different ones, ask for specific {kind} in your next query."
    )

# ─────────────────────────────────────────────────────────────────────────────
# DATE RANGE EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────

class _DateRangeExtractor:
    """
    Parses a natural-language query and extracts an optional (start, end)
    date range.  Handles year spans, single years, months, quarters, and
    relative phrases ("since 2020", "before 2022", etc.).
    """

    _MONTH_MAP = {
        "january": 1, "jan": 1, "february": 2, "feb": 2,
        "march": 3,   "mar": 3, "april": 4,    "apr": 4,
        "may": 5,     "june": 6, "jun": 6,
        "july": 7,    "jul": 7, "august": 8,   "aug": 8,
        "september": 9, "sep": 9, "sept": 9,
        "october": 10, "oct": 10, "november": 11, "nov": 11,
        "december": 12, "dec": 12,
    }
    _QUARTER_MAP = {"q1": (1, 3), "q2": (4, 6), "q3": (7, 9), "q4": (10, 12)}

    def extract(self, query: str) -> tuple[datetime | None, datetime | None]:
        q = query.lower().strip()
        try:
            return self._try_all(q)
        except Exception as exc:
            logger.warning("DateRangeExtractor failed: %s", exc)
            return None, None

    def _try_all(self, q: str) -> tuple[datetime | None, datetime | None]:
        # 1. "from YEAR to YEAR" / "YEAR-YEAR" / "YEAR through YEAR"
        m = re.search(r'\b(?:from\s+)?(\d{4})\s*(?:to|through|–|-|until)\s*(\d{4})\b', q)
        if m:
            y1, y2 = int(m.group(1)), int(m.group(2))
            return datetime(min(y1, y2), 1, 1), datetime(max(y1, y2), 12, 31)

        # 2. "from MONTH YEAR to MONTH YEAR"
        month_pat = '|'.join(self._MONTH_MAP.keys())
        m = re.search(
            rf'\b(?:from\s+)?({month_pat})\s+(\d{{4}})\s+(?:to|through|until)\s+({month_pat})\s+(\d{{4}})\b', q
        )
        if m:
            m1 = self._MONTH_MAP[m.group(1)]; y1 = int(m.group(2))
            m2 = self._MONTH_MAP[m.group(3)]; y2 = int(m.group(4))
            start = datetime(y1, m1, 1); end = self._month_end(y2, m2)
            return (start, end) if start <= end else (end, start)

        # 3. "Q1 YEAR to Q4 YEAR"
        m = re.search(r'\b(q[1-4])\s+(\d{4})\s*(?:to|through|until)\s*(q[1-4])\s+(\d{4})\b', q)
        if m:
            s_m = self._QUARTER_MAP[m.group(1)]; e_m = self._QUARTER_MAP[m.group(3)]
            y1, y2 = int(m.group(2)), int(m.group(4))
            start = datetime(y1, s_m[0], 1); end = self._month_end(y2, e_m[1])
            return (start, end) if start <= end else (end, start)

        # 4. "Q1 YEAR" (single quarter)
        m = re.search(r'\b(q[1-4])\s+(\d{4})\b', q)
        if m:
            months = self._QUARTER_MAP[m.group(1)]; y = int(m.group(2))
            return datetime(y, months[0], 1), self._month_end(y, months[1])

        # 5. "MONTH YEAR" (single month)
        m = re.search(rf'\b({month_pat})\s+(\d{{4}})\b', q)
        if m:
            mo = self._MONTH_MAP[m.group(1)]; y = int(m.group(2))
            return datetime(y, mo, 1), self._month_end(y, mo)

        # 6. "since/after/from YEAR"
        m = re.search(r'\b(?:since|after|from)\s+(\d{4})\b', q)
        if m:
            y = int(m.group(1))
            return datetime(y, 1, 1), datetime(datetime.now().year, 12, 31)

        # 7. "before/until/up to YEAR"
        m = re.search(r'\b(?:before|until|up to|through)\s+(\d{4})\b', q)
        if m:
            y = int(m.group(1))
            return datetime(2000, 1, 1), datetime(y, 12, 31)

        # 8. "in/for/year YEAR"
        m = re.search(r'\b(?:in|for|year|of)\s+(\d{4})\b', q)
        if m:
            y = int(m.group(1))
            return datetime(y, 1, 1), datetime(y, 12, 31)

        # 9. Bare 4-digit year (last resort)
        m = re.search(r'\b(20\d{2}|19\d{2})\b', q)
        if m:
            y = int(m.group(1))
            return datetime(y, 1, 1), datetime(y, 12, 31)

        return None, None

    @staticmethod
    def _month_end(year: int, month: int) -> datetime:
        import calendar
        return datetime(year, month, calendar.monthrange(year, month)[1])


_date_range_extractor = _DateRangeExtractor()

_DATE_NAME_HINTS = ("date", "time", "period", "order_date", "ship", "created", "updated")


def _find_date_column(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
    for col in df.columns:
        if any(h in col.lower() for h in _DATE_NAME_HINTS) and df[col].dtype == object:
            if pd.to_datetime(df[col], errors="coerce").notna().mean() >= 0.5:
                return col
    for col in df.select_dtypes(include="object").columns:
        if pd.to_datetime(df[col], errors="coerce").notna().mean() >= 0.7:
            return col
    return None


def _apply_date_filter(df: pd.DataFrame, query: str) -> pd.DataFrame:
    """
    Return a copy of df filtered to the date range implied by `query`.
    FIX 7: the date column is stored back as datetime64 dtype in the
    returned DataFrame so the transformer always sorts chronologically.
    """
    try:
        start, end = _date_range_extractor.extract(query)
        if start is None and end is None:
            return df

        date_col = _find_date_column(df)
        if date_col is None:
            logger.info("Date filter requested but no date column found")
            return df

        parsed = (
            df[date_col]
            if pd.api.types.is_datetime64_any_dtype(df[date_col])
            else pd.to_datetime(df[date_col], errors="coerce")
        )

        mask = pd.Series(True, index=df.index)
        if start is not None:
            mask &= parsed >= pd.Timestamp(start)
        if end is not None:
            mask &= parsed <= pd.Timestamp(end)

        filtered = df[mask].copy()

        if filtered.empty:
            logger.warning(
                "Date filter (%s → %s) on %r produced 0 rows — returning full dataset",
                start, end, date_col,
            )
            return df

        # FIX 7: persist parsed datetime dtype so transformer sorts correctly
        filtered[date_col] = parsed[mask].values

        logger.info(
            "Date filter applied: col=%r  range=[%s, %s]  rows %d → %d",
            date_col, start, end, len(df), len(filtered),
        )
        return filtered

    except Exception as exc:
        logger.warning("_apply_date_filter failed (%s) — using full dataset", exc)
        return df



# ─────────────────────────────────────────────────────────────────────────────
# DATASET CONTEXT BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_context_val(v: float) -> str:
    """Compact formatter for dataset context — keeps the profile readable."""
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


def _apply_aggregation_inference(llm_schema, col_dt_list: list) -> None:
    """
    Post-process LLM chart configs: fill in aggregation for any chart whose
    y/z column has aggregation="none" but a deterministic correct aggregation
    can be inferred from _infer_aggregation.

    ONLY fills in "none" — never overrides a concrete aggregation the LLM
    already provided (sum/mean/count/min/max).  This prevents the inference
    from wrongly changing correct LLM choices.

    Skips chart types that legitimately use "none" (scatter, histogram, box).
    Mutates llm_schema.charts in place.
    """
    dtype_map = {col: str(dt) for col, dt in col_dt_list}
    _NO_AGG_TYPES = {"scatter", "histogram", "box"}

    for chart in llm_schema.charts:
        if chart.type in _NO_AGG_TYPES:
            continue
        # Only fill in when the LLM left aggregation as "none"
        if chart.aggregation != "none":
            continue
        target_col = chart.y
        if chart.type == "heatmap":
            target_col = getattr(chart, "z", None)
        if not target_col or target_col not in dtype_map:
            continue
        inferred = _infer_aggregation(target_col, dtype_map[target_col])
        if inferred != "none":
            logger.debug(
                "Aggregation fill-in: chart=%s col=%r none→%r",
                chart.type, target_col, inferred,
            )
            chart.aggregation = inferred


# ─────────────────────────────────────────────────────────────────────────────
# ChartGenerator
# ─────────────────────────────────────────────────────────────────────────────

class ChartGenerator:
    def __init__(self, data: pd.DataFrame, dataset_id: str | None = None):
        self.data               = data
        self._dataset_id        = dataset_id
        self._llm               = LLMClient()
        # Detect any multiplicative column pair (e.g. price × quantity) and
        # pre-compute a derived column on self.data before building stats/normalizer.
        # This must run BEFORE DataFrameStats/ChartConfigNormalizer so they include
        # the derived column in their num_cols list from the start.
        self._derived_col: str | None = self._inject_derived_col()
        self._col_dt_list       = list(zip(data.columns, data.dtypes))
        # Compute DataFrame statistics once — shared by normalizer, scorecard
        # builder, and chart builder so no module scans the DataFrame independently.
        self._stats             = DataFrameStats(data)
        self._normalizer        = ChartConfigNormalizer(data, stats=self._stats)
        self._transformer       = DataTransformer(data)
        self._builder           = ChartBuilder(self._transformer)
        self._table_builder     = TableBuilder(data)
        self._scorecard_builder = ScorecardBuilder(data, stats=self._stats)
        logger.info("ChartGenerator initialised (%d rows, %d cols)", *data.shape)

    # ── Derived column detection ──────────────────────────────────────────────

    # Substrings that signal a column is a per-row rate (unit price, hourly rate,
    # cost-per-unit, fee-per-item, …).  Matched against the fully-lowercased,
    # space/underscore-stripped column name.
    _RATE_HINTS = frozenset({
        "price", "rate", "unitcost", "unit_cost",
        "unitprice", "unit_price", "saleprice", "sale_price",
        "sellingprice", "selling_price", "costperunit", "cost_per_unit",
        "feeper", "chargeper", "wageper", "salaryper",
        "perperson", "perunit", "peritem", "perorder",
        "wage", "salary", "fee", "tariff",
    })

    # Substrings that signal a column is a count / quantity.
    _QTY_HINTS = frozenset({
        "quantity", "qty", "units", "count", "volume",
        "sold", "ordered", "shipped", "produced", "purchased",
        "hours", "hrs", "days", "weeks", "months",
        "numberof", "numof", "total_items", "items",
    })

    def _inject_derived_col(self) -> str | None:
        """
        Detect a (rate_col × qty_col) pair in self.data using column-name
        heuristics that work for ANY dataset — no hardcoded column names.

        Rules (all must pass):
          1. Exactly one column matches _RATE_HINTS and is numeric.
          2. Exactly one column matches _QTY_HINTS and is numeric.
          3. The two columns are different.
          4. The derived product has a sum meaningfully larger than the raw
             rate column's sum (guards against false positives where a rate
             column coincidentally has a qty-like sibling).

        If a pair is found, adds a "<rate>_x_<qty>" column to self.data and
        returns its name.  Otherwise returns None.
        """
        df = self.data
        num_cols = df.select_dtypes(include="number").columns.tolist()

        rate_candidates = [
            c for c in num_cols
            if any(h in c.lower().replace(" ", "").replace("_", "") for h in self._RATE_HINTS)
        ]
        qty_candidates = [
            c for c in num_cols
            if any(h in c.lower().replace(" ", "").replace("_", "") for h in self._QTY_HINTS)
        ]

        # Only act when there is exactly one unambiguous candidate on each side
        if len(rate_candidates) != 1 or len(qty_candidates) != 1:
            return None
        rate_col = rate_candidates[0]
        qty_col  = qty_candidates[0]
        if rate_col == qty_col:
            return None

        try:
            derived_sum = float((df[rate_col] * df[qty_col]).sum())
            rate_sum    = float(df[rate_col].sum())
            # Sanity check: derived total must be larger than the raw rate sum
            # (if they're equal or derived is smaller, the pair is likely wrong)
            if derived_sum <= rate_sum:
                return None

            derived_col = f"{rate_col}_x_{qty_col}"
            df[derived_col] = df[rate_col] * df[qty_col]
            logger.info(
                "ChartGenerator: derived column '%s' = %s × %s  (sum=%.2f vs rate_sum=%.2f)",
                derived_col, rate_col, qty_col, derived_sum, rate_sum,
            )
            return derived_col
        except Exception as exc:
            logger.warning("ChartGenerator: derived column detection failed: %s", exc)
            return None

    def _make_query_plan(self, query: str) -> QueryPlan:
        """
        Parse corrected_query once and return a QueryPlan.
        All downstream methods consume the plan — no method re-parses the query.
        """
        mode            = self._query_mode(query)
        requested_types = self._detect_requested_chart_types(query)
        quantity        = self._extract_quantity(query)
        show_tables     = self._should_show_tables(query)
        show_scorecards = self._should_show_scorecards(query)
        tbl_qty = self._extract_non_chart_quantity(query, "table")
        if tbl_qty == 1:
            tbl_qty = self._extract_non_chart_quantity(query, "tables")
        sc_qty  = self._extract_non_chart_quantity(query, "scorecard")
        if sc_qty == 1:
            sc_qty  = self._extract_non_chart_quantity(query, "scorecards")
        return QueryPlan(
            mode=mode,
            requested_types=requested_types,
            quantity=quantity,
            show_tables=show_tables,
            show_scorecards=show_scorecards,
            table_quantity=min(tbl_qty, TABLE_MAX),
            sc_quantity=min(sc_qty if sc_qty > 1 else SCORECARD_MAX, SCORECARD_MAX),
        )

    def _get_dataset_metadata(self) -> dict:
        """
        Return the metadata dict for self.data, computing it at most once.

        Two-level cache:
          1. Instance-level (_metadata_cache): computed at most once per
             ChartGenerator lifetime regardless of dataset_id.  Eliminates
             the double-compute seen when _augment_query() and generate()
             both call this method in the same request.
          2. Process-level (dataset_metadata_cache): keyed by dataset_id,
             survives across requests for the same dataset.  Only active
             when dataset_id was provided at construction time.
        """
        if hasattr(self, "_metadata_cache"):
            return self._metadata_cache  # type: ignore[return-value]

        if self._dataset_id is not None:
            meta = dataset_metadata_cache.get_or_compute(
                self._dataset_id, self.data
            )
        else:
            # Fallback: compute without the process-level cache (no dataset_id)
            from app.utils.data_store import _compute_dataset_metadata
            meta = _compute_dataset_metadata(self.data)

        self._metadata_cache = meta  # cache on instance — survives this request
        return meta

    _BROAD_INTENT = {
        "dashboard", "analyze", "analyse", "analysis", "analyses",
        "overview", "report", "explore", "exploration",
        "insights", "insight", "understand", "examine", "investigate",
        "full", "complete", "everything", "all", "whole", "entire",
        "show me", "tell me", "give me", "what is", "what are",
        "create dashboard", "make dashboard", "build dashboard",
        "deep dive", "deep-dive", "summarize", "summarise",
    }
    _SUMMARY_INTENT = {
        "total", "average", "mean", "count", "sum", "max", "min",
        "kpi", "metric", "performance", "summary", "summaries",
        "aggregate", "aggregation", "statistics", "stats",
    }
    _PIVOT_INTENT = {
        "breakdown", "break down", "by category", "by region", "by segment",
        "by product", "by month", "by year", "by date", "by customer",
        "compare", "comparison", "across", "per group", "per category",
        "distribution table", "tabulate", "cross tab", "crosstab",
    }
    _SCORECARD_EXPLICIT = {
        "scorecard", "scorecards", "kpi", "kpis", "key metric", "key metrics",
        "metric card", "metric cards", "stat card", "stat cards",
        "summary card", "summary cards",
    }
    _TABLE_EXPLICIT = {
        "table", "tables", "tabular", "grid", "data table",
        "show table", "give table", "add table", "include table",
    }
    _CHART_TYPE_KEYWORDS = {
        "line": "line", "line chart": "line",
        "area": "area", "area chart": "area",
        "bar": "bar", "bar chart": "bar", "column chart": "bar",
        "stacked bar": "stacked_bar", "stacked_bar": "stacked_bar",
        "grouped bar": "grouped_bar", "grouped_bar": "grouped_bar",
        "pie": "pie", "pie chart": "pie",
        "donut": "pie", "donut chart": "pie",
        "scatter": "scatter", "scatter plot": "scatter", "scatter chart": "scatter",
        "bubble": "bubble", "bubble chart": "bubble",
        "histogram": "histogram",
        "box": "box", "box plot": "box", "box chart": "box", "whisker": "box",
        "heatmap": "heatmap", "heat map": "heatmap",
        "funnel": "funnel", "funnel chart": "funnel",
        "treemap": "treemap", "tree map": "treemap",
        "waterfall": "waterfall", "waterfall chart": "waterfall",
    }

    @staticmethod
    def _query_matches(query: str, keywords: set) -> bool:
        q = query.lower()
        return any(kw in q for kw in keywords)

    @staticmethod
    def _extract_quantity(query: str) -> int:
        """
        Returns:
          -1  → "all / every / possible / any" wildcard (generate as many as data allows)
          n>0 → exact count requested
          1   → default (no quantity word found)
        """
        q = query.lower()
        chart_type_pattern = '|'.join(
            re.escape(k) for k in sorted(ChartGenerator._CHART_TYPE_KEYWORDS, key=len, reverse=True)
        )
        # "all bar charts", "every possible histogram", "all possible bar charts"
        wildcard_match = re.search(
            r'\b(?:all|every|any)(?:\s+possible)?\s+(?:' + chart_type_pattern + r')',
            q,
        )
        if wildcard_match:
            return -1  # sentinel: generate as many as the data supports

        digit_match = re.search(r'\b(\d+)\s+(?:' + chart_type_pattern + r')', q)
        if digit_match:
            return max(1, int(digit_match.group(1)))
        word_pattern = '|'.join(re.escape(w) for w in _NUMBER_WORDS)
        word_match = re.search(
            r'\b(' + word_pattern + r')\s+(?:' + chart_type_pattern + r')',
            q,
        )
        if word_match:
            return _NUMBER_WORDS.get(word_match.group(1), 1)
        return 1

    @staticmethod
    def _extract_non_chart_quantity(query: str, keyword: str) -> int:
        """
        Extract a numeric quantity for non-chart items like tables or scorecards.
        e.g. "give 3 tables" → 3,  "show five scorecards" → 5
        Returns 1 if no number found.
        """
        q = query.lower()
        # digit: "3 tables"
        m = re.search(r'\b(\d+)\s+' + re.escape(keyword), q)
        if m:
            return max(1, int(m.group(1)))
        # word: "three tables"
        word_pattern = '|'.join(re.escape(w) for w in _NUMBER_WORDS)
        m = re.search(r'\b(' + word_pattern + r')\s+' + re.escape(keyword), q)
        if m:
            return _NUMBER_WORDS.get(m.group(1), 1)
        return 1

    def _detect_requested_chart_types(self, query: str) -> list[str]:
        q = query.lower()
        seen: dict[str, int] = {}
        for phrase, chart_type in self._CHART_TYPE_KEYWORDS.items():
            pos = q.find(phrase)
            if pos != -1 and chart_type not in seen:
                seen[chart_type] = pos
        return [t for t, _ in sorted(seen.items(), key=lambda x: x[1])]

    @staticmethod
    def _count_chart_dimensions(query: str, col_dt_list: list) -> int:
        """
        Count how many distinct dataset columns are referenced in the query
        joined by 'and' / 'or' / commas alongside a chart type request.
        Returns the count so callers can decide whether to treat the query
        as multi-dimension (more than one chart needed) vs single-chart.

        Only considers actual column names (case-insensitive) so generic
        words like 'and' in natural language don't falsely inflate the count.
        """
        if not col_dt_list:
            return 0
        q_lower = query.lower()
        matched = sum(
            1 for col, _ in col_dt_list
            if col.lower() in q_lower
        )
        return matched

    def _query_mode(self, query: str) -> str:
        types    = self._detect_requested_chart_types(query)
        quantity = self._extract_quantity(query)
        # Check for table-only / scorecard-only BEFORE falling to exploratory,
        # so "give 2 tables" never becomes an exploratory chart dump.
        if len(types) == 0:
            if self._query_matches(query, self._TABLE_EXPLICIT):      return "tables_only"
            if self._query_matches(query, self._SCORECARD_EXPLICIT):  return "scorecards_only"
            return "exploratory"
        if quantity == -1:                    return "all_of_type"
        if len(types) == 1 and quantity == 1:
            # Treat as "multi" when the user names multiple columns joined by
            # "and"/"or"/comma alongside a single chart type — e.g.
            # "pie chart for countries and continent".
            # We detect this by counting column-name matches AND by checking
            # whether the query contains " and " / " or " / "," between
            # a column-like word pattern (generic heuristic, no hardcoding).
            dim_count = self._count_chart_dimensions(query, self._col_dt_list)
            if dim_count >= 2:
                return "multi"
            # Generic heuristic: "pie chart for X and Y" where X/Y may not
            # exactly match column names (e.g. partial words / aliases).
            # Look for connector words between two non-chart noun phrases.
            _connector_re = re.compile(
                r'\b(?:for|of|by|on)\b.+\b(?:and|or)\b.+',
                re.IGNORECASE,
            )
            if _connector_re.search(query):
                return "multi"
            return "specific"
        return "multi"

    def _should_show_scorecards(self, query: str) -> bool:
        mode = self._query_mode(query)
        if mode == "scorecards_only":                          return True
        if mode == "tables_only":                              return False
        if mode in ("specific", "multi", "all_of_type"):
            return self._query_matches(query, self._SCORECARD_EXPLICIT)
        if self._query_matches(query, self._SCORECARD_EXPLICIT): return True
        return self._query_matches(query, self._BROAD_INTENT | self._SUMMARY_INTENT)

    def _should_show_tables(self, query: str) -> bool:
        mode = self._query_mode(query)
        if mode == "tables_only":                              return True
        if mode == "scorecards_only":                          return False
        if mode in ("specific", "multi", "all_of_type"):
            return self._query_matches(query, self._TABLE_EXPLICIT)
        if self._query_matches(query, self._TABLE_EXPLICIT):   return True
        return self._query_matches(query, self._BROAD_INTENT | self._PIVOT_INTENT | self._SUMMARY_INTENT)

    def _needs_table(self, query: str) -> bool:
        return self._query_matches(query, self._BROAD_INTENT | self._PIVOT_INTENT | self._TABLE_EXPLICIT)

    def _augment_query_with_plan(self, query: str, plan: QueryPlan) -> str:
        """
        Build the augmented query using a pre-computed QueryPlan.
        Identical logic to _augment_query but uses plan fields directly —
        no repeated calls to _query_mode, _extract_quantity, _detect_requested_chart_types.
        """
        mode     = plan.mode
        quantity = plan.quantity
        types    = plan.requested_types
        _meta    = self._get_dataset_metadata()

        if mode == "tables_only":
            n = quantity if quantity > 1 else 2
            return query + (
                f"\n\n[TABLE INSTRUCTION]\nThe user wants ONLY tables — no charts, no scorecards. "
                f"Return exactly {n} pivot table(s), each grouping by a different meaningful "
                f"categorical column and aggregating a real numeric metric. "
                f"Return an empty 'charts' array and empty 'scorecards' array."
            )

        if mode == "scorecards_only":
            n = quantity if quantity > 1 else 6
            return query + (
                f"\n\n[SCORECARD INSTRUCTION]\nReturn exactly {n} KPI scorecards. "
                f"No charts, no tables. Return empty 'charts' and 'tables' arrays."
            )

        if mode == "specific":
            t = types[0] if types else "bar"
            return query + f"\n\n[CHART INSTRUCTION]\nReturn exactly one {t} chart."

        if mode == "multi":
            if quantity > 1 and len(types) == 1:
                return query + f"\n\n[CHART INSTRUCTION]\nReturn exactly {quantity} {types[0]} charts using different columns."
            if len(types) == 1:
                # Single chart type but multiple dimensions/columns requested
                # (e.g. "pie chart for countries and continent").
                # Instruct the LLM to produce one chart per dimension mentioned.
                return query + (
                    f"\n\n[CHART INSTRUCTION]\nThe user wants a {types[0]} chart for EACH "
                    f"dimension/column mentioned in the query. Return one separate "
                    f"{types[0]} chart per dimension — do NOT merge them into a single chart."
                )
            return query + f"\n\n[CHART INSTRUCTION]\nReturn one chart of each type: {', '.join(types)}."

        if mode == "all_of_type":
            type_str = " and ".join(types) if types else "chart"
            return query + f"\n\n[CHART INSTRUCTION]\nReturn as many {type_str} charts as meaningful. No other chart types."

        # exploratory — inject dataset structure + analytical priority
        num_cols = _meta.get("num_cols", [])
        cat_cols = _meta.get("cat_cols", [])
        dt_cols  = _meta.get("dt_cols",  [])
        primary_metric = _meta.get("primary_metric")
        profit_metric  = _meta.get("profit_metric")
        primary_cat    = _meta.get("primary_cat")

        col_ctx = "\n\n[DATASET STRUCTURE — use this to generate a full dashboard]\n"
        if dt_cols:  col_ctx += f"Date/time columns: {dt_cols}\n"
        if num_cols: col_ctx += f"Numeric columns: {num_cols}\n"
        if cat_cols: col_ctx += f"Categorical columns: {cat_cols}\n"

        col_ctx += "\n[ANALYTICAL PRIORITY FOR THIS DATASET]\n"
        if dt_cols:
            col_ctx += (
                f"1. FIRST chart MUST be a large line/area time-series using date column "
                f"'{dt_cols[0]}'"
            )
            if primary_metric:
                col_ctx += f" with y='{primary_metric}'"
            if profit_metric and profit_metric != primary_metric:
                col_ctx += f" and color/series also showing '{profit_metric}'"
            col_ctx += ". layout_size: 'large'.\n"
        if primary_metric and primary_cat:
            col_ctx += (
                f"2. SECOND chart: bar or grouped_bar of '{primary_metric}' "
                f"by '{primary_cat}'. layout_size: 'medium'.\n"
            )
        col_ctx += (
            "3. Then: category splits (pie/treemap for low-cardinality columns), "
            "year-over-year comparisons (grouped_bar), "
            "followed by distributions (histogram/box) last.\n"
            "4. DO NOT place scatter/bubble/histogram/box as the first two charts.\n"
        )
        col_ctx += (
            "\nGenerate as many charts as makes sense for a complete dashboard. "
            "Cover time trends, category comparisons, and distributions."
        )
        return query + col_ctx
        """
        Identical to the original, except the 'Numeric columns' context injected
        for exploratory queries uses _meaningful_numeric_cols so the LLM never
        sees TransactionID, CustomerID, Zip, etc. as plottable metrics.
        """
        mode     = self._query_mode(query)
        quantity = self._extract_quantity(query)
        types    = self._detect_requested_chart_types(query)
 
        if mode == "tables_only":
            n = quantity if quantity > 1 else 2
            return query + (
                f"\n\n[TABLE INSTRUCTION]\nThe user wants ONLY tables — no charts, no scorecards. "
                f"Return exactly {n} pivot table(s), each grouping by a different meaningful "
                f"categorical column and aggregating a real numeric metric. "
                f"Return an empty 'charts' array and empty 'scorecards' array."
            )

        if mode == "scorecards_only":
            return query + (
                "\n\n[SCORECARD INSTRUCTION]\nThe user wants ONLY KPI scorecards — no charts, no tables. "
                "Return 4-6 meaningful scorecards covering different business dimensions. "
                "Return an empty 'charts' array and empty 'tables' array."
            )

        if mode == "exploratory":
            # Use cached metadata — all column classifications and cardinalities
            # were computed once at dataset upload, not on every query.
            _meta       = self._get_dataset_metadata()
            num_cols    = _meta["num_cols"]
            cat_cols    = _meta["cat_cols"]
            dt_cols     = _meta["dt_cols"]
            primary_metric = _meta["primary_metric"]
            profit_metric  = _meta["profit_metric"]
            primary_cat    = _meta["primary_cat"]

            col_ctx = "\n\n[DATASET STRUCTURE — use this to generate a full dashboard]\n"
            if dt_cols:  col_ctx += f"Date/time columns: {dt_cols}\n"
            if num_cols: col_ctx += f"Numeric columns: {num_cols}\n"
            if cat_cols: col_ctx += f"Categorical columns: {cat_cols}\n"

            col_ctx += "\n[ANALYTICAL PRIORITY FOR THIS DATASET]\n"
            if dt_cols:
                col_ctx += (
                    f"1. FIRST chart MUST be a large line/area time-series using date column "
                    f"'{dt_cols[0]}'"
                )
                if primary_metric:
                    col_ctx += f" with y='{primary_metric}'"
                if profit_metric and profit_metric != primary_metric:
                    col_ctx += f" and color/series also showing '{profit_metric}'"
                col_ctx += ". layout_size: 'large'.\n"
            if primary_metric and primary_cat:
                col_ctx += (
                    f"2. SECOND chart: bar or grouped_bar of '{primary_metric}' "
                    f"by '{primary_cat}'. layout_size: 'medium'.\n"
                )
            col_ctx += (
                "3. Then: category splits (pie/treemap for low-cardinality columns), "
                "year-over-year comparisons (grouped_bar), "
                "followed by distributions (histogram/box) last.\n"
                "4. DO NOT place scatter/bubble/histogram/box as the first two charts.\n"
            )
            col_ctx += (
                "\nGenerate as many charts as makes sense for a complete dashboard. "
                "Cover time trends, category comparisons, and distributions."
            )
            return query + col_ctx
 
        if quantity > 1 and len(types) == 1:
            hint = (
                f"\n\n[QUANTITY INSTRUCTION]\nThe user explicitly requested "
                f"{quantity} {types[0]} charts. "
                f"You MUST return exactly {quantity} charts of type '{types[0]}', "
                f"each using a DIFFERENT column or dimension. Do NOT return fewer."
            )
            return query + hint

        if quantity == -1 and len(types) >= 1:
            # "all possible bar charts" — generate as many distinct charts of
            # the requested type(s) as the data meaningfully supports.
            type_list = types if len(types) > 1 else [types[0]]
            type_str  = " and ".join(f"'{t}'" for t in type_list)
            hint = (
                f"\n\n[ALL INSTRUCTION]\nThe user wants ALL possible {type_str} chart(s) "
                f"that can be meaningfully built from this dataset. "
                f"Generate one chart for EACH different combination of columns that "
                f"produces a useful, non-redundant {type_str} chart. "
                f"Only include charts that are genuinely informative — skip meaningless ones. "
                f"Return ONLY charts of type(s): {type_list}. "
                f"Do NOT add scorecards or tables. Do NOT return other chart types."
            )
            return query + hint
 
        if self._needs_table(query):
            return query + (
                "\n\n[TABLE INSTRUCTION]\nInclude up to 2 pivot tables. "
                "Each table must have a non-null 'index' and 'values' field "
                "and produce multiple rows."
            )
 
        if self._query_matches(query, self._SCORECARD_EXPLICIT):
            return query + "\n\n[NOTE] The user also wants KPI scorecards shown above the charts."
 
        return query

    def _filter_table_configs(self, table_cfgs: list) -> list:
        filtered = []
        for cfg in table_cfgs:
            if isinstance(cfg, dict):
                filtered.append(cfg)
            elif cfg.type == "summary":
                logger.debug("Dropping scalar summary table %r — covered by scorecards", cfg.title)
            else:
                filtered.append(cfg)
        return filtered

    @staticmethod
    def _check_cancelled(task_id: str, stage: str) -> None:
        if is_cancelled(task_id):
            logger.warning("Cancel flag detected at stage '%s' for task %s", stage, task_id)
            raise asyncio.CancelledError(f"Cancelled at stage: {stage}")

    async def generate(self, query: str, task_id: str = "") -> dict:
        # Build cache key from the pre-computed fingerprint stored at upload time.
        # Falls back to full DataFrame hashing only when dataset_id is unavailable.
        if self._dataset_id is not None:
            fingerprint = data_store.get_fingerprint(self._dataset_id)
            key = (
                _cache_key_from_fingerprint(fingerprint, query)
                if fingerprint is not None
                else _cache_key(self.data, query)
            )
        else:
            key = _cache_key(self.data, query)

        if (cached := _result_cache.get(key)) is not None:
            logger.info("Cache hit for query: %r", query[:60])
            return cached

        # ── Strategy: run is_data_query concurrently with the full pipeline ──
        # correct_query is fast (~200ms) and must complete before get_chart_config
        # so we keep it serial with the main call.  is_data_query is independent
        # — fire it as a background task and check the result just before we
        # commit to processing the LLM output.  This hides its latency entirely
        # behind the typo-correction + data-prep + LLM-call sequence.
        relevance_task = asyncio.create_task(
            self._llm.is_data_query(query, self._col_dt_list)
        )

        corrected_query = await self._llm.correct_query(query, self._col_dt_list)

        # Check relevance — if already done (fast model), this is free;
        # otherwise we await however long remains.
        is_relevant, rejection_reason = await relevance_task
        if not is_relevant:
            logger.info("Query rejected as off-topic: %r", query[:80])
            return {"error": rejection_reason, "scorecards": [], "charts": [], "tables": []}
        # ─────────────────────────────────────────────────────────────────────

        self._check_cancelled(task_id, "pre-start")

        loop = asyncio.get_running_loop()

        # Apply date filter
        filtered_data = await loop.run_in_executor(
            EXECUTOR, _apply_date_filter, self.data, corrected_query,
        )

        # Build stats for the working data once — shared by all working components.
        # If no date filter was applied, reuse the stats already on the instance.
        if filtered_data is not self.data:
            working_stats         = DataFrameStats(filtered_data)
            working_normalizer    = ChartConfigNormalizer(filtered_data, stats=working_stats)
            working_transformer   = DataTransformer(filtered_data)
            working_builder       = ChartBuilder(working_transformer)
            working_table_builder = TableBuilder(filtered_data)
            working_scorecard     = ScorecardBuilder(filtered_data, stats=working_stats)
        else:
            working_stats         = self._stats
            working_normalizer    = self._normalizer
            working_builder       = self._builder
            working_table_builder = self._table_builder
            working_scorecard     = self._scorecard_builder

        # Filter columns for LLM — use pre-computed num_cols from stats,
        # no additional select_dtypes / _meaningful_numeric_cols call.
        # Also exclude non-KPI numeric columns (area, size, lat/lon, etc.)
        # so the LLM never receives them as chart or scorecard candidates.
        from app.utils.data_store import _is_kpi_column as _kpi
        num_col_set  = set(working_stats.num_cols)
        all_num_cols = set(filtered_data.select_dtypes(include="number").columns)
        working_col_dt_list = [
            (c, dt) for c, dt in zip(filtered_data.columns, filtered_data.dtypes)
            if c not in all_num_cols          # keep non-numeric cols (categoricals, dates)
            or (c in num_col_set and _kpi(c)) # keep numeric cols only if they are KPIs
        ]

        # If a derived column was created at init time (e.g. Price_x_Quantity),
        # inject it at the front of working_col_dt_list so the LLM always sees it
        # as the preferred revenue/total metric.  Also remove the raw rate column
        # (the one whose name contains a rate-hint) so the LLM cannot pick it and
        # produce a meaningless sum.  This is generic — no hardcoded column names.
        if self._derived_col and self._derived_col in filtered_data.columns:
            # Find and drop the raw rate factor from the list
            derived_lower = self._derived_col.lower()
            rate_factor = next(
                (c for c, _ in working_col_dt_list
                 if any(h in c.lower().replace(" ", "").replace("_", "")
                        for h in self._RATE_HINTS)
                 and c != self._derived_col),
                None,
            )
            if rate_factor:
                working_col_dt_list = [(c, dt) for c, dt in working_col_dt_list if c != rate_factor]
            # Prepend derived column if not already present
            existing = {c for c, _ in working_col_dt_list}
            if self._derived_col not in existing:
                working_col_dt_list.insert(0, (self._derived_col, filtered_data[self._derived_col].dtype))

        # Compute query plan ONCE — all mode/show/quantity decisions come from here.
        # Eliminates 5+ repeated calls to _query_mode and friends per request.
        plan = self._make_query_plan(corrected_query)

        effective_query = self._augment_query_with_plan(corrected_query, plan)

        # head(3).to_string() is trivial — no benefit from thread offload
        sample = filtered_data.head(3).to_string()

        # Retrieve the dataset profile from cache — computed once at upload.
        dataset_context = self._get_dataset_metadata()["profile_str"]

        self._check_cancelled(task_id, "pre-llm")

        # ── tables_only: skip get_chart_config entirely ───────────────────────
        if plan.mode == "tables_only":
            warning    = _over_limit_msg("tables", plan.table_quantity, TABLE_MAX) if plan.table_quantity > TABLE_MAX else None
            logger.info("tables_only mode — calling get_table_config directly (max=%d)", plan.table_quantity)
            raw_table_cfgs = await self._llm.get_table_config(
                working_col_dt_list, sample, dataset_context, corrected_query,
                max_tables=plan.table_quantity,
            )
            from app.schemas.chart_schema import TableConfigSchema
            from pydantic import ValidationError
            validated = []
            for raw in raw_table_cfgs:
                try:
                    validated.append(TableConfigSchema.model_validate(raw))
                except (ValidationError, Exception) as exc:
                    logger.warning("Table config validation failed: %s", exc)

            tables = working_table_builder.build_all(validated[:plan.table_quantity])
            result = {"scorecards": [], "charts": [], "tables": tables}
            if warning:
                result["warning"] = warning
            logger.info("tables_only completed: %d table(s) built", len(tables))
            self._check_cancelled(task_id, "post-processing")
            _result_cache.set(key, result)
            return result

        # ── scorecards_only: skip get_chart_config entirely ───────────────────
        if plan.mode == "scorecards_only":
            warning   = _over_limit_msg("scorecards", plan.sc_quantity, SCORECARD_MAX) if plan.sc_quantity > SCORECARD_MAX else None
            logger.info("scorecards_only mode — calling get_scorecard_config directly (max=%d)", plan.sc_quantity)
            raw_scorecard_cfgs = await self._llm.get_scorecard_config(
                working_col_dt_list, sample, dataset_context, corrected_query,
                max_scorecards=plan.sc_quantity,
            )
            from app.schemas.chart_schema import ScorecardConfigSchema
            from pydantic import ValidationError
            validated_sc = []
            for raw in raw_scorecard_cfgs:
                try:
                    validated_sc.append(ScorecardConfigSchema.model_validate(raw))
                except (ValidationError, Exception) as exc:
                    logger.warning("Scorecard config validation failed: %s", exc)
            scorecards = working_scorecard.build_from_llm(validated_sc[:plan.sc_quantity])
            result = {"scorecards": scorecards, "charts": [], "tables": []}
            if warning:
                result["warning"] = warning
            logger.info("scorecards_only completed: %d scorecard(s) built", len(scorecards))
            self._check_cancelled(task_id, "post-processing")
            _result_cache.set(key, result)
            return result
        # ── end short-circuits ────────────────────────────────────────────────

        try:
            # For exploratory/dashboard queries, fire get_table_config concurrently
            # with get_chart_config — independent calls, table result not needed until both finish.
            if plan.mode == "exploratory" and plan.show_tables:
                chart_coro = self._llm.get_chart_config(
                    working_col_dt_list, sample, "", effective_query,
                    dataset_context=dataset_context, query_mode=plan.mode,
                )
                table_coro = self._llm.get_table_config(
                    working_col_dt_list, sample, dataset_context, corrected_query,
                    max_tables=2,
                )
                llm_schema, raw_table_cfgs_concurrent = await asyncio.wait_for(
                    asyncio.gather(chart_coro, table_coro, return_exceptions=False),
                    timeout=180,
                )
            else:
                llm_schema = await asyncio.wait_for(
                    self._llm.get_chart_config(
                        working_col_dt_list, sample, "", effective_query,
                        dataset_context=dataset_context, query_mode=plan.mode,
                    ),
                    timeout=180,
                )
                raw_table_cfgs_concurrent = None
        except asyncio.TimeoutError:
            logger.warning("LLM call timed out for query: %r", query[:60])
            llm_schema = self._llm._fallback_config(working_col_dt_list)
            raw_table_cfgs_concurrent = None

        # Apply Python-side aggregation inference
        _apply_aggregation_inference(llm_schema, working_col_dt_list)

        self._check_cancelled(task_id, "post-llm")

        if plan.show_tables and not llm_schema.tables:
            logger.info("No tables from main LLM — requesting separately")
            raw_table_cfgs = raw_table_cfgs_concurrent or await self._llm.get_table_config(
                working_col_dt_list, sample, dataset_context, corrected_query,
                max_tables=2,
            )
            if raw_table_cfgs:
                from app.schemas.chart_schema import TableConfigSchema
                from pydantic import ValidationError
                validated = []
                for raw in raw_table_cfgs:
                    try:
                        validated.append(TableConfigSchema.model_validate(raw))
                    except (ValidationError, Exception) as exc:
                        logger.warning("Table config validation failed: %s", exc)
                if validated:
                    llm_schema.tables = validated
                    logger.info("Injected %d table(s) from table-config call", len(validated))

        result = await loop.run_in_executor(
            EXECUTOR, self._process_sync,
            llm_schema, corrected_query, filtered_data,
            working_normalizer, working_builder, working_table_builder, working_scorecard,
            plan,
        )

        self._check_cancelled(task_id, "post-processing")
        _result_cache.set(key, result)
        return result

    def _process_sync(
        self,
        llm_schema,
        query: str,
        working_data=None,
        working_normalizer=None,
        working_builder=None,
        working_table_builder=None,
        working_scorecard=None,
        plan: QueryPlan = None,
    ) -> dict:
        import time
        from app.utils.task_manager import is_cancelled

        t0 = time.perf_counter()
        if working_data is None:          working_data          = self.data
        if working_normalizer is None:    working_normalizer    = self._normalizer
        if working_builder is None:       working_builder       = self._builder
        if working_table_builder is None: working_table_builder = self._table_builder
        if working_scorecard is None:     working_scorecard     = self._scorecard_builder

        # Use QueryPlan when available — all mode/show/quantity decisions already
        # computed once in generate().  Fall back to recomputing only when called
        # directly (e.g. tests) without a plan.
        if plan is None:
            plan = self._make_query_plan(query)

        mode            = plan.mode
        requested_types = plan.requested_types
        quantity        = plan.quantity

        _num_cols_cache = (
            list(working_normalizer._stats.num_cols)
            if working_normalizer._stats is not None
            else _meaningful_numeric_cols(working_data)
        )

        scorecards = (
            working_scorecard.build_from_llm(llm_schema.scorecards)
            if plan.show_scorecards else []
        )
 
        if mode == "tables_only":
            # User asked only for tables — skip charts and scorecards entirely.
            n          = quantity if quantity > 1 else 2
            table_cfgs = self._filter_table_configs(llm_schema.tables)[:n]
            tables     = working_table_builder.build_all(table_cfgs)
            result = {"scorecards": [], "charts": [], "tables": tables}
            logger.info(
                "_process_sync END  mode=tables_only  tables=%d  elapsed=%.2fs",
                len(tables), time.perf_counter() - t0,
            )
            return result

        if mode == "scorecards_only":
            # User asked only for scorecards — skip charts and tables entirely.
            result = {"scorecards": scorecards, "charts": [], "tables": []}
            logger.info(
                "_process_sync END  mode=scorecards_only  scorecards=%d  elapsed=%.2fs",
                len(scorecards), time.perf_counter() - t0,
            )
            return result

        if mode == "specific":
            target   = requested_types[0]
            matching = [c for c in llm_schema.charts if c.type == target]
            if not matching:
                fallback_cfg = self._fallback_chart_config(target, working_data, num_cols=_num_cols_cache)
                if fallback_cfg:
                    from app.schemas.chart_schema import ChartConfigSchema
                    try:
                        matching = [ChartConfigSchema.model_validate(fallback_cfg)]
                    except Exception:
                        pass
            if not matching:
                return {
                    "scorecards": [],
                    "charts": [{"type": "not_possible", "requested_type": target}],
                    "tables": [],
                }
            llm_schema.charts = matching[:1]
            if not plan.show_scorecards: scorecards = []
            if not plan.show_tables:     llm_schema.tables = []
 
        elif mode == "multi":
            if len(requested_types) == 1:
                # Only one chart type was requested, but the query implies
                # multiple charts of that type — either via an explicit
                # quantity ("3 pie charts") or via multiple named dimensions
                # ("pie chart for countries and continent"). Either way there
                # is only one type to consider, so keep EVERY chart of that
                # type the LLM returned. (Deduping by type, as the multi-type
                # branch below does, would wrongly collapse these down to a
                # single chart since they all share the same `type`.)
                target = requested_types[0]
                kept   = [c for c in llm_schema.charts if c.type == target]
                if quantity > 1:
                    while len(kept) < quantity:
                        extra = self._fallback_chart_config_for_index(
                            target, len(kept), working_data, num_cols=_num_cols_cache
                        )
                        if extra is None:
                            break
                        from app.schemas.chart_schema import ChartConfigSchema
                        try:
                            kept.append(ChartConfigSchema.model_validate(extra))
                        except Exception:
                            break
                    kept = kept[:quantity]
                llm_schema.charts = kept
            else:
                kept, covered = [], set()
                for chart in llm_schema.charts:
                    if chart.type in requested_types and chart.type not in covered:
                        kept.append(chart)
                        covered.add(chart.type)
                for t in requested_types:
                    if t not in covered:
                        fc = self._fallback_chart_config(t, working_data, num_cols=_num_cols_cache)
                        if fc:
                            from app.schemas.chart_schema import ChartConfigSchema
                            try:
                                kept.append(ChartConfigSchema.model_validate(fc))
                            except Exception:
                                pass
                llm_schema.charts = kept

        elif mode == "all_of_type":
            # Keep only charts of the requested type(s); strip everything else.
            # The LLM was already instructed (via _augment_query) to generate all
            # meaningful variants — here we just enforce the type filter and ensure
            # scorecards / tables are cleared.
            kept = [c for c in llm_schema.charts if c.type in requested_types]
            if not kept:
                # LLM didn't comply — build one fallback per requested type
                from app.schemas.chart_schema import ChartConfigSchema
                for t in requested_types:
                    fc = self._fallback_chart_config(t, working_data, num_cols=_num_cols_cache)
                    if fc:
                        try:
                            kept.append(ChartConfigSchema.model_validate(fc))
                        except Exception:
                            pass
            llm_schema.charts  = kept
            llm_schema.tables  = []   # user only asked for charts
            scorecards         = []   # user only asked for charts
 
        charts = self._build_charts(
            llm_schema,
            working_normalizer,
            working_builder,
            working_data,
            requested_types=requested_types,
            num_cols=_num_cols_cache,
        )
        charts = self._post_process_charts(charts, working_data, stats=working_normalizer._stats)

        # ── Per-mode chart cap ────────────────────────────────────────────────
        chart_warning = None
        if mode == "exploratory" and len(charts) > CHART_MAX_EXPLORATORY:
            logger.info("Capping exploratory charts %d → %d", len(charts), CHART_MAX_EXPLORATORY)
            chart_warning = _over_limit_msg("charts", len(charts), CHART_MAX_EXPLORATORY)
            charts = charts[:CHART_MAX_EXPLORATORY]
        elif mode == "all_of_type" and len(charts) > CHART_MAX_ALL_OF_TYPE:
            logger.info("Capping all_of_type charts %d → %d", len(charts), CHART_MAX_ALL_OF_TYPE)
            chart_warning = _over_limit_msg("charts", len(charts), CHART_MAX_ALL_OF_TYPE)
            charts = charts[:CHART_MAX_ALL_OF_TYPE]
        elif mode == "multi" and len(charts) > CHART_MAX_MULTI:
            logger.info("Capping multi charts %d → %d", len(charts), CHART_MAX_MULTI)
            chart_warning = _over_limit_msg("charts", len(charts), CHART_MAX_MULTI)
            charts = charts[:CHART_MAX_MULTI]
        # ── end cap ───────────────────────────────────────────────────────────

        # ── FIX 3: HARD STOP — never return a silent empty result ────────────
        # (tables_only and scorecards_only return early above — this guard is
        #  only reached by chart-producing modes)
        if not charts:
            meaningful = _num_cols_cache   # already computed above — no extra call
            if not meaningful:
                raise ValueError(
                    "This dataset has no plottable numeric columns. "
                    "Columns like TransactionID, CustomerID, and Zip are "
                    "identifiers, not metrics. Upload a dataset that contains "
                    "real business metrics (e.g. Sales, Profit, Quantity)."
                )
            raise ValueError(
                "No renderable charts could be built from the LLM configuration "
                "or any fallback. This usually means the requested chart type is "
                "incompatible with the available columns. "
                f"Available numeric columns: {meaningful}."
            )
        # ── end FIX 3 ─────────────────────────────────────────────────────────
 
        table_cfgs = self._filter_table_configs(llm_schema.tables)[:2]
        tables     = working_table_builder.build_all(table_cfgs)
        if not plan.show_tables:
            tables = []
 
        result = {"scorecards": scorecards, "charts": charts, "tables": tables}
        if chart_warning:
            result["warning"] = chart_warning
        logger.info(
            "_process_sync END  mode=%s  charts=%d  tables=%d  scorecards=%d  elapsed=%.2fs",
            mode, len(charts), len(tables), len(scorecards),
            time.perf_counter() - t0,
        )
        return result

    def _post_process_charts(self, charts: list, df: pd.DataFrame, stats=None) -> list:
        filtered = []
        for c in charts:
            if c.get("type") == "scatter":
                x_col = c.get("x_label") or c.get("x")
                y_col = c.get("y_label") or c.get("y")
                if x_col in df.columns and y_col in df.columns:
                    # Use pre-computed cardinality when available — avoids nunique() scan
                    x_card = stats.card(x_col) if stats else df[x_col].nunique()
                    y_card = stats.card(y_col) if stats else df[y_col].nunique()
                    if x_card < 5 or y_card < 5:
                        continue
            filtered.append(c)
        unique, seen = [], set()
        for c in filtered:
            key = (c.get("type"), c.get("x_label"), c.get("y_label"))
            if key not in seen:
                seen.add(key); unique.append(c)
        return unique

    def _build_charts(self, llm_schema, normalizer=None, builder=None, working_data=None, requested_types=None, num_cols=None) -> list:
        if normalizer is None:   normalizer   = self._normalizer
        if builder is None:      builder      = self._builder
        if working_data is None: working_data = self.data
        # Use pre-computed list when available — avoids re-running the logged call
        if num_cols is None:     num_cols     = _meaningful_numeric_cols(working_data)

        # Sequential loop — pandas ops are GIL-bound so thread-per-chart
        # parallelism adds context-switching overhead with zero parallelism gain.
        # The groupby cache on DataTransformer also only works correctly when
        # charts are built sequentially on the same instance.
        charts = []
        for chart_schema in llm_schema.charts:
            try:
                cfg   = normalizer.normalize(chart_schema)
                chart = builder.build(cfg) if cfg else None
                if chart:
                    charts.append(chart)
            except Exception as exc:
                logger.warning("Chart build failed (%s): %s", getattr(chart_schema, 'type', '?'), exc)

        if charts:
            return charts

        # Type-aware fallback
        for chart_type in (requested_types or []):
            fc = self._fallback_chart_config(chart_type, working_data, num_cols=num_cols)
            if fc:
                from app.schemas.chart_schema import ChartConfigSchema
                try:
                    schema = ChartConfigSchema.model_validate(fc)
                    cfg = normalizer.normalize(schema)
                    if cfg:
                        chart = builder.build(cfg)
                        if chart:
                            logger.info("Type-aware fallback succeeded for %r", chart_type)
                            return [chart]
                except Exception:
                    pass

        return self._fallback_chart(working_data, num_cols=num_cols)

    def _fallback_chart(self, df=None, num_cols=None) -> list:
        df = df if df is not None else self.data
        # FIX 1: _meaningful_numeric_cols — never returns TransactionID etc.
        # Accept pre-computed list to avoid redundant logged calls.
        if num_cols is None:
            num_cols = _meaningful_numeric_cols(df)
        if not num_cols:
            logger.error(
                "_fallback_chart: no meaningful numeric columns in dataset — "
                "cannot produce any chart"
            )
            return []  # pipeline will raise in _process_sync
        def _score_column(df, col):
            s = df[col].dropna()

            # 1. Must have enough variation
            uniq = s.nunique()
            if uniq < 5:
                return -1

            # 2. Prefer wider spread (more informative)
            spread = s.max() - s.min() if len(s) else 0

            # 3. Penalize low variance (flat columns)
            std = s.std() if len(s) > 1 else 0

            # 4. Penalize columns that look like IDs (just in case)
            # (safety layer even though you filtered already)
            if uniq / len(s) > 0.95:
                return -2

            return spread + std


        scored = [(col, _score_column(df, col)) for col in num_cols]
        scored = [c for c in scored if c[1] > 0]

        if not scored:
            return []

        col = sorted(scored, key=lambda x: x[1], reverse=True)[0][0]

        values = df[col].dropna()
        if len(values) > 1000:
            values = values.sample(1000, random_state=0)
        logger.info("_fallback_chart: using column %r", col)
        return [{
            "type":        "histogram",
            "title":       f"Distribution of {col}",
            "values":      values.tolist(),
            "x_label":     col,
            "layout_size": "medium",
        }]

    def _fallback_chart_config(self, chart_type: str, df=None, num_cols=None) -> dict | None:
        df = df if df is not None else self.data
        # FIX 1: use filtered list — never includes ID columns
        # Accept pre-computed list to avoid redundant logged calls.
        if num_cols is None:
            num_cols = _meaningful_numeric_cols(df)
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
 
        if chart_type == "histogram" and num_cols:
            return {
                "type": "histogram", "x": num_cols[0], "y": None,
                "color": None, "aggregation": "none",
                "time_granularity": "none", "layout_size": "medium",
                "title": f"Distribution of {num_cols[0]}",
            }
        if chart_type in ("area", "line") and len(num_cols) >= 2:
            return {
                "type": chart_type, "x": num_cols[0], "y": num_cols[1],
                "color": None, "aggregation": "none",
                "time_granularity": "none", "layout_size": "large",
                "title": f"{num_cols[1]} trend",
            }
        if chart_type in ("pie", "funnel", "treemap") and cat_cols and num_cols:
            return {
                "type": chart_type, "x": cat_cols[0], "y": num_cols[0],
                "color": None, "aggregation": "sum",
                "time_granularity": "none", "layout_size": "medium",
                "title": f"{num_cols[0]} by {cat_cols[0]}",
            }
        if chart_type in ("bar", "stacked_bar", "grouped_bar", "waterfall") and cat_cols and num_cols:
            return {
                "type": chart_type, "x": cat_cols[0], "y": num_cols[0],
                "color": None, "aggregation": "sum",
                "time_granularity": "none", "layout_size": "medium",
                "title": f"{num_cols[0]} by {cat_cols[0]}",
            }
        if chart_type == "scatter" and len(num_cols) >= 2:
            return {
                "type": "scatter", "x": num_cols[0], "y": num_cols[1],
                "color": None, "aggregation": "none",
                "time_granularity": "none", "layout_size": "medium",
                "title": f"{num_cols[0]} vs {num_cols[1]}",
            }
        if chart_type == "bubble" and len(num_cols) >= 3:
            return {
                "type": "bubble", "x": num_cols[0], "y": num_cols[1],
                "size": num_cols[2], "color": None, "aggregation": "none",
                "time_granularity": "none", "layout_size": "medium",
                "title": f"{num_cols[0]} vs {num_cols[1]}",
            }
        if chart_type == "heatmap":
            usable_cat = [
                c for c in cat_cols
                if 2 <= df[c].nunique() <= 25
                and c not in num_cols   # use already-computed list
            ]
            if len(usable_cat) >= 2 and num_cols:
                return {
                    "type": "heatmap", "x": usable_cat[0], "y": usable_cat[1],
                    "z": num_cols[0], "color": None, "aggregation": "mean",
                    "time_granularity": "none", "layout_size": "large",
                    "title": f"{num_cols[0]} heatmap",
                }
            return None
        if chart_type == "box" and num_cols:
            return {
                "type": "box", "x": num_cols[0], "y": num_cols[0],
                "color": None, "aggregation": "none",
                "time_granularity": "none", "layout_size": "medium",
                "title": f"Distribution of {num_cols[0]}",
            }
        return None


    def _fallback_chart_config_for_index(
        self, chart_type: str, index: int, df=None, num_cols=None
    ) -> dict | None:
        df = df if df is not None else self.data
        # FIX 1: filtered list — accept pre-computed to avoid redundant logged calls.
        if num_cols is None:
            num_cols = _meaningful_numeric_cols(df)
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
 
        if chart_type == "histogram" and index < len(num_cols):
            col = num_cols[index]
            return {
                "type": "histogram", "x": col, "y": None,
                "color": None, "aggregation": "none",
                "time_granularity": "none", "layout_size": "medium",
                "title": f"Distribution of {col}",
            }
        if chart_type == "box" and index < len(num_cols):
            col = num_cols[index]
            return {
                "type": "box", "x": col, "y": col,
                "color": None, "aggregation": "none",
                "time_granularity": "none", "layout_size": "medium",
                "title": f"Distribution of {col}",
            }
        if chart_type in ("bar", "stacked_bar", "grouped_bar") and cat_cols and index < len(num_cols):
            return {
                "type": chart_type, "x": cat_cols[0], "y": num_cols[index],
                "color": None, "aggregation": "sum",
                "time_granularity": "none", "layout_size": "medium",
                "title": f"{num_cols[index]} by {cat_cols[0]}",
            }
        if chart_type in ("line", "area") and len(num_cols) >= 2:
            col = num_cols[min(index + 1, len(num_cols) - 1)]
            return {
                "type": chart_type, "x": num_cols[0], "y": col,
                "color": None, "aggregation": "none",
                "time_granularity": "none", "layout_size": "large",
                "title": f"{col} trend",
            }
        return self._fallback_chart_config(chart_type, df)