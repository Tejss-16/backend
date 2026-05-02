# app/services/chart_service.py

import asyncio
import concurrent.futures
import logging
import os
import re
import time
from datetime import datetime
from app.utils.column_utils import _meaningful_numeric_cols


import pandas as pd

# _meaningful_numeric_cols logs at INFO on every call. It is called once per
# chart config processed (inside ChartConfigNormalizer.normalize, which runs
# in the thread pool) — producing 10–15 identical lines per query. The function
# itself cannot be changed here, so we cap its logger at WARNING so only genuine
# warnings surface. This is intentional and documented.
logging.getLogger("app.utils.column_utils").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=(os.cpu_count() or 4) * 2,
    thread_name_prefix="chart-worker",
)

from app.pipeline.llm_client import LLMClient
from app.pipeline.normalizer import ChartConfigNormalizer
from app.pipeline.transformer import DataTransformer
from app.pipeline.chart_builder import ChartBuilder
from app.pipeline.table_builder import TableBuilder
from app.pipeline.scorecard import ScorecardBuilder
from app.schemas.chart_schema import LLMResponseSchema
from app.utils.cache import _cache_key, _result_cache
from app.utils.task_manager import is_cancelled


def _set_loop_executor(loop: asyncio.AbstractEventLoop | None = None) -> None:
    (loop or asyncio.get_event_loop()).set_default_executor(EXECUTOR)


_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

# Maximum charts returned per query mode.
# Tune these constants to control dashboard density without touching logic.
CHART_MAX_EXPLORATORY = 12   # open-ended: "dashboard", "analyze", "overview"
CHART_MAX_ALL_OF_TYPE = 10   # wildcard:   "all possible bar charts"
CHART_MAX_MULTI       = 8    # named set:  "histogram and pie and scatter"
# "specific" mode always returns exactly 1 — no cap needed

TABLE_MAX     = 5   # max tables in any single response
SCORECARD_MAX = 8   # max scorecards in any single response (mirrors scorecard.py _SCORECARD_MAX)

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
# ChartGenerator
# ─────────────────────────────────────────────────────────────────────────────

class ChartGenerator:
    def __init__(self, data: pd.DataFrame):
        self.data               = data
        self._col_dt_list       = list(zip(data.columns, data.dtypes))
        self._llm               = LLMClient()
        self._normalizer        = ChartConfigNormalizer(data)
        self._transformer       = DataTransformer(data)
        self._builder           = ChartBuilder(self._transformer)
        self._table_builder     = TableBuilder(data)
        self._scorecard_builder = ScorecardBuilder(data)
        logger.info("ChartGenerator initialised (%d rows, %d cols)", *data.shape)

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
        if len(types) == 1 and quantity == 1: return "specific"
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

    def _augment_query(self, query: str) -> str:
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
                "Return 5–8 meaningful scorecards covering different business dimensions. "
                "Return an empty 'charts' array and empty 'tables' array."
            )

        if mode == "exploratory":
            # FIX 1: filter ID columns out of the column context
            num_cols = _meaningful_numeric_cols(self.data)
            cat_cols = self.data.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()
            dt_cols = [
                c for c in self.data.columns
                if "date" in c.lower() or "time" in c.lower()
                or str(self.data[c].dtype).startswith("datetime")
            ]
            col_ctx = "\n\n[DATASET STRUCTURE — use this to generate a full dashboard]\n"
            if dt_cols:  col_ctx += f"Date/time columns: {dt_cols}\n"
            if num_cols: col_ctx += f"Numeric columns: {num_cols}\n"
            if cat_cols:
                useful = [c for c in cat_cols if self.data[c].nunique() <= 30]
                if useful: col_ctx += f"Categorical columns: {useful}\n"
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
        key = _cache_key(self.data, query)
        if (cached := _result_cache.get(key)) is not None:
            logger.info("Cache hit for query: %r", query[:60])
            return cached

        self._check_cancelled(task_id, "pre-start")

        loop = asyncio.get_running_loop()
        corrected_query = await self._llm.correct_query(query, self._col_dt_list)

        # Apply date filter — FIX 7: returns df with date col as datetime64
        filtered_data = await loop.run_in_executor(
            EXECUTOR, _apply_date_filter, self.data, corrected_query,
        )

        # ALWAYS define working components
        if filtered_data is not self.data:
            working_normalizer    = ChartConfigNormalizer(filtered_data)
            working_transformer   = DataTransformer(filtered_data)
            working_builder       = ChartBuilder(working_transformer)
            working_table_builder = TableBuilder(filtered_data)
            working_scorecard     = ScorecardBuilder(filtered_data)
        else:
            working_normalizer    = self._normalizer
            working_builder       = self._builder
            working_table_builder = self._table_builder
            working_scorecard     = self._scorecard_builder


        # ALWAYS filter columns for LLM (independent of above)
        def _filter_cols(df):
            return [
                (c, dt) for c, dt in zip(df.columns, df.dtypes)
                if c not in df.select_dtypes(include="number").columns
                or c in _meaningful_numeric_cols(df)
            ]

        working_col_dt_list = _filter_cols(filtered_data)

        effective_query = self._augment_query(corrected_query)

        sample, stats = await loop.run_in_executor(
            EXECUTOR,
            lambda: (filtered_data.head(5).to_string(), filtered_data.describe().round(2).to_string()),
        )

        self._check_cancelled(task_id, "pre-llm")

        mode = self._query_mode(corrected_query)

        # ── tables_only: skip get_chart_config entirely ───────────────────────
        # Going through get_chart_config would trigger the "charts must be
        # non-empty" schema validator and waste 2–3 retry/repair rounds.
        if mode == "tables_only":
            requested  = self._extract_non_chart_quantity(corrected_query, "table")
            requested  = self._extract_non_chart_quantity(corrected_query, "tables") if requested == 1 else requested
            capped     = min(requested, TABLE_MAX)
            warning    = _over_limit_msg("tables", requested, TABLE_MAX) if requested > TABLE_MAX else None
            logger.info("tables_only mode — calling get_table_config directly (requested=%d, max=%d)", requested, capped)
            raw_table_cfgs = await self._llm.get_table_config(
                working_col_dt_list, sample, stats, corrected_query,
                max_tables=capped,
            )
            from app.schemas.chart_schema import TableConfigSchema
            from pydantic import ValidationError
            validated = []
            for raw in raw_table_cfgs:
                try:
                    validated.append(TableConfigSchema.model_validate(raw))
                except (ValidationError, Exception) as exc:
                    logger.warning("Table config validation failed: %s", exc)

            tables = working_table_builder.build_all(validated[:capped])
            result = {"scorecards": [], "charts": [], "tables": tables}
            if warning:
                result["warning"] = warning
            logger.info("tables_only completed: %d table(s) built", len(tables))
            self._check_cancelled(task_id, "post-processing")
            _result_cache.set(key, result)
            return result

        # ── scorecards_only: skip get_chart_config entirely ───────────────────
        if mode == "scorecards_only":
            requested = self._extract_non_chart_quantity(corrected_query, "scorecard")
            requested = self._extract_non_chart_quantity(corrected_query, "scorecards") if requested == 1 else requested
            capped    = min(requested if requested > 1 else SCORECARD_MAX, SCORECARD_MAX)
            warning   = _over_limit_msg("scorecards", requested, SCORECARD_MAX) if requested > SCORECARD_MAX else None
            logger.info("scorecards_only mode — calling get_scorecard_config directly (max=%d)", capped)
            raw_scorecard_cfgs = await self._llm.get_scorecard_config(
                working_col_dt_list, sample, stats, corrected_query,
                max_scorecards=capped,
            )
            from app.schemas.chart_schema import ScorecardConfigSchema
            from pydantic import ValidationError
            validated_sc = []
            for raw in raw_scorecard_cfgs:
                try:
                    validated_sc.append(ScorecardConfigSchema.model_validate(raw))
                except (ValidationError, Exception) as exc:
                    logger.warning("Scorecard config validation failed: %s", exc)
            scorecards = working_scorecard.build_from_llm(validated_sc[:capped])
            result = {"scorecards": scorecards, "charts": [], "tables": []}
            if warning:
                result["warning"] = warning
            logger.info("scorecards_only completed: %d scorecard(s) built", len(scorecards))
            self._check_cancelled(task_id, "post-processing")
            _result_cache.set(key, result)
            return result
        # ── end short-circuits ────────────────────────────────────────────────

        try:
            llm_schema = await asyncio.wait_for(
                self._llm.get_chart_config(working_col_dt_list, sample, stats, effective_query),
                timeout=180,
            )
        except asyncio.TimeoutError:
            logger.warning("LLM call timed out for query: %r", query[:60])
            llm_schema = self._llm._fallback_config(working_col_dt_list)

        self._check_cancelled(task_id, "post-llm")

        if self._should_show_tables(corrected_query) and not llm_schema.tables:
            logger.info("No tables from main LLM — requesting separately")
            raw_table_cfgs = await self._llm.get_table_config(
                working_col_dt_list, sample, stats, corrected_query,
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
    ) -> dict:
        """
        Identical to the original _process_sync, with ONE change:
 
        FIX 3 — After _build_charts + all fallback attempts, if `charts` is
        still empty, raise ValueError immediately.  The caller (_run in routes.py)
        catches this, writes {"status": "error", "error": "<message>"} to
        _results, and the frontend shows a clear error toast instead of an
        empty dashboard with no explanation.
 
        The error message tells the operator WHY (no meaningful numeric columns
        or all chart configs were invalid) so it's actionable.
        """
        import time, concurrent.futures
        from app.utils.task_manager import is_cancelled
 
        logger.info("_process_sync START")
        t0 = time.perf_counter()
 
        if working_data is None:          working_data          = self.data
        if working_normalizer is None:    working_normalizer    = self._normalizer
        if working_builder is None:       working_builder       = self._builder
        if working_table_builder is None: working_table_builder = self._table_builder
        if working_scorecard is None:     working_scorecard     = self._scorecard_builder

        # Compute once here — passed into _build_charts and both fallback methods
        # so _meaningful_numeric_cols (which logs on every call) fires exactly once
        # per query instead of once per chart config.
        _num_cols_cache = _meaningful_numeric_cols(working_data)
 
        mode            = self._query_mode(query)
        requested_types = self._detect_requested_chart_types(query)
        quantity        = self._extract_quantity(query)
 
        show_scorecards = self._should_show_scorecards(query)
        scorecards = (
            working_scorecard.build_from_llm(llm_schema.scorecards)
            if show_scorecards else []
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
            if not self._should_show_scorecards(query): scorecards = []
            if not self._should_show_tables(query):     llm_schema.tables = []
 
        elif mode == "multi":
            if quantity > 1 and len(requested_types) == 1:
                target = requested_types[0]
                kept   = [c for c in llm_schema.charts if c.type == target]
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
                llm_schema.charts = kept[:quantity]
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
        charts = self._post_process_charts(charts, working_data)

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
        if not self._should_show_tables(query):
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

    def _post_process_charts(self, charts: list, df: pd.DataFrame) -> list:
        filtered = []
        for c in charts:
            if c.get("type") == "scatter":
                x_col = c.get("x_label") or c.get("x")
                y_col = c.get("y_label") or c.get("y")
                if x_col in df.columns and y_col in df.columns:
                    if df[x_col].nunique() < 5 or df[y_col].nunique() < 5:
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

        def process_one(chart_schema):
            cfg = normalizer.normalize(chart_schema)
            return builder.build(cfg) if cfg else None

        futures = [EXECUTOR.submit(process_one, cs) for cs in llm_schema.charts]
        charts  = [f.result() for f in concurrent.futures.as_completed(futures) if f.result()]

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