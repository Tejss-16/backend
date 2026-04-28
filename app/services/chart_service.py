# app/services/chart_service.py
#
# FIXES in this version (cumulative — includes all previous fixes):
#
# FIX 1–5: (unchanged from previous version)
# FIX 6: Date-range filtering from natural language queries (previous version)
# FIX 7 (NEW — pairs with transformer FIX A/B/C/D): _apply_date_filter() now
#   stores the parsed datetime Series back into the filtered DataFrame under
#   the original column name.  This means the transformer receives a proper
#   datetime64 column instead of raw strings, so:
#     a) _is_time() immediately recognises it → no re-parsing overhead
#     b) _sort_chronologically() sorts datetime64 directly → correct order
#     c) The string-sort bug that scrambled the x-axis is eliminated

import asyncio
import concurrent.futures
import logging
import os
import re
import time
from datetime import datetime

import pandas as pd

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
        q = query.lower()
        digit_match = re.search(r'\b(\d+)\s+(?:' + '|'.join(
            re.escape(k) for k in sorted(ChartGenerator._CHART_TYPE_KEYWORDS, key=len, reverse=True)
        ) + r')', q)
        if digit_match:
            return max(1, int(digit_match.group(1)))
        word_pattern = '|'.join(re.escape(w) for w in _NUMBER_WORDS)
        word_match = re.search(
            r'\b(' + word_pattern + r')\s+(?:' + '|'.join(
                re.escape(k) for k in sorted(ChartGenerator._CHART_TYPE_KEYWORDS, key=len, reverse=True)
            ) + r')', q
        )
        if word_match:
            return _NUMBER_WORDS.get(word_match.group(1), 1)
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
        if len(types) == 0:   return "exploratory"
        if len(types) == 1 and quantity == 1: return "specific"
        return "multi"

    def _should_show_scorecards(self, query: str) -> bool:
        if self._query_matches(query, self._SCORECARD_EXPLICIT): return True
        return self._query_matches(query, self._BROAD_INTENT | self._SUMMARY_INTENT)

    def _should_show_tables(self, query: str) -> bool:
        if self._query_matches(query, self._TABLE_EXPLICIT): return True
        return self._query_matches(query, self._BROAD_INTENT | self._PIVOT_INTENT | self._SUMMARY_INTENT)

    def _needs_table(self, query: str) -> bool:
        return self._query_matches(query, self._BROAD_INTENT | self._PIVOT_INTENT | self._TABLE_EXPLICIT)

    def _augment_query(self, query: str) -> str:
        mode     = self._query_mode(query)
        quantity = self._extract_quantity(query)
        types    = self._detect_requested_chart_types(query)

        if mode == "exploratory":
            num_cols = list(self.data.select_dtypes(include="number").columns)
            cat_cols = list(self.data.select_dtypes(include=["object", "category"]).columns)
            dt_cols  = [c for c in self.data.columns
                        if "date" in c.lower() or "time" in c.lower()
                        or str(self.data[c].dtype).startswith("datetime")]
            col_ctx  = "\n\n[DATASET STRUCTURE — use this to generate a full dashboard]\n"
            if dt_cols:    col_ctx += f"Date/time columns: {dt_cols}\n"
            if num_cols:   col_ctx += f"Numeric columns: {num_cols}\n"
            if cat_cols:
                useful = [c for c in cat_cols if self.data[c].nunique() <= 30]
                if useful: col_ctx += f"Categorical columns: {useful}\n"
            col_ctx += ("\nGenerate as many charts as makes sense for a complete dashboard. "
                        "Cover time trends, category comparisons, and distributions.")
            return query + col_ctx

        if quantity > 1 and len(types) == 1:
            hint = (f"\n\n[QUANTITY INSTRUCTION]\nThe user explicitly requested {quantity} {types[0]} charts. "
                    f"You MUST return exactly {quantity} charts of type '{types[0]}', "
                    f"each using a DIFFERENT column or dimension. Do NOT return fewer.")
            return query + hint

        if self._needs_table(query):
            return query + ("\n\n[TABLE INSTRUCTION]\nInclude up to 2 pivot tables. "
                            "Each table must have a non-null 'index' and 'values' field and produce multiple rows.")

        if self._query_matches(query, self._SCORECARD_EXPLICIT):
            return query + "\n\n[NOTE] The user also wants KPI scorecards shown above the charts."

        return query

    def _filter_table_configs(self, table_cfgs: list) -> list:
        filtered = []
        for cfg in table_cfgs:
            if isinstance(cfg, dict):
                filtered.append(cfg)
            elif cfg.type == "summary":
                logger.info("Dropping scalar summary table %r — covered by scorecards", cfg.title)
            else:
                filtered.append(cfg)
        return filtered

    @staticmethod
    def _check_cancelled(task_id: str, stage: str) -> None:
        if is_cancelled(task_id):
            logger.info("Cancel flag detected at stage '%s' for task %s", stage, task_id)
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

        if filtered_data is not self.data:
            working_normalizer    = ChartConfigNormalizer(filtered_data)
            working_transformer   = DataTransformer(filtered_data)
            working_builder       = ChartBuilder(working_transformer)
            working_table_builder = TableBuilder(filtered_data)
            working_scorecard     = ScorecardBuilder(filtered_data)
            working_col_dt_list   = list(zip(filtered_data.columns, filtered_data.dtypes))
            logger.info("Using filtered dataset (%d rows) for %r", len(filtered_data), corrected_query[:60])
        else:
            working_normalizer    = self._normalizer
            working_builder       = self._builder
            working_table_builder = self._table_builder
            working_scorecard     = self._scorecard_builder
            working_col_dt_list   = self._col_dt_list

        effective_query = self._augment_query(corrected_query)

        sample, stats = await loop.run_in_executor(
            EXECUTOR,
            lambda: (filtered_data.head(5).to_string(), filtered_data.describe().round(2).to_string()),
        )

        self._check_cancelled(task_id, "pre-llm")

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
                working_col_dt_list, sample, stats, corrected_query
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
        llm_schema: LLMResponseSchema,
        query: str,
        working_data: pd.DataFrame | None = None,
        working_normalizer=None,
        working_builder=None,
        working_table_builder=None,
        working_scorecard=None,
    ) -> dict:
        logger.info("_process_sync START")
        t0 = time.perf_counter()

        if working_data is None:           working_data          = self.data
        if working_normalizer is None:     working_normalizer    = self._normalizer
        if working_builder is None:        working_builder       = self._builder
        if working_table_builder is None:  working_table_builder = self._table_builder
        if working_scorecard is None:      working_scorecard     = self._scorecard_builder

        mode            = self._query_mode(query)
        requested_types = self._detect_requested_chart_types(query)
        quantity        = self._extract_quantity(query)

        show_scorecards = self._should_show_scorecards(query)
        scorecards = working_scorecard.build_from_llm(llm_schema.scorecards) if show_scorecards else []

        if mode == "specific":
            target   = requested_types[0]
            matching = [c for c in llm_schema.charts if c.type == target]
            if not matching:
                fallback_cfg = self._fallback_chart_config(target, working_data)
                if fallback_cfg:
                    from app.schemas.chart_schema import ChartConfigSchema
                    try:   matching = [ChartConfigSchema.model_validate(fallback_cfg)]
                    except Exception: pass
            if not matching:
                return {"scorecards": [], "charts": [{"type": "not_possible", "requested_type": target}], "tables": []}
            llm_schema.charts = matching[:1]
            if not self._should_show_scorecards(query): scorecards = []
            if not self._should_show_tables(query):     llm_schema.tables = []

        elif mode == "multi":
            if quantity > 1 and len(requested_types) == 1:
                target = requested_types[0]
                kept   = [c for c in llm_schema.charts if c.type == target]
                while len(kept) < quantity:
                    extra = self._fallback_chart_config_for_index(target, len(kept), working_data)
                    if extra is None: break
                    from app.schemas.chart_schema import ChartConfigSchema
                    try:   kept.append(ChartConfigSchema.model_validate(extra))
                    except Exception: break
                llm_schema.charts = kept[:quantity]
            else:
                kept, covered = [], set()
                for chart in llm_schema.charts:
                    if chart.type in requested_types and chart.type not in covered:
                        kept.append(chart); covered.add(chart.type)
                for t in requested_types:
                    if t not in covered:
                        fc = self._fallback_chart_config(t, working_data)
                        if fc:
                            from app.schemas.chart_schema import ChartConfigSchema
                            try:   kept.append(ChartConfigSchema.model_validate(fc))
                            except Exception: pass
                llm_schema.charts = kept

        charts = self._build_charts(llm_schema, working_normalizer, working_builder, working_data)
        charts = self._post_process_charts(charts, working_data)

        table_cfgs = self._filter_table_configs(llm_schema.tables)[:2]
        tables     = working_table_builder.build_all(table_cfgs)
        if not self._should_show_tables(query): tables = []

        result = {"scorecards": scorecards, "charts": charts, "tables": tables}
        logger.info(
            "_process_sync END  mode=%s  charts=%d  tables=%d  scorecards=%d  elapsed=%.2fs",
            mode, len(charts), len(tables), len(scorecards), time.perf_counter() - t0,
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

    def _build_charts(self, llm_schema, normalizer=None, builder=None, working_data=None) -> list:
        if normalizer is None:   normalizer   = self._normalizer
        if builder is None:      builder      = self._builder
        if working_data is None: working_data = self.data

        def process_one(chart_schema):
            cfg = normalizer.normalize(chart_schema)
            return builder.build(cfg) if cfg else None

        futures = [EXECUTOR.submit(process_one, cs) for cs in llm_schema.charts]
        charts  = [f.result() for f in concurrent.futures.as_completed(futures) if f.result()]
        return charts or self._fallback_chart(working_data)

    def _fallback_chart(self, df=None) -> list:
        df = df if df is not None else self.data
        num_cols = df.select_dtypes(include="number").columns
        if not len(num_cols): return []
        col = num_cols[0]; values = df[col].dropna()
        if len(values) > 1000: values = values.sample(1000, random_state=0)
        return [{"type": "histogram", "title": f"Distribution of {col}", "values": values.tolist(), "x_label": col, "layout_size": "medium"}]

    def _fallback_chart_config(self, chart_type: str, df=None) -> dict | None:
        df = df if df is not None else self.data
        num_cols = df.select_dtypes(include="number").columns
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        if chart_type == "histogram" and len(num_cols):
            return {"type":"histogram","x":num_cols[0],"y":None,"color":None,"aggregation":"none","time_granularity":"none","layout_size":"medium","title":f"Distribution of {num_cols[0]}"}
        if chart_type in ("area","line") and len(num_cols)>=2:
            return {"type":chart_type,"x":num_cols[0],"y":num_cols[1],"color":None,"aggregation":"none","time_granularity":"none","layout_size":"large","title":f"{num_cols[1]} trend"}
        if chart_type in ("pie","funnel","treemap") and len(cat_cols) and len(num_cols):
            return {"type":chart_type,"x":cat_cols[0],"y":num_cols[0],"color":None,"aggregation":"sum","time_granularity":"none","layout_size":"medium","title":f"{num_cols[0]} by {cat_cols[0]}"}
        if chart_type in ("bar","stacked_bar","grouped_bar","waterfall") and len(cat_cols) and len(num_cols):
            return {"type":chart_type,"x":cat_cols[0],"y":num_cols[0],"color":None,"aggregation":"sum","time_granularity":"none","layout_size":"medium","title":f"{num_cols[0]} by {cat_cols[0]}"}
        if chart_type == "scatter" and len(num_cols)>=2:
            return {"type":"scatter","x":num_cols[0],"y":num_cols[1],"color":None,"aggregation":"none","time_granularity":"none","layout_size":"medium","title":f"{num_cols[0]} vs {num_cols[1]}"}
        if chart_type == "bubble" and len(num_cols)>=3:
            return {"type":"bubble","x":num_cols[0],"y":num_cols[1],"size":num_cols[2],"color":None,"aggregation":"none","time_granularity":"none","layout_size":"medium","title":f"{num_cols[0]} vs {num_cols[1]}"}
        if chart_type == "heatmap":
            usable_cat = [c for c in cat_cols if 2 <= df[c].nunique() <= 25]
            if len(usable_cat)>=2 and len(num_cols)>=1:
                return {"type":"heatmap","x":usable_cat[0],"y":usable_cat[1],"z":num_cols[0],"color":None,"aggregation":"mean","time_granularity":"none","layout_size":"large","title":f"{num_cols[0]} heatmap"}
            return None
        if chart_type == "box" and len(num_cols):
            return {"type":"box","x":num_cols[0],"y":num_cols[0],"color":None,"aggregation":"none","time_granularity":"none","layout_size":"medium","title":f"Distribution of {num_cols[0]}"}
        return None

    def _fallback_chart_config_for_index(self, chart_type: str, index: int, df=None) -> dict | None:
        df = df if df is not None else self.data
        num_cols = list(df.select_dtypes(include="number").columns)
        cat_cols = list(df.select_dtypes(include=["object","category"]).columns)
        if chart_type == "histogram" and index < len(num_cols):
            col = num_cols[index]
            return {"type":"histogram","x":col,"y":None,"color":None,"aggregation":"none","time_granularity":"none","layout_size":"medium","title":f"Distribution of {col}"}
        if chart_type == "box" and index < len(num_cols):
            col = num_cols[index]
            return {"type":"box","x":col,"y":col,"color":None,"aggregation":"none","time_granularity":"none","layout_size":"medium","title":f"Distribution of {col}"}
        if chart_type in ("bar","stacked_bar","grouped_bar") and cat_cols and index < len(num_cols):
            return {"type":chart_type,"x":cat_cols[0],"y":num_cols[index],"color":None,"aggregation":"sum","time_granularity":"none","layout_size":"medium","title":f"{num_cols[index]} by {cat_cols[0]}"}
        if chart_type in ("line","area") and len(num_cols)>=2:
            col = num_cols[min(index+1, len(num_cols)-1)]
            return {"type":chart_type,"x":num_cols[0],"y":col,"color":None,"aggregation":"none","time_granularity":"none","layout_size":"large","title":f"{col} trend"}
        return self._fallback_chart_config(chart_type, df)