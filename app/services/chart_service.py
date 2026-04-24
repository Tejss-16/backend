import asyncio
import concurrent.futures
import logging
import os
import time

import pandas as pd

logger = logging.getLogger(__name__)

# Single shared executor — sized for CPU work across concurrent requests.
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
    """
    Call once at startup so asyncio.to_thread() uses the shared pool.

        from app.pipeline.chart_service import EXECUTOR, _set_loop_executor
        _set_loop_executor()
    """
    (loop or asyncio.get_event_loop()).set_default_executor(EXECUTOR)


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

    # ── Intent keyword sets ──────────────────────────────────────────────────

    _BROAD_INTENT = {
        "dashboard", "analyze", "analyse", "analysis", "analyses",
        "overview", "report", "explore", "exploration",
        "insights", "insight", "understand", "examine", "investigate",
        "full", "complete", "everything", "all", "whole", "entire",
        "show me", "tell me", "give me", "what is", "what are",
        "create dashboard", "make dashboard", "build dashboard",
        "deep dive", "deep-dive",
        "summarize", "summarise",
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

    @staticmethod
    def _query_matches(query: str, keywords: set) -> bool:
        q = query.lower()
        return any(kw in q for kw in keywords)

    def _augment_query(self, query: str) -> str:
        if not self._needs_table(query):
            return query

        num_cols = list(self.data.select_dtypes(include="number").columns[:3])
        cat_cols = list(self.data.select_dtypes(include=["object", "category"]).columns[:2])
        dt_cols  = [c for c in self.data.columns
                    if "date" in c.lower() or "time" in c.lower()
                    or str(self.data[c].dtype).startswith("datetime")]

        chart_hint = (
            f"Date/time columns (use for line charts): {dt_cols}\n"
            f"Numeric columns: {num_cols}\n"
            f"Categorical columns: {cat_cols}\n"
        ) if dt_cols else (
            f"Numeric columns: {num_cols}\n"
            f"Categorical columns: {cat_cols}\n"
        )

        hint = (
            "\n\n[MANDATORY INSTRUCTION - DO NOT IGNORE]\n"
            "Return at least 2-3 diverse charts (line, bar, pie — not all the same type).\n"
            + chart_hint +
            "Tables: MAX 2. Only pivot tables with multiple rows. "
            "NEVER generate a single-scalar summary table — those are shown as scorecards already.\n"
            "Every pivot table MUST have a non-null 'index' and 'values' field."
        )
        logger.info("Augmenting query with chart+table hint for: %r", query[:50])
        return query + hint

    def _needs_table(self, query: str) -> bool:
        return (
            self._query_matches(query, self._BROAD_INTENT)
            or self._query_matches(query, self._SUMMARY_INTENT)
            or self._query_matches(query, self._PIVOT_INTENT)
        )

    def _default_tables(self, query: str) -> list:
        df     = self.data
        tables = []

        num_cols = df.select_dtypes(include="number").columns.tolist()
        if not num_cols:
            return []

        cat_cols = [
            c for c in df.select_dtypes(include=["object", "category"]).columns
            if 2 <= df[c].nunique() <= 30
        ]

        if not cat_cols:
            logger.info("_default_tables: no categorical columns — skipping")
            return []

        cat1, num1 = cat_cols[0], num_cols[0]
        try:
            tbl = (
                df.groupby(cat1, observed=True)[num1]
                .agg(["sum", "mean", "count"])
                .reset_index()
                .rename(columns={"sum": f"Total {num1}", "mean": f"Avg {num1}", "count": "Count"})
                .sort_values(f"Total {num1}", ascending=False)
                .head(20)
            )
            tbl[f"Total {num1}"] = tbl[f"Total {num1}"].round(2)
            tbl[f"Avg {num1}"]   = tbl[f"Avg {num1}"].round(2)
            tables.append({"title": f"{num1} by {cat1}", "data": tbl.to_dict(orient="records")})
        except Exception as exc:
            logger.error("Default table 1 failed: %s", exc)

        if len(tables) < 2:
            if len(cat_cols) >= 2:
                cat2 = cat_cols[1]
                try:
                    pivot = (
                        pd.pivot_table(df, index=cat1, columns=cat2,
                                       values=num_cols[0], aggfunc="sum", observed=True)
                        .fillna(0).reset_index().head(20)
                    )
                    pivot.columns = [
                        str(c) if not isinstance(c, tuple) else "_".join(str(x) for x in c if x)
                        for c in pivot.columns
                    ]
                    tables.append({
                        "title": f"{num_cols[0]} by {cat1} and {cat2}",
                        "data": pivot.to_dict(orient="records"),
                    })
                except Exception as exc:
                    logger.error("Default pivot table failed: %s", exc)
            elif len(num_cols) >= 2:
                num2 = num_cols[1]
                try:
                    tbl2 = (
                        df.groupby(cat1, observed=True)[num2]
                        .agg(["sum", "mean"])
                        .reset_index()
                        .rename(columns={"sum": f"Total {num2}", "mean": f"Avg {num2}"})
                        .sort_values(f"Total {num2}", ascending=False)
                        .head(20)
                    )
                    tbl2[f"Total {num2}"] = tbl2[f"Total {num2}"].round(2)
                    tbl2[f"Avg {num2}"]   = tbl2[f"Avg {num2}"].round(2)
                    tables.append({"title": f"{num2} by {cat1}", "data": tbl2.to_dict(orient="records")})
                except Exception as exc:
                    logger.error("Default table 2 failed: %s", exc)

        logger.info("Built %d default table(s) for query: %r", len(tables), query[:60])
        return tables

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

    # ── Cooperative cancellation ─────────────────────────────────────────────

    @staticmethod
    def _check_cancelled(task_id: str, stage: str) -> None:
        """
        Raise CancelledError if the cancel flag has been set for this task.
        Call this at every async stage boundary — it's instant and has no I/O.

        The LLM HTTP request itself cannot be interrupted mid-flight, so the
        first safe kill points are:
          • before we start any work          (catches instant cancels)
          • after _prepare_llm_input          (before LLM is even called)
          • after the LLM call returns        (discard result if flagged)
          • after _process_sync               (before writing 'completed')
        """
        if is_cancelled(task_id):
            logger.info("Cancel flag detected at stage '%s' for task %s", stage, task_id)
            raise asyncio.CancelledError(f"Cancelled at stage: {stage}")

    # ── Main entry point ─────────────────────────────────────────────────────

    async def generate(self, query: str, task_id: str = "") -> dict:
        # ── Cache check ──────────────────────────────────────────────────────
        key = _cache_key(self.data, query)
        if (cached := _result_cache.get(key)) is not None:
            logger.info("Cache hit for query: %r", query[:60])
            return cached

        # Stage 0: catch immediate cancels before any work starts
        self._check_cancelled(task_id, "pre-start")

        effective_query = self._augment_query(query)
        loop = asyncio.get_running_loop()

        # ── Stage 1: prepare LLM input (CPU — offloaded to thread) ──────────
        sample, stats = await loop.run_in_executor(EXECUTOR, self._prepare_llm_input)

        # Stage boundary: cancel before paying LLM API costs
        self._check_cancelled(task_id, "pre-llm")

        # ── Stage 2: LLM call ────────────────────────────────────────────────
        # wait_for enforces a hard timeout. The underlying HTTP request cannot
        # be killed once sent, but wait_for lets CancelledError propagate here
        # instead of inside the LLM library.
        try:
            llm_schema = await asyncio.wait_for(
                self._llm.get_chart_config(
                    self._col_dt_list, sample, stats, effective_query
                ),
                timeout=180,
            )
        except asyncio.TimeoutError:
            logger.warning("LLM call timed out for query: %r", query[:60])
            raise Exception("LLM timed out — please try again")

        # Stage boundary: LLM is done; check flag before paying CPU costs.
        # This is the most important check — it's where "stop was pressed
        # during LLM call" gets caught cleanly.
        self._check_cancelled(task_id, "post-llm")

        # ── Stage 3: CPU processing (offloaded to thread) ────────────────────
        result = await loop.run_in_executor(
            EXECUTOR,
            self._process_sync,
            llm_schema,
            query,
        )

        # Stage boundary: CPU done; last check before caching + returning.
        self._check_cancelled(task_id, "post-processing")

        _result_cache.set(key, result)
        return result

    def _prepare_llm_input(self):
        sample = self.data.head(5).to_string()
        stats  = self.data.describe(include="all").to_string()
        return sample, stats

    def _process_sync(self, llm_schema: LLMResponseSchema, query: str) -> dict:
        """
        CPU-bound processing step.  Runs inside a ThreadPoolExecutor thread.

        Cancellation note
        -----------------
        asyncio.CancelledError cannot propagate into a thread — it will only
        arrive back in the event loop after run_in_executor() returns.  We
        therefore add a lightweight cooperative check via a threading.Event
        if you need finer-grained interruption in the future; for now, the
        yield points in `generate()` cover all stage boundaries.
        """
        logger.info("_process_sync START")
        t0 = time.perf_counter()

        scorecards = self._scorecard_builder.build()

        charts = self._build_charts(llm_schema)

        table_cfgs = llm_schema.tables
        table_cfgs = self._filter_table_configs(table_cfgs)
        table_cfgs = table_cfgs[:2]

        if not table_cfgs and self._needs_table(query):
            logger.info("LLM returned no tables for broad query — injecting defaults")
            table_cfgs = self._default_tables(query)

        tables = self._table_builder.build_all(table_cfgs)

        result = {"scorecards": scorecards, "charts": charts, "tables": tables}
        logger.info(
            "_process_sync END  charts=%d  tables=%d  elapsed=%.2fs",
            len(charts), len(tables), time.perf_counter() - t0,
        )
        return result

    def _build_charts(self, llm_schema: LLMResponseSchema) -> list:
        def process_one(chart_schema):
            cfg = self._normalizer.normalize(chart_schema)
            if cfg is None:
                return None
            return self._builder.build(cfg)

        futures = [EXECUTOR.submit(process_one, cs) for cs in llm_schema.charts]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
        charts  = [r for r in results if r is not None]
        return charts or self._fallback_chart()

    def _fallback_chart(self) -> list:
        num_cols = self.data.select_dtypes(include="number").columns
        if not len(num_cols):
            return []
        col    = num_cols[0]
        values = self.data[col].dropna()
        if len(values) > 1000:
            values = values.sample(1000, random_state=0)
        logger.info("Using fallback histogram for column %r", col)
        return [{
            "type": "histogram",
            "title": f"Distribution of {col}",
            "values": values.tolist(),
            "x_label": col,
            "layout_size": "medium",
        }]