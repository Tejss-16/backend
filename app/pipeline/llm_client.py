# app/pipeline/llm_client.py

import json
import os
import re
import logging
logger = logging.getLogger(__name__)
from openai import AsyncOpenAI
from dotenv import load_dotenv
load_dotenv()
from pydantic import ValidationError

from app.schemas.chart_schema import LLMResponseSchema


# ─────────────────────────────────────────────────────────────────────────────
# AGGREGATION INFERENCE  (pure Python — no LLM needed)
# Maps column name patterns and dtype to a default aggregation.
# Used to post-correct LLM-chosen aggregations and to build scorecard hints.
# ─────────────────────────────────────────────────────────────────────────────

_AGG_SUM = frozenset({
    "revenue", "sales", "profit", "amount", "cost", "price", "total",
    "income", "gross", "net", "spend", "spending", "expenditure",
    "quantity", "qty", "units", "items", "sold", "volume",
    "transactions", "orders", "count",
})
_AGG_MEAN = frozenset({
    "rate", "ratio", "percent", "percentage", "pct", "margin",
    "discount", "rating", "score", "age", "tenure", "duration",
    "avg", "average", "mean", "satisfaction", "nps",
})
# Only pure identifier columns — NOT business metrics like "orders", "quantity"
_AGG_COUNT = frozenset({
    "id", "key", "code",
})

# Columns whose names suggest they are physical/operational attributes that
# should not appear as KPI scorecards (neither a business metric nor an ID).
# Used by _is_kpi_column() to filter scorecard candidates.
_NON_KPI_SUBSTRINGS = frozenset({
    "area", "size", "weight", "height", "width", "length", "depth",
    "latitude", "longitude", "lat", "lon", "lng", "zip", "postal",
    "phone", "fax", "email", "url", "address", "description",
    "notes", "comment", "remark", "flag", "status_code",
})


def _is_kpi_column(col_name: str) -> bool:
    """
    Return True when a column is a meaningful business KPI (safe to scorecard).
    Returns False for physical/operational attributes and pure identifiers.
    """
    lower = col_name.lower()
    # Disallow pure identifier columns
    if any(h in lower for h in _AGG_COUNT):
        return False
    # Disallow non-KPI physical/operational attributes
    if any(h in lower for h in _NON_KPI_SUBSTRINGS):
        return False
    return True


def _infer_aggregation(col_name: str, dtype_str: str) -> str:
    """
    Return the most appropriate aggregation for a column based on its name
    and dtype — without involving the LLM.

    Returns one of: "sum", "mean", "count", "none"

    Precedence (fixed): sum > mean > count > dtype fallback > "none"

    Previously mean was checked first, causing columns like "StoreArea" to
    fall through to float→mean and appear as useless KPI scorecards.
    Now sum is checked first so business metrics (revenue, quantity, profit)
    are always aggregated correctly, and mean only wins when no sum hint exists.
    """
    lower = col_name.lower()
    # 1. Sum signals win outright — business volume metrics
    if any(h in lower for h in _AGG_SUM):
        return "sum"
    # 2. Mean signals — rates, ratios, averages
    if any(h in lower for h in _AGG_MEAN):
        return "mean"
    # 3. Count signals — pure identifier columns
    if any(h in lower for h in _AGG_COUNT):
        return "count"
    # 4. dtype fallback: float → mean, int → sum
    if "float" in dtype_str:
        return "mean"
    if "int" in dtype_str:
        return "sum"
    return "none"


def _build_agg_hint(col_dt_list: list) -> str:
    """
    Build a compact aggregation-hint string for the LLM user message.
    Format: "col1→sum col2→mean col3→count ..."
    Gives the model the correct aggregation without it needing to reason.

    Only emits hints for genuine KPI columns (_is_kpi_column).
    Non-KPI physical/operational attributes (StoreArea, Latitude, etc.) are
    excluded so the LLM never proposes them as scorecard candidates.
    """
    parts = []
    for col, dt in col_dt_list:
        if not _is_kpi_column(col):
            continue
        agg = _infer_aggregation(col, str(dt))
        if agg != "none":
            parts.append(f"{col}→{agg}")
    return "Aggregation hints: " + ", ".join(parts) if parts else ""


# ─────────────────────────────────────────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a data visualization expert. Design charts, scorecards, and tables for a dataset.
Return ONLY valid JSON — no explanation, no markdown, no extra text.

OUTPUT FORMAT
{"scorecards":[{"column":"col","aggregation":"sum|mean|count|min|max","label":"...","subtitle":"..."}],"charts":[{"type":"...","x":"col","y":"col or null","z":"col or null","size":"col or null","color":"col or null","aggregation":"sum|mean|count|none","time_granularity":"day|week|month|year|none","layout_size":"small|medium|large","title":"..."}],"tables":[]}

SCORECARDS
Include 4-6 KPI scorecards. Each scorecard MUST represent a genuine business metric.
SKIP: identifier columns (ID, Key, Code), physical/operational attributes (Area, Size, Weight,
Height, Width, Latitude, Longitude, Zip, Phone), and any column whose sum or mean has no
meaningful business interpretation.
PREFER: revenue, sales, profit, cost, quantity/units sold, orders, margin, discount, rating.
subtitle: short contextual annotation (e.g. "58.6% margin", "2016–2021") or "".
If aggregation hints are provided, follow them exactly — do not use columns absent from hints.

⚠️  DERIVED COLUMNS: If the dataset profile lists any column whose name follows the pattern
"<col>_x_<col>" (e.g. "Price_x_Quantity", "Rate_x_Hours"), that column is a pre-computed product
of two raw columns and represents the TRUE aggregated metric — prefer it over the raw factor columns.
More generally: never sum a per-row rate column (price-per-item, rate, fee-per-unit) on its own —
summing per-row rates produces a meaningless total. Use the pre-multiplied column instead.

CHART TYPES
bar,line,area,scatter,histogram,pie,box,heatmap,bubble,funnel,treemap,waterfall,stacked_bar,grouped_bar
time trends→line/area | category compare→bar/grouped_bar | part-of-whole→pie/treemap | correlation→scatter | distribution→histogram/box | 2-dim matrix→heatmap | stage funnel→funnel | cumulative→waterfall

RULES
1. Use ONLY column names from the user message. Never invent columns.
2. y=null ONLY for histogram. All other types (bar,line,area,scatter,pie,box,heatmap,bubble,funnel,treemap,waterfall,stacked_bar,grouped_bar) require a non-null y.
3. heatmap: x,y=categorical, z=numeric. bubble: size,x,y=numeric.
4. funnel: only if a genuine stage/step column exists — else use bar or line.
5. color: only for category ≤ 6 unique values on line/bar/area/stacked_bar/grouped_bar.
6. layout_size: large=time-series/heatmap/bubble, small=pie(2-4 slices), medium=everything else.
7. No two charts with identical x+y.
8. If aggregation hints are provided, follow them exactly — do not override.

EXPLORATORY ORDER (when MODE=EXPLORATORY)
1st chart: large line/area over time (if date col exists).
2nd chart: bar/grouped_bar of primary metric by primary category.
3rd+: pie/treemap for low-cardinality splits. histogram/box/scatter LAST.

TABLES
≤2 tables. Prefer charts. Pivot: {"type":"pivot","index":"cat_col","columns":null,"values":"num_col","aggregation":"sum","title":"..."}
"values" must be a single column name string, NOT a list.
"""

REPAIR_PROMPT = """JSON failed validation.
Error: {error}
Fix ALL errors. Return ONLY valid JSON.
Schema: {{"scorecards":[],"charts":[{{"type":"bar|line|area|scatter|histogram|pie|box|heatmap|bubble|funnel|treemap|waterfall|stacked_bar|grouped_bar","x":"col","y":"col or null","z":null,"size":null,"color":null,"aggregation":"sum|mean|count|none","time_granularity":"none","layout_size":"medium","title":"..."}}],"tables":[]}}
Available columns: {columns}
Fixes: non-histogram needs non-null y; histogram y=null; heatmap needs z; bubble needs size; aggregation must be sum/mean/count/none; layout_size must be small/medium/large.
"""

# Kept short — examples removed since chart-type names are already in SYSTEM_PROMPT
TYPO_CORRECTION_PROMPT = """Fix typos/spelling in the user's data analytics query.
Rules: fix chart-type misspellings (hsitogram→histogram, barchat→bar chart, pei→pie, scater→scatter, heatmpa→heatmap, watterfall→waterfall, funnle→funnel, buble→bubble, dashbord→dashboard). Fix general English typos. NEVER change column names (known: {columns}). NEVER change intent, chart types, numbers, or filters. Return ONLY the corrected query string — no quotes, no explanation. If no errors, return original unchanged."""


def _strip_fences(text: str) -> str:
    return re.sub(r"```(?:json)?", "", text).replace("```", "").strip()


def _parse_and_validate(raw: str) -> LLMResponseSchema:
    data = json.loads(_strip_fences(raw))
    return LLMResponseSchema.model_validate(data)


class LLMClient:
    MAX_RETRIES = 2
    MAX_REPAIRS = 1

    def __init__(self):
        self._client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            timeout=290.0
        )

    # ── Typo correction ───────────────────────────────────────────────────────
    async def correct_query(self, query: str, col_dt_list: list) -> str:
        """
        Lightweight LLM call that fixes typos and normalises phrasing.
        Returns original query on any failure — fully fail-safe.
        """
        columns = [col for col, _ in col_dt_list]
        prompt  = TYPO_CORRECTION_PROMPT.format(columns=columns)
        try:
            corrected = await self._call([
                {"role": "system", "content": "Return ONLY the corrected query string. No quotes, no explanation."},
                {"role": "user",   "content": f"{prompt}\n\nQuery: {query}"},
            ])
            corrected = corrected.strip().strip('"').strip("'")
            if not corrected:
                return query
            if corrected != query:
                logger.info("Query corrected: %r → %r", query, corrected)
            return corrected
        except Exception as exc:
            logger.warning("Typo correction failed (%s) — using original", exc)
            return query

    async def is_data_query(self, query: str, col_dt_list: list) -> tuple[bool, str]:
        """
        Classify whether the query is relevant to data analysis.
        Returns (True, "") on any error — fail-safe.
        """
        columns = [col for col, _ in col_dt_list]
        prompt = (
            f"Dataset columns: {columns}\n"
            f"Query: \"{query}\"\n"
            f"Is this a data analysis query (charts/tables/scorecards/dashboard)? "
            f"Reply ONLY: {{\"relevant\":true}} or {{\"relevant\":false,\"reason\":\"one sentence\"}}"
        )
        try:
            raw = await self._call([
                {"role": "system", "content": "Strict classifier. Reply ONLY with valid JSON."},
                {"role": "user",   "content": prompt},
            ])
            clean = re.sub(r"```(?:json)?|```", "", raw).strip()
            data  = json.loads(clean)
            if data.get("relevant", True):
                return True, ""
            reason = data.get("reason") or (
                "I can only help with data analysis — charts, tables, scorecards, and dashboards."
            )
            return False, reason
        except Exception as exc:
            logger.warning("is_data_query failed (%s) — allowing query through", exc)
            return True, ""

    async def get_chart_config(
        self, col_dt_list: list, sample: str, stats: str, query: str,
        dataset_context: str = "",
        query_mode: str = "",
    ) -> LLMResponseSchema:
        # stats is accepted for backward-compat but NOT sent — dataset_context
        # already covers it at a fraction of the token cost.
        mode_hint     = f"\n[MODE: {query_mode.upper()}]" if query_mode else ""
        context_block = f"\n{dataset_context}" if dataset_context else ""
        agg_hint      = _build_agg_hint(col_dt_list)
        agg_block     = f"\n{agg_hint}" if agg_hint else ""

        user_msg = (
            f"Columns: {[col for col, _ in col_dt_list]}\n\n"
            f"Sample (3 rows):\n{sample}\n"
            f"{context_block}"
            f"{agg_block}"
            f"{mode_hint}\n\n"
            f"Query:\n{query}"
        )
        last_raw   = ""
        last_error = ""

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                last_raw = await self._call([
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ])
                schema = _parse_and_validate(last_raw)
                logger.info("LLM config validated (attempt %d)", attempt)
                return schema
            except (json.JSONDecodeError, ValidationError, KeyError, IndexError) as exc:
                last_error = str(exc)
                logger.warning("LLM attempt %d failed: %s", attempt, last_error)

        columns = [col for col, _ in col_dt_list]
        for rep in range(1, self.MAX_REPAIRS + 1):
            try:
                logger.info("Attempting LLM repair (repair %d)", rep)
                repair_msg = REPAIR_PROMPT.format(
                    error=last_error, original=last_raw, columns=columns
                )
                last_raw = await self._call([
                    {"role": "system",    "content": SYSTEM_PROMPT},
                    {"role": "user",      "content": user_msg},
                    {"role": "assistant", "content": last_raw},
                    {"role": "user",      "content": repair_msg},
                ])
                schema = _parse_and_validate(last_raw)
                logger.info("LLM repair succeeded (repair %d)", rep)
                return schema
            except (json.JSONDecodeError, ValidationError, KeyError, IndexError) as exc:
                last_error = str(exc)
                logger.warning("LLM repair %d failed: %s", rep, last_error)

        logger.error("All LLM attempts failed — using deterministic fallback config")
        return self._fallback_config(col_dt_list)

    async def get_scorecard_config(
        self, col_dt_list: list, sample: str, stats: str, query: str,
        max_scorecards: int = 6,
    ) -> list:
        """
        Dedicated scorecard-config call for scorecards_only mode.
        stats accepted for backward-compat but not forwarded — dataset_context
        (embedded in stats by the caller) covers it more compactly.
        """
        columns  = [col for col, _ in col_dt_list]
        agg_hint = _build_agg_hint(col_dt_list)
        # stats may contain "\n\n[DATASET PROFILE]..." appended by the caller
        profile  = stats.split("\n\n", 1)[1] if "\n\n" in stats else ""
        prompt = (
            f"Design {max_scorecards} KPI scorecards for this dataset.\n"
            f"Columns: {columns}\n"
            f"{profile}\n"
            f"{agg_hint}\n"
            f"Query: {query}\n\n"
            f"Rules: cover different business dimensions; skip identifier columns "
            f"(ID, Key, Zip, Phone); follow aggregation hints exactly; "
            f"labels must be plain business English.\n"
            f"Return ONLY: {{\"scorecards\":[{{\"column\":\"...\",\"aggregation\":\"sum|mean|count|min|max\",\"label\":\"...\",\"subtitle\":\"...\"}}]}}"
        )
        try:
            raw = await self._call([
                {"role": "system", "content": "Return ONLY valid JSON. No markdown."},
                {"role": "user",   "content": prompt},
            ])
            clean     = re.sub(r"```(?:json)?|```", "", raw).strip()
            data      = json.loads(clean)
            scorecards = data.get("scorecards", [])
            logger.info("get_scorecard_config returned %d scorecard(s)", len(scorecards))
            return scorecards
        except Exception as exc:
            logger.warning("get_scorecard_config failed (%s) — returning empty", exc)
            return []

    async def get_table_config(
        self, col_dt_list: list, sample: str, stats: str, query: str,
        max_tables: int = 2,
    ) -> list:
        """
        Dedicated table-config call. stats accepted for backward-compat but
        only the embedded dataset profile is forwarded — not the full describe().
        """
        columns = [col for col, _ in col_dt_list]
        profile = stats.split("\n\n", 1)[1] if "\n\n" in stats else ""
        prompt = (
            f"Design 1–{max_tables} useful pivot tables for this dataset.\n"
            f"Columns: {columns}\n"
            f"{profile}\n"
            f"Query: {query}\n\n"
            f"Rules: group by a meaningful categorical column; aggregate a real numeric metric; "
            f"skip identifier/code columns (ID, Zip, Phone).\n"
            f"Return ONLY: {{\"tables\":[{{\"type\":\"pivot\",\"index\":\"cat_col\",\"columns\":null,\"values\":\"num_col\",\"aggregation\":\"sum\",\"title\":\"...\"}}]}}"
        )
        try:
            raw = await self._call([
                {"role": "system", "content": "Return ONLY valid JSON. No markdown."},
                {"role": "user",   "content": prompt},
            ])
            clean  = re.sub(r"```(?:json)?|```", "", raw).strip()
            data   = json.loads(clean)
            tables = data.get("tables", [])
            logger.info("get_table_config returned %d table(s)", len(tables))
            return tables
        except Exception as exc:
            logger.warning("get_table_config failed (%s) — returning empty", exc)
            return []

    async def _call(self, messages: list) -> str:
        response = await self._client.chat.completions.create(
            model="openrouter/owl-alpha",
            messages=messages,
        )
        return response.choices[0].message.content

    @staticmethod
    def _fallback_config(col_dt_list: list) -> LLMResponseSchema:
        cols     = {col: str(dt) for col, dt in col_dt_list}
        num_cols = [c for c, dt in cols.items() if "int" in dt or "float" in dt]
        dt_cols  = [c for c, dt in cols.items() if "datetime" in dt or "date" in dt.lower()]

        if dt_cols and num_cols:
            raw = [{
                "type": "line", "x": dt_cols[0], "y": num_cols[0],
                "aggregation": "sum", "time_granularity": "month",
                "layout_size": "large", "title": f"{num_cols[0]} over time",
            }]
        elif len(num_cols) >= 2:
            raw = [{
                "type": "scatter", "x": num_cols[0], "y": num_cols[1],
                "aggregation": "none", "time_granularity": "none",
                "layout_size": "medium", "title": f"{num_cols[0]} vs {num_cols[1]}",
            }]
        elif num_cols:
            raw = [{
                "type": "histogram", "x": num_cols[0], "y": None,
                "aggregation": "none", "time_granularity": "none",
                "layout_size": "medium", "title": f"Distribution of {num_cols[0]}",
            }]
        else:
            first = list(cols.keys())[0]
            raw = [{
                "type": "histogram", "x": first, "y": None,
                "aggregation": "none", "time_granularity": "none",
                "layout_size": "medium", "title": f"Distribution of {first}",
            }]

        return LLMResponseSchema.model_validate({"scorecards": [], "charts": raw, "tables": []})