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


# ─────────────────────────────────────────────
# 2. LLM CLIENT  (prompt + parse + repair + fallback)
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """
You are a data visualization expert.
Return ONLY valid JSON — no explanation, no markdown fences.

OUTPUT FORMAT:
{
  "charts": [
    {
      "type": "bar | line | scatter | histogram | pie",
      "x": "column_name",
      "y": "column_name or null",
      "color": "column_name or null",
      "aggregation": "sum | mean | count | none",
      "time_granularity": "day | week | month | year | none",
      "layout_size": "small | medium | large",
      "title": "clear title"
    }
  ],
  "tables": []
}

CHART RULES:
1. Use ONLY columns listed in the user message.
2. Chart type priority (high → low): line > bar > pie > scatter > histogram.
   - trend over time     → line  (ALWAYS use when a date column exists)
   - category comparison → bar
   - part-of-whole       → pie   (only when ≤ 6 unique values)
   - relationship        → scatter
   - distribution        → histogram (last resort)
3. use chart type priority only when query is related to dashboard/overview/analysis AND user request is broad/ambiguous.
3. Date column present → ALWAYS produce at least one line chart aggregated by month or year.
4. Aggregation: sales/qty/revenue/cost → sum; price/rate/score → mean; else → count.
5. Use "color" ONLY when: query compares across categories AND the
   column has <= 6 unique values AND chart type is line or bar.
6. Max 20 categories per axis.
7. For dashboard/overview queries return diverse charts ONLY if the user request is broad.
   Do NOT apply this rule when the user explicitly requests a specific chart type.
   (e.g. one line + one bar, not two bars on the same column).

IMPORTANT:
- If the user explicitly asks for a specific chart type (e.g., "line chart", "bar chart"),
  return ONLY that chart and nothing else.
- Do NOT generate extra charts in that case.

STRICT OUTPUT CONTROL:
- If query is specific → 1 chart
- If query is exploratory → multiple charts
- Never mix both behaviors

If the user explicitly mentions multiple chart types (e.g. "histogram and pie chart"),
you MUST return ALL requested chart types.
Do NOT return only one chart in that case.

TABLE RULES — READ CAREFULLY:

HARD LIMIT: Return AT MOST 2 tables total. Never exceed this.

NEVER generate a summary table that produces only 1 row.
Single-value KPIs (total revenue, avg price, count) are already shown as
scorecards — DO NOT duplicate them as tables. Tables must have multiple rows.

ONLY generate a table when it shows grouped/cross-tabulated data that a
chart cannot easily show. Good table candidates:
  - pivot of category × category (e.g. gender × product)
  - grouped breakdown with 3+ metrics per group

ALWAYS prefer adding another chart over adding a table.
If you are unsure whether to add a table, add a chart instead.

ALWAYS generate at least 1 pivot table when the query contains any of:
  breakdown, by category, by region, by segment, by product,
  compare, comparison, across, per, group, distribution table

TABLE SCHEMAS:

1. Pivot table — use to cross-tabulate two categorical dimensions:
{
  "type": "pivot",
  "index": "categorical_column",
  "columns": "another_categorical_column_or_null",
  "values": "numeric_column",
  "aggregation": "sum | mean | count",
  "title": "clear title"
}

2. Summary table — ONLY use when grouped (multiple rows).
   NEVER use for a single scalar value.
{
  "type": "summary",
  "values": "numeric_column",
  "aggregation": "sum | mean | count | min | max",
  "title": "clear title"
}

RULES:
- "values" must be a single numeric column name (a string, NOT a list).
- "index" is required for pivot.
- MAX 2 tables. Prefer charts over tables when in doubt.
- For dashboard/analyze/overview queries: return at least 2-3 charts + 0-2 tables.
"""

REPAIR_PROMPT = """
The JSON you returned failed schema validation.

Error: {error}

Original response:
{original}

Fix ALL validation errors and return ONLY valid JSON matching this schema:
{{
  "charts": [
    {{
      "type": "bar | line | scatter | histogram | pie",
      "x": "column_name",
      "y": "column_name or null for histogram",
      "color": null,
      "aggregation": "sum | mean | count | none",
      "time_granularity": "none",
      "layout_size": "medium",
      "title": "title"
    }}
  ],
  "tables": []
}}

Available columns: {columns}
"""


def _strip_fences(text: str) -> str:
    return re.sub(r"```(?:json)?", "", text).replace("```", "").strip()


def _parse_and_validate(raw: str) -> LLMResponseSchema:
    data = json.loads(_strip_fences(raw))
    return LLMResponseSchema.model_validate(data)


class LLMClient:
    MAX_RETRIES = 2   # normal parse retries
    MAX_REPAIRS = 1   # repair-loop attempts after retries exhausted

    def __init__(self):
        self._client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )

    # ── public ───────────────────────────────────────────────────────
    async def get_chart_config(
        self, col_dt_list: list, sample: str, stats: str, query: str
    ) -> LLMResponseSchema:
        """
        Returns a validated LLMResponseSchema.
        Attempt order:
          1. MAX_RETRIES normal LLM calls
          2. MAX_REPAIRS repair calls (error + original sent back to model)
          3. Deterministic fallback built from column metadata — never crashes
        """
        user_msg   = f"Columns: {col_dt_list}\n\nSample:\n{sample}\n\nStats:\n{stats}\n\nQuery:\n{query}"
        last_raw   = ""
        last_error = ""

        # --- normal retries ---
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

        # --- repair loop ---
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

        # --- deterministic fallback ---
        logger.error("All LLM attempts failed — using deterministic fallback config")
        return self._fallback_config(col_dt_list)

    # ── private ──────────────────────────────────────────────────────
    async def _call(self, messages: list) -> str:
        response = await self._client.chat.completions.create(
            model="z-ai/glm-4.5-air:free",
            messages=messages,
        )
        return response.choices[0].message.content

    @staticmethod
    def _fallback_config(col_dt_list: list) -> LLMResponseSchema:
        """
        Build a safe minimal config from column dtypes alone.
        Priority: datetime+numeric -> line | two numerics -> scatter | one numeric -> histogram
        """
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

        return LLMResponseSchema.model_validate({"charts": raw, "tables": []})