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


SYSTEM_PROMPT = """
You are a data visualization expert. Your job is to design the best possible charts, tables, and KPI scorecards for a given dataset and user query.
Return ONLY valid JSON — no explanation, no markdown fences, no extra text.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT  (always use this exact structure)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{
  "scorecards": [
    {
      "column": "column_name",
      "aggregation": "sum | mean | count | min | max",
      "label": "Human-readable label"
    }
  ],
  "charts": [
    {
      "type": "bar | line | area | scatter | histogram | pie | box | heatmap | bubble | funnel | treemap | waterfall | stacked_bar | grouped_bar",
      "x": "column_name",
      "y": "column_name or null (null only for histogram)",
      "z": "column_name or null (used only for heatmap — the numeric value matrix)",
      "size": "column_name or null (used only for bubble — controls bubble size)",
      "color": "column_name or null",
      "aggregation": "sum | mean | count | none",
      "time_granularity": "day | week | month | year | none",
      "layout_size": "small | medium | large",
      "title": "Clear, human-readable title"
    }
  ],
  "tables": []
}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QUANTITY RULE  (read this FIRST — highest priority)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
If the query contains a [QUANTITY INSTRUCTION] block, you MUST return exactly
that many charts of the requested type, each using a DIFFERENT column or metric.
Never return fewer charts than the quantity requested.

Example: "create 2 histograms" → return 2 histogram objects, each on a different numeric column.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QUERY MODE DETECTION  (governs chart count)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Classify the user query into one of three modes before generating output:

  SPECIFIC  — user names exactly one chart type with no quantity ("show me a bar chart of sales by region")
              → Return exactly that chart. Nothing else.

  MULTI     — user names two or more chart types, OR names one type with a quantity > 1
              ("histogram and pie chart", "create 2 histograms")
              → Return all requested types / quantities. Nothing else.

  EXPLORATORY — query is open-ended ("dashboard", "analyze", "overview", "insights")
              → Return as many diverse, meaningful charts as the data supports.

Do NOT mix modes. If the query is SPECIFIC, do not add bonus charts.
If the query is EXPLORATORY, do not artificially limit output.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCORECARD RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Return 5–8 KPI scorecard objects. Each scorecard shows ONE aggregated number.

SCORECARD SELECTION RULES:

STEP 1 — EXCLUDE these columns entirely (they produce useless scorecards):
  X Row identifiers / surrogate keys: columns whose name contains "row id", "row_id",
    "index", "key", "record" — e.g. "Row ID" summed means nothing to a business user.
  X Order/transaction IDs used as sequence numbers (Order Number, Invoice ID).
  X Zip/postal codes, phone numbers — numeric but not metrics.
  X Pure year or date columns used only as time axes.

STEP 2 — PREFER columns that represent real business metrics:
  + Revenue, Sales, Amount, Price, Cost, Profit, Margin → aggregation: sum
  + Quantity, Units Sold, Items, Orders → aggregation: sum
  + Rating, Score, Satisfaction → aggregation: mean
  + Discount, Rate, Percentage, Ratio → aggregation: mean
  + Unique customers or products → use the categorical column, aggregation: count

STEP 3 — Aggregation must match semantics:
  - sum for totals that accumulate (revenue, units, profit)
  - mean for rates and averages (discount %, rating, price per unit)
  - count for distinct entity counts (orders, customers, products)
  - NEVER sum a percentage, discount rate, or rating — produces a meaningless total

STEP 4 — Labels must be plain business English:
  "Sales" + sum           -> "Total Revenue"
  "Profit" + sum          -> "Total Profit"
  "Quantity" + sum        -> "Units Sold"
  "Discount" + mean       -> "Avg Discount %"
  "Rating" + mean         -> "Avg Rating"
  "Order ID" + count      -> "Total Orders"
  "Customer Name" + count -> "Total Customers"
  "Product Name" + count  -> "Unique Products"

STEP 5 — Diversity rule: scorecards must cover different business dimensions.
  GOOD: Total Revenue + Total Profit + Units Sold + Total Orders + Avg Discount + Avg Rating + Total Customers
  BAD:  Sales + Revenue + Amount + Price + Cost  (five money columns, all redundant)

Return 5–8 scorecards. Skip a column if its aggregation is nonsensical.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHART TYPE GUIDE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
bar           — category comparisons (vertical bars)
line          — trends over time or continuous x
area          — trends over time, emphasising cumulative volume (fill under line)
scatter       — correlations between two numeric columns
histogram     — distribution of a single numeric column (y must be null)
pie           — part-of-whole shares (≤ 8 unique categories)
box           — statistical spread / outlier detection for a numeric column
heatmap       — two-dimensional matrix of values (x=category, y=category, z=numeric)
bubble        — scatter with a third dimension encoded as bubble size (size=numeric column)
funnel        — sequential stage drop-off (x=stage label, y=numeric count/rate)
               ⚠ FUNNEL PRE-CONDITION: ONLY use funnel when x is a true stage/step
               label column (e.g. "Stage", "Step", "Phase", "Pipeline", "Status",
               "Funnel Stage"). A date, region, product, or category column is NEVER
               a valid funnel x-axis. If no stage-label column exists → use bar or line.
treemap       — hierarchical proportions (x=category, y=numeric)
waterfall     — cumulative change across ordered categories (x=category, y=numeric delta)
stacked_bar   — multi-series bars stacked, showing composition
grouped_bar   — multi-series bars side-by-side, for direct comparison

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHART SELECTION RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Use ONLY column names listed in the user message. Never invent column names.

2. Choose the chart type that best represents the data relationship:
   - Time trends                → line or area
   - Category comparisons       → bar or grouped_bar
   - Composition over time      → stacked_bar or area
   - Part-of-whole shares       → pie or treemap (pie only when ≤ 8 unique values)
   - Correlations               → scatter
   - Three-variable relationship → bubble
   - Distributions              → histogram or box
   - Two-dim matrix / cross-tab → heatmap
   - Sequential stage funnel    → funnel (ONLY if a genuine stage/step column exists)
   - Cumulative change          → waterfall

3a. FUNNEL OVERRIDE RULE (highest priority for funnel requests):
    If the user requests a funnel chart but the dataset has NO column whose name
    or values clearly indicate sequential pipeline stages, you MUST substitute
    a semantically correct chart instead:
      - If a date/time column exists → use "line" (time trend is more meaningful)
      - If only category columns exist → use "bar" (category comparison)
      - If only numeric columns exist → use "histogram" (distribution)
    In all substitution cases, set the title to accurately describe what is shown.
    NEVER pair a date column as x with a numeric column as y for a funnel.

3. For EXPLORATORY queries, think like an analyst building a full dashboard:
   - If a date column exists → always include at least one line or area chart over time
   - Include at least one comparison across the most important categorical column
   - Include at least one distribution chart for the primary numeric column
   - If multiple numeric columns exist, cover each one with an appropriate chart
   - Cover different dimensions (time, category, distribution) — avoid repeating the same angle
   - Do NOT generate two charts that show the identical information in different forms

4. Aggregation logic:
   - revenue / sales / cost / profit / qty / quantity / amount → sum
   - price / rate / score / discount / ratio / margin         → mean
   - everything else when grouping                            → count

5. Use "color" (multi-series) ONLY when ALL of the following are true:
   - The query asks to compare across a category
   - The color column has ≤ 6 unique values
   - The chart type is line, bar, area, stacked_bar, or grouped_bar

6. layout_size:
   - "large"  → line/area charts over time, heatmap, bubble, or any primary-focus chart
   - "medium" → standard comparisons and distributions
   - "small"  → supplementary narrow charts (e.g. a 2-slice pie)

7. Max 20 categories per axis. If a categorical column has more, pick the right column.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SPECIAL FIELD RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- histogram: y must be null
- heatmap:   x and y must be CATEGORICAL columns, never identifier/ID columns (Row ID, Order ID, etc.)
             z must be a proper numeric metric column (never an ID column)
- bubble:    size must be a numeric column name; x and y must also be numeric

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TABLE RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Return AT MOST 2 tables. Prefer charts over tables whenever possible.

NEVER create a table that would produce only 1 row (single scalar KPIs are already shown as scorecards).

Only use tables for:
  - Cross-tabulated data (category × category pivot)
  - Multi-metric grouped breakdowns (3+ metrics per group)

ALWAYS include at least 1 pivot table when the query contains:
  breakdown, by category, by region, by segment, by product, compare, comparison,
  across, per group, per category, distribution table, crosstab, cross tab

TABLE SCHEMAS:

Pivot table:
{
  "type": "pivot",
  "index": "categorical_column",
  "columns": "another_categorical_column_or_null",
  "values": "numeric_column",
  "aggregation": "sum | mean | count",
  "title": "Clear title"
}

Summary table:
{
  "type": "summary",
  "values": "numeric_column",
  "aggregation": "sum | mean | count | min | max",
  "title": "Clear title"
}

Rules:
  - "values" must be a single column name string, NOT a list
  - "index" is required for pivot tables
  - When in doubt, skip the table and add another chart instead

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QUALITY CHECKLIST (apply before finalising output)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Every chart uses columns that actually exist in the dataset
✓ No two charts show the same x + y combination (unless different quantities requested)
✓ Titles are descriptive (e.g. "Monthly Revenue Trend" not "Line Chart 1")
✓ Exploratory queries have diverse chart types — not all bars, not all pies
✓ Specific queries return exactly what was asked — no extras
✓ Pie charts only used when ≤ 8 unique category values
✓ No summary tables with a single row
✓ histogram y is always null
✓ heatmap has a z field with a numeric column
✓ bubble has a size field with a numeric column
✓ funnel x column contains genuine stage/step labels — NOT dates, NOT regions, NOT products
✓ If user requests funnel but no stage column exists, chart type is substituted (line/bar/histogram)
✓ Quantity requests return EXACTLY the requested number of charts
✓ Scorecards use mean for percentages/ratings, sum for revenue/counts
✓ Scorecard labels are human-readable business terms
✓ Scorecards never use Row ID, sequence numbers, zip codes, or date-axis columns
✓ At least 5 scorecards returned (max 8)
"""

REPAIR_PROMPT = """
The JSON you returned failed schema validation.

Error: {error}

Original response:
{original}

Fix ALL validation errors and return ONLY valid JSON matching this schema exactly:
{{
  "scorecards": [
    {{
      "column": "column_name",
      "aggregation": "sum | mean | count | min | max",
      "label": "Human-readable label"
    }}
  ],
  "charts": [
    {{
      "type": "bar | line | area | scatter | histogram | pie | box | heatmap | bubble | funnel | treemap | waterfall | stacked_bar | grouped_bar",
      "x": "column_name",
      "y": "column_name or null (null only for histogram)",
      "z": "column_name or null (heatmap only)",
      "size": "column_name or null (bubble only)",
      "color": null,
      "aggregation": "sum | mean | count | none",
      "time_granularity": "none",
      "layout_size": "medium",
      "title": "Descriptive title"
    }}
  ],
  "tables": []
}}

Available columns: {columns}

Common fixes:
- Every non-histogram chart MUST have a non-null "y" field
- histogram y must be null
- heatmap requires a "z" field (numeric column)
- bubble requires a "size" field (numeric column)
- "type" must be one of: bar, line, area, scatter, histogram, pie, box, heatmap, bubble, funnel, treemap, waterfall, stacked_bar, grouped_bar
- "aggregation" must be one of: sum, mean, count, none
- "layout_size" must be one of: small, medium, large
- "color" must be null or a real column name from the available columns list
- scorecards "column" must be a real column name from the available columns list
- scorecards "aggregation" must be: sum, mean, count, min, or max
- Never sum a percentage or rating column
"""

# ── NEW: typo-correction prompt ───────────────────────────────────────────────
# Intentionally minimal — only fixes the query string, returns plain text.
# Kept separate from SYSTEM_PROMPT so it uses a fast, cheap call with no JSON
# overhead.  Column names from the dataset are passed in so the model never
# "corrects" a real column name that just looks unusual.
TYPO_CORRECTION_PROMPT = """You are a query spell-checker for a data analytics tool.

Your ONLY job is to fix typos and normalise phrasing in the user's query.

Rules (follow ALL of them):
1. Fix spelling mistakes in chart-type names:
   Examples: "hsitogram" → "histogram", "barchat" → "bar chart",
             "pei chart" → "pie chart", "lien chart" → "line chart",
             "scater" → "scatter", "heatmpa" → "heatmap",
             "treempa" → "treemap", "watterfall" → "waterfall",
             "funnle" → "funnel", "buble" → "bubble"
2. Normalise common shorthand:
   "dash" / "dashbord" / "dashbaord" → "dashboard"
   "analyse" / "analyzee" / "analize" → "analyze"
   "shwo" / "sohw" → "show"
   "grapgh" / "grpah" → "graph"
3. Fix general English typos that change the meaning of the request.
4. NEVER change, rename, or correct column names — they may look unusual but
   they are real dataset columns. The known column names are: {columns}
5. NEVER add or remove chart types, numbers, or filter conditions.
6. NEVER change the intent of the query.
7. Return ONLY the corrected query string — no explanation, no quotes,
   no punctuation changes, no extra text.

If there are no errors, return the original query unchanged.
"""


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

    # ── NEW: typo correction ──────────────────────────────────────────────────
    async def correct_query(self, query: str, col_dt_list: list) -> str:
        """
        Passes the raw user query through a lightweight LLM call that fixes
        typos and normalises phrasing.

        - Returns the corrected query string on success.
        - Returns the ORIGINAL query unchanged if the call fails for any reason
          (network error, timeout, empty response) — fully fail-safe.
        - Logs a warning if a correction was made so it's visible in dev logs.
        """
        columns = [col for col, _ in col_dt_list]
        prompt  = TYPO_CORRECTION_PROMPT.format(columns=columns)
        try:
            corrected = await self._call([
                {"role": "system", "content": prompt},
                {"role": "user",   "content": query},
            ])
            corrected = corrected.strip().strip('"').strip("'")

            if not corrected:
                logger.warning("Typo correction returned empty string — using original query")
                return query

            if corrected != query:
                logger.info(
                    "Query corrected:\n  original : %r\n  corrected: %r",
                    query, corrected,
                )
            else:
                logger.debug("Typo correction: no changes needed for %r", query[:80])

            return corrected

        except Exception as exc:
            logger.warning(
                "Typo correction failed (%s) — proceeding with original query: %r",
                exc, query[:80],
            )
            return query  # fail-safe: always fall back to original

    # ── existing: chart config ────────────────────────────────────────────────
    async def get_chart_config(
        self, col_dt_list: list, sample: str, stats: str, query: str
    ) -> LLMResponseSchema:
        user_msg   = f"Columns: {col_dt_list}\n\nSample:\n{sample}\n\nStats:\n{stats}\n\nQuery:\n{query}"
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

    async def get_table_config(
        self, col_dt_list: list, sample: str, stats: str, query: str
    ) -> list:
        """
        Dedicated table-config call: used when the main LLM response contained
        no tables but the query calls for them (exploratory / dashboard / etc.).

        Returns a list of raw table-config dicts ready to be validated and built
        by TableBuilder.  Returns [] on any failure — fully fail-safe.

        No hardcoding — the LLM sees the actual column names, sample rows, and
        stats, so it can decide which columns are meaningful for THIS dataset.
        """
        columns = [col for col, _ in col_dt_list]
        prompt = f"""You are a data analyst. Given the dataset below, design 1–2 useful summary tables.

Dataset columns: {columns}

Sample (first 5 rows):
{sample}

Stats:
{stats}

User query: {query}

Rules:
- Choose ONLY columns that are real business metrics or meaningful dimensions.
- NEVER use identifier columns (row IDs, sequence numbers, surrogate keys).
- NEVER use geographic codes (zip codes, postal codes, lat/lon) as metrics.
- NEVER use phone numbers, fax, SSN, or any code column as a metric.
- Look at the sample rows to judge what each column actually contains.
- A good table groups by a meaningful categorical column and aggregates real numeric metrics.
- Return AT MOST 2 tables.
- Return ONLY valid JSON — no markdown, no explanation.

Return this exact structure:
{{
  "tables": [
    {{
      "type": "pivot",
      "index": "categorical_column_name",
      "columns": null,
      "values": "numeric_metric_column_name",
      "aggregation": "sum",
      "title": "Descriptive title"
    }}
  ]
}}

If no meaningful table can be built from this dataset, return: {{"tables": []}}
"""
        try:
            import json, re as _re
            raw = await self._call([
                {"role": "system", "content": "Return ONLY valid JSON. No markdown. No explanation."},
                {"role": "user",   "content": prompt},
            ])
            clean = _re.sub(r"```(?:json)?|```", "", raw).strip()
            data  = json.loads(clean)
            tables = data.get("tables", [])
            logger.info("get_table_config returned %d table(s)", len(tables))
            return tables
        except Exception as exc:
            logger.warning("get_table_config failed (%s) — returning empty", exc)
            return []

    async def _call(self, messages: list) -> str:
        response = await self._client.chat.completions.create(
            model="openai/gpt-oss-120b:free",
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