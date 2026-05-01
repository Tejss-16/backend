# app/services/chat_service.py 
from app.pipeline.llm_client import LLMClient
import logging
import asyncio
import json
import re
import textwrap
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# UNIVERSAL CLEANER  (unchanged)
# ─────────────────────────────────────────────

def clean_value(val):
    if pd.isna(val):
        return None
    if isinstance(val, pd.Period):
        return str(val)
    if isinstance(val, pd.Timestamp):
        return val.strftime("%Y-%m")
    if isinstance(val, pd.Timedelta):
        return str(val)
    if hasattr(val, "item") and not isinstance(val, (pd.Series, pd.DataFrame)):
        return val.item()
    return val


def clean_records(df, n=3):
    return [
        {k: clean_value(v) for k, v in row.items()}
        for row in df.head(n).to_dict(orient="records")
    ]


# ─────────────────────────────────────────────
# POST-EXEC RESULT NORMALISER  (unchanged)
# ─────────────────────────────────────────────

def _normalise_result(result):
    if not isinstance(result, pd.DataFrame):
        return result
    idx = result.index
    if isinstance(idx, pd.RangeIndex) and idx.name is None:
        return result
    try:
        return result.reset_index()
    except Exception:
        return result


def _is_correlation_matrix(result) -> bool:
    if not isinstance(result, pd.DataFrame):
        return False
    cols = list(result.columns)
    if result.shape[0] != result.shape[1]:
        return False
    try:
        index_vals = list(result.index.astype(str))
        col_vals = [str(c) for c in cols]
        return index_vals == col_vals
    except Exception:
        return False


def _is_metric_label_table(result) -> bool:
    if not isinstance(result, pd.DataFrame):
        return False
    if result.shape[1] != 2:
        return False
    label_col = result.iloc[:, 0]
    return label_col.dtype == object and result.shape[0] <= 20


def _extract_top_correlations(corr_df: pd.DataFrame) -> pd.DataFrame:
    _JUNK_COLS = {"row id", "row_id", "postal code", "postal_code", "zip", "phone"}
    records = []
    cols = list(corr_df.columns)
    for i, c1 in enumerate(cols):
        if c1.lower() in _JUNK_COLS:
            continue
        for j, c2 in enumerate(cols):
            if j <= i:
                continue
            if c2.lower() in _JUNK_COLS:
                continue
            val = corr_df.iloc[i, j]
            if pd.isna(val) or abs(val) == 1.0:
                continue
            records.append({"Feature Pair": f"{c1} vs {c2}", "Correlation": round(val, 4)})
    if not records:
        return corr_df
    df = pd.DataFrame(records).sort_values("Correlation", key=abs, ascending=False)
    return df.head(10).reset_index(drop=True)


# ─────────────────────────────────────────────
# SAFE EXEC GLOBALS  (unchanged)
# ─────────────────────────────────────────────

_SAFE_GLOBALS = {
    "__builtins__": {},
    "pd": pd,
    "np": __import__("numpy"),
}


# ─────────────────────────────────────────────
# CODE NORMALISER  (unchanged)
# ─────────────────────────────────────────────

def _normalise_code(code: str) -> str:
    code = code.replace(chr(0), "")
    code = re.sub(r"^\s*(?:import\s+\S+.*|from\s+\S+\s+import\s+.*)$",
                  "", code, flags=re.MULTILINE)
    return textwrap.dedent(code).strip()


# ─────────────────────────────────────────────
# PANDAS COMPATIBILITY RULES  (unchanged)
# ─────────────────────────────────────────────

_PANDAS_COMPAT_RULES = """\
PANDAS 2.0 COMPATIBILITY — follow these rules exactly:

value_counts():
  CORRECT:
      vc = df['col'].value_counts().reset_index()
      vc.columns = ['ColName', 'Count']
  WRONG (crashes in pandas 2.0):
      vc = vc.rename(columns={'index': 'X'})

set_index():
  NEVER call .set_index('X') unless 'X' is currently a column of the DataFrame.

pd.concat():
  When concatenating DataFrames that already have named indexes, reset indexes first:
      result = pd.concat([df1.reset_index(), df2.reset_index()], ignore_index=True)

corr():
  ALWAYS use: df.corr(numeric_only=True)
  NEVER use:  df.corr()

FINAL RESULT:
  `result` must ALWAYS be a flat DataFrame with regular RangeIndex."""

_VALUE_COUNTS_RULE = """\
CRITICAL — pandas 2.0+ value_counts() rule:
  CORRECT (pandas 2.0+):
      vc = df['col'].value_counts().reset_index()
      vc.columns = ['Label', 'Count']
  NEVER use rename({'index': ...}) — 'index' is not a column in pandas 2.0.
  ALWAYS use df.corr(numeric_only=True) — never bare df.corr()."""


# ─────────────────────────────────────────────
# SAFE EXEC  (unchanged)
# ─────────────────────────────────────────────

async def safe_exec(code: str, df: pd.DataFrame, llm) -> tuple:
    async def _run_with_timeout(c, d, timeout=10.0):
        try:
            return await asyncio.wait_for(asyncio.to_thread(_run_code, c, d), timeout=timeout)
        except asyncio.TimeoutError:
            logger.error("Execution timed out after %s seconds.", timeout)
            return None, "Execution timed out."

    code = _normalise_code(code)
    result, exec_error = await _run_with_timeout(code, df)
    result = _normalise_result(result)

    if exec_error and "timed out" not in exec_error:
        logger.warning("First exec failed (%s) — asking LLM to fix", exec_error)
        fixed_code = await _fix_code(code, exec_error, llm)
        if fixed_code:
            result, exec_error = await _run_with_timeout(fixed_code, df)
            result = _normalise_result(result)

    if _is_correlation_matrix(result):
        result = _extract_top_correlations(result)

    return result, exec_error


def _run_code(code: str, df: pd.DataFrame):
    try:
        local_vars = {"df": df.copy()}
        exec(code, _SAFE_GLOBALS, local_vars)
        return local_vars.get("result"), None
    except Exception as e:
        logger.error("Exec error: %s\nCode:\n%s", e, code)
        return None, str(e)


async def _fix_code(bad_code: str, error: str, llm) -> str:
    retry_prompt = f"""Fix this Python pandas code.

Error:
{error}

Code:
{bad_code}

{_VALUE_COUNTS_RULE}

Additional rules:
- Use `df` as the dataframe variable.
- Store result in `result` as a flat DataFrame.
- Do NOT use import statements — pd and np are already available.
- Return ONLY valid JSON: {{"code": "corrected code here"}}
"""
    try:
        resp = await llm._call([
            {"role": "system", "content": "Return valid JSON only. No markdown."},
            {"role": "user", "content": retry_prompt},
        ])
        clean = re.sub(r"```(?:json)?|```", "", resp).strip()
        fixed = json.loads(clean).get("code", "").strip()
        return _normalise_code(fixed) if fixed else ""
    except Exception as e:
        logger.error("Fix attempt failed: %s", e)
        return ""


# ─────────────────────────────────────────────
# LAYER 1 — Result Interpreter  (unchanged)
# ─────────────────────────────────────────────

def interpret_result(result) -> dict:
    if result is None:
        return {"type": "empty"}

    if isinstance(result, bool):
        return {"type": "boolean", "value": bool(result)}
    if hasattr(result, "item") and not isinstance(result, (pd.Series, pd.DataFrame)):
        raw = result.item()
        if isinstance(raw, bool):
            return {"type": "boolean", "value": raw}
        return {"type": "scalar", "value": float(raw)}

    if isinstance(result, (int, float)):
        return {"type": "scalar", "value": float(result)}

    if isinstance(result, pd.Series):
        return {
            "type": "list",
            "count": len(result),
            "sample": [clean_value(x) for x in result.head(5).tolist()],
            "unique": int(result.nunique()),
        }

    if isinstance(result, pd.DataFrame):
        rows, cols = result.shape
        return {
            "type": "table",
            "rows": rows,
            "cols": cols,
            "columns": list(result.columns),
            "sample": clean_records(result),
        }

    return {"type": "unknown", "raw": str(result)}


# ─────────────────────────────────────────────
# LAYER 2 — Table Decision
# [CHANGE 5] LLM-driven, falls back to rule-based
# ─────────────────────────────────────────────

def _rule_based_should_show_table(meta: dict) -> bool:
    """Original rule-based fallback."""
    t = meta["type"]
    if t in ["scalar", "boolean", "empty", "unknown"]:
        return False
    if t == "list":
        return meta["count"] > 1
    if t == "table":
        if meta["rows"] == 1 and meta["cols"] == 1:
            return False
        return meta["rows"] > 0
    return False


async def _llm_table_decision(query: str, meta: dict, columns: list, llm) -> bool:
    """
    Ask the LLM whether showing a table alongside the answer adds value.
    Returns True (show table) or False (skip table).
    Falls back to rule-based on any failure.
    """
    prompt = f"""You are a data analytics assistant deciding whether to show a data table alongside a text answer.

User query: {query}

Result metadata:
{json.dumps(meta, indent=2)}

Dataset columns: {columns}

Question: Does showing a table (listing the rows/values) add meaningful value to the answer for this query?

Rules:
- Scalars (single number) almost never need a table unless the user explicitly asked for one.
- Boolean results never need a table.
- Lists or DataFrames with >1 row usually benefit from a table when they contain ranked or multi-column data.
- If the result is a single-column list of >5 items it can be worth showing as a table.
- If the user asked "list", "show", "give me all", "top N", a table is almost always helpful.

Return ONLY valid JSON: {{"show_table": true}} or {{"show_table": false}}
No explanation. No markdown.
"""
    try:
        raw = await llm._call([
            {"role": "system", "content": "Return only JSON. No markdown. No explanation."},
            {"role": "user", "content": prompt},
        ])
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        return bool(json.loads(clean).get("show_table", False))
    except Exception as e:
        logger.warning("LLM table decision failed (%s) — using rule-based fallback", e)
        return _rule_based_should_show_table(meta)


def _format_cell(val):
    if isinstance(val, float):
        if abs(val) >= 1000:
            return round(val, 2)
        if abs(val) < 0.01 and val != 0:
            return round(val, 6)
        return round(val, 4)
    return val


def to_table_records(result, meta: dict):
    if isinstance(result, pd.DataFrame):
        df = result.head(50).copy()
        date_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
        time_keywords = ["date", "month", "year", "time", "period"]
        for col in df.columns:
            if any(k in str(col).lower() for k in time_keywords):
                try:
                    parsed = pd.to_datetime(df[col], errors="coerce")
                    if parsed.notna().mean() > 0.7:
                        df[col] = parsed
                        date_cols = [col]
                        break
                except Exception:
                    pass
        if date_cols:
            df = df.sort_values(date_cols[0])
        else:
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            if numeric_cols:
                df = df.sort_values(numeric_cols[0], ascending=False)
        return [{k: _format_cell(clean_value(v)) for k, v in row.items()} for row in df.to_dict(orient="records")]

    if isinstance(result, pd.Series):
        df = result.head(50).reset_index()
        return [{k: _format_cell(clean_value(v)) for k, v in row.items()} for row in df.to_dict(orient="records")]

    return None


# ─────────────────────────────────────────────
# LAYER 0 — Off-Topic Guard  (NEW)
# Fires BEFORE any other step. Two-layer approach:
#   Layer A — Fast keyword pre-filter (no LLM call, sub-millisecond)
#             Catches obviously off-topic requests and obvious data queries
#             to avoid wasting an LLM call on clear-cut cases.
#   Layer B — LLM judgment for ambiguous queries
#             Only called when Layer A is uncertain.
# Returns (is_off_topic: bool, rejection_message: str | None)
# ─────────────────────────────────────────────

# Patterns that are DEFINITELY off-topic regardless of dataset columns.
# These are intent signals — the user clearly wants something other than
# data analysis. Matched as substrings on the lowercased query.
_OFFTOPIC_SIGNALS = {
    # generative / creative requests
    "create image", "generate image", "draw", "make a picture", "make an image",
    "paint", "sketch", "render", "illustrate",
    # identity / personality
    "who are you", "what are you", "are you an ai", "are you human",
    "what is your name", "your name", "what can you do",
    # general knowledge / web
    "what is the capital", "who is the president", "who won", "what is the weather",
    "tell me a joke", "write a poem", "write a story", "write a song",
    "translate", "in french", "in spanish", "in hindi", "in german",
    "recipe for", "how to cook", "how to make", "ingredients",
    "stock price", "crypto", "bitcoin", "news today", "current news",
    # coding unrelated to the dataset
    "write a python script", "write code for", "build an app", "create a website",
    "fix my code", "debug this", "write a function",
    # media / pop culture
    "iron man", "spider man", "avengers", "movie", "song lyrics",
    "recommend a book", "recommend a movie", "recommend a show",
}

# Patterns that are DEFINITELY data-related — skip the LLM guard entirely.
# These are strong positive signals that the user is asking about their dataset.
_DATA_SIGNALS = {
    "total", "average", "mean", "sum", "count", "max", "min",
    "trend", "compare", "show me", "what is the", "how many", "how much",
    "top", "bottom", "highest", "lowest", "rank", "breakdown",
    "by category", "by region", "by month", "by year", "by segment",
    "distribution", "correlation", "growth", "decline", "revenue",
    "sales", "profit", "orders", "customers", "products",
    "filter", "where", "group by", "pivot", "chart", "graph", "table",
    "increase", "decrease", "over time", "per", "across",
}


def _keyword_offtopic_precheck(query: str) -> str | None:
    """
    Layer A: Pure string matching. No LLM call.

    Returns:
      "offtopic"  — definitely off-topic, skip LLM guard
      "data"      — definitely data-related, skip LLM guard
      "uncertain" — send to LLM for judgment
    """
    q = query.lower().strip()

    # Check off-topic signals first (higher priority)
    for signal in _OFFTOPIC_SIGNALS:
        if signal in q:
            return "offtopic"

    # Check data signals
    for signal in _DATA_SIGNALS:
        if signal in q:
            return "data"

    # Short queries with no data signal are more likely off-topic
    # but we can't be sure — send to LLM
    return "uncertain"


async def _llm_offtopic_check(query: str, columns: list, llm) -> tuple[bool, str | None]:
    """
    Layer B: LLM judgment for queries that passed keyword pre-check as "uncertain".

    Returns (is_off_topic, rejection_message).
    If the LLM call fails, defaults to NOT rejecting (fail-open) so legitimate
    queries are never wrongly blocked.
    """
    col_sample = columns[:15]  # don't bloat the prompt with huge column lists

    prompt = f"""You are a query validator for a data analytics chatbot.

The user has uploaded a dataset and is asking a question. Your job is to decide
whether the query is asking about the dataset or is completely off-topic.

Dataset columns: {col_sample}

User query: "{query}"

OFF-TOPIC examples (reject these):
- "create an image of Iron Man"
- "write me a poem"
- "who is the Prime Minister of India"
- "what's the weather today"
- "tell me a joke"
- "write Python code to scrape a website"
- "translate this to French"
- Any creative, general knowledge, or coding request unrelated to the dataset.

ON-TOPIC examples (allow these):
- "what is the total revenue?"
- "show sales by region"
- "which product has the highest profit margin?"
- "how has customer count changed over the year?"
- "is there a correlation between discount and sales?"
- Any question that can be answered by querying or analyzing the uploaded dataset,
  even if phrased casually or without using column names.

IMPORTANT: Be lenient. If there is any reasonable way the query could be about
the dataset, mark it as on-topic. Only reject when the query is clearly and
completely unrelated to data analysis.

Return ONLY valid JSON:
{{"off_topic": false}}
OR
{{"off_topic": true, "reason": "one short sentence explaining why"}}
No markdown. No explanation outside JSON.
"""
    try:
        raw = await llm._call([
            {"role": "system", "content": "Return only JSON. No markdown. No explanation."},
            {"role": "user", "content": prompt},
        ])
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        data = json.loads(clean)
        if data.get("off_topic"):
            reason = data.get("reason", "This query is not related to your dataset.")
            return True, reason
        return False, None
    except Exception as e:
        logger.warning("Off-topic LLM check failed (%s) — defaulting to allow", e)
        return False, None  # fail-open: never wrongly block a legitimate query


def _format_rejection(reason: str) -> dict:
    """Format a consistent rejection response."""
    return {
        "answer": (
            f"I can only answer questions about your uploaded dataset. {reason} "
            "Try asking something about your data!"
        ),
        "table": None,
        "off_topic": True,
    }


async def guard_offtopic(query: str, columns: list, llm) -> tuple[bool, dict | None]:
    """
    Entry point for the off-topic guard. Call this as the very first step.

    Returns:
      (False, None)         — query is on-topic, proceed normally
      (True, error_response) — query is off-topic, return error_response immediately
    """
    precheck = _keyword_offtopic_precheck(query)

    if precheck == "offtopic":
        logger.info("Off-topic guard: keyword pre-filter rejected query %r", query[:60])
        return True, _format_rejection("Your query doesn't appear to be about data analysis.")

    if precheck == "data":
        logger.debug("Off-topic guard: keyword pre-filter approved query %r", query[:60])
        return False, None

    # "uncertain" — escalate to LLM
    logger.debug("Off-topic guard: uncertain query, escalating to LLM: %r", query[:60])
    is_off_topic, reason = await _llm_offtopic_check(query, columns, llm)

    if is_off_topic:
        logger.info("Off-topic guard: LLM rejected query %r — %s", query[:60], reason)
        return True, _format_rejection(reason)

    return False, None


# ─────────────────────────────────────────────
# LAYER 3 — Query Classifier
# [CHANGE 1] LLM-driven with keyword fallback
# ─────────────────────────────────────────────

EXPLANATION_TRIGGERS = ["why", "reason", "cause", "explain", "how come", "what caused"]
COMPARISON_TRIGGERS  = ["compare", "vs", "versus", "difference between", "which is better"]
TREND_TRIGGERS       = [
    "increase", "decrease", "increasing", "decreasing",
    "trend", "growth", "decline", "up", "down",
]
SUMMARY_TRIGGERS = [
    "give me insights", "give insights", "analyse", "analyze", "analysis",
    "summarize", "summarise", "summary", "overview", "give information",
    "tell me about", "what can you tell", "give me information",
    "deep dive", "deep analysis", "full analysis", "complete analysis",
    "breakdown", "performance", "executive summary", "report",
    "insights", "key findings", "key metrics", "key insights",
    "what does the data show", "what does the data say", "explore",
    "overall", "holistic", "comprehensive", "give me a report",
]


def _keyword_classify_query(query: str) -> str:
    """Original keyword-based fallback."""
    q = query.lower()
    if any(t in q for t in SUMMARY_TRIGGERS):
        return "summary"
    if any(t in q for t in EXPLANATION_TRIGGERS):
        return "explanatory"
    if any(t in q for t in COMPARISON_TRIGGERS):
        return "comparison"
    if any(t in q for t in TREND_TRIGGERS):
        return "trend"
    return "lookup"


async def _llm_classify_query(query: str, columns: list, llm) -> str:
    """
    Use the LLM to classify the query intent.
    Returns one of: explanatory | comparison | trend | lookup | summary
    Falls back to keyword matching on failure.
    """
    prompt = f"""Classify the user's data analytics query into exactly one intent type.

User query: {query}
Dataset columns: {columns}

Intent types:
- "summary"     — user wants a broad overview, insights, analysis, report, or exploration of the whole dataset
- "explanatory" — user asks WHY something happened, wants causes/drivers/reasons
- "comparison"  — user wants to compare two or more groups, segments, or time periods
- "trend"       — user wants to see how a metric changed over time (up/down/flat)
- "lookup"      — user wants to find a specific value, list, ranking, or aggregate

Rules:
- Pick the single BEST intent. Do not mix.
- "summary" applies to: "give me insights", "analyse this", "give information", "summarize", "overview", "tell me about the data", "what does the data show", "explore the data", "give me a report", "key metrics", "executive summary".
- "trend" requires a time dimension; without dates, re-classify as "lookup".
- "explanatory" queries often contain: why, reason, cause, explain, what drove, contributing.
- Phrasing like "has sales gone up" is a trend, even without the word "trend".

Return ONLY valid JSON: {{"intent": "summary"}} (or explanatory/comparison/trend/lookup)
No markdown. No explanation.
"""
    try:
        raw = await llm._call([
            {"role": "system", "content": "Return only JSON. No markdown. No explanation."},
            {"role": "user", "content": prompt},
        ])
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        intent = json.loads(clean).get("intent", "").strip().lower()
        if intent in ("explanatory", "comparison", "trend", "lookup", "summary"):
            logger.info("LLM query classification: %r → %s", query[:60], intent)
            return intent
        logger.warning("LLM returned unknown intent %r — falling back to keyword match", intent)
    except Exception as e:
        logger.warning("LLM query classification failed (%s) — using keyword fallback", e)
    return _keyword_classify_query(query)


# ─────────────────────────────────────────────
# LAYER 3b — Schema Inspector
# [CHANGE 2] LLM-enriched with dtype fallback
# ─────────────────────────────────────────────

def _dtype_inspect_schema(df: pd.DataFrame) -> dict:
    """Original dtype-based fallback."""
    metrics = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    cardinality_cap = max(2, len(df) // 10)
    dimensions = [c for c in cat_cols if df[c].nunique() <= cardinality_cap]

    date_col = None
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            date_col = col
            break
        if df[col].dtype == object:
            try:
                parsed = pd.to_datetime(df[col], errors="coerce")
                if parsed.notna().mean() > 0.7:
                    date_col = col
                    break
            except Exception:
                pass

    return {"metrics": metrics, "dimensions": dimensions, "date_col": date_col}


async def _llm_inspect_schema(df: pd.DataFrame, llm) -> dict:
    """
    Ask the LLM to annotate the dataset schema:
    - Which columns are real business metrics (numeric, meaningful to aggregate)?
    - Which columns are dimensions (categorical grouping columns)?
    - Which columns are junk (IDs, codes, phone numbers, etc.)?
    - Which column is the primary date/time column?

    Falls back to dtype-based inspection on failure.
    """
    sample_str = df.head(5).to_string()
    col_types = [(col, str(dt)) for col, dt in zip(df.columns, df.dtypes)]
    cardinalities = {col: int(df[col].nunique()) for col in df.columns}

    prompt = f"""You are a data schema analyst. Inspect the dataset below and annotate its columns.

Column names and dtypes: {col_types}
Cardinalities (unique value counts): {json.dumps(cardinalities)}

Sample rows:
{sample_str}

Your task:
1. Identify real BUSINESS METRIC columns — numeric columns that represent measurable quantities
   a business user would want to sum, average, or count (e.g. Revenue, Sales, Quantity, Profit, Rating).
   Exclude: row IDs, zip codes, phone numbers, postal codes, any number used as an identifier.

2. Identify DIMENSION columns — categorical columns meaningful for grouping or filtering
   (e.g. Region, Category, Product, Segment, Status).
   Exclude: free-text columns, high-cardinality columns with more than 50 unique values,
   columns that are effectively IDs (e.g. Order ID, Customer ID).

3. Identify the PRIMARY DATE column (if any) — the column most suitable for time-series analysis.
   Prefer columns whose dtype is datetime or whose values parse as dates.

Return ONLY valid JSON (no markdown, no explanation):
{{
  "metrics": ["col1", "col2"],
  "dimensions": ["col3", "col4"],
  "date_col": "col5_or_null"
}}
"""
    try:
        raw = await llm._call([
            {"role": "system", "content": "Return only JSON. No markdown. No explanation."},
            {"role": "user", "content": prompt},
        ])
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        schema = json.loads(clean)
        # Validate that returned columns actually exist in df
        valid_cols = set(df.columns)
        metrics = [c for c in schema.get("metrics", []) if c in valid_cols]
        dimensions = [c for c in schema.get("dimensions", []) if c in valid_cols]
        date_col = schema.get("date_col")
        if date_col and date_col not in valid_cols:
            date_col = None
        logger.info("LLM schema: %d metrics, %d dims, date_col=%s", len(metrics), len(dimensions), date_col)
        return {"metrics": metrics, "dimensions": dimensions, "date_col": date_col}
    except Exception as e:
        logger.warning("LLM schema inspection failed (%s) — using dtype fallback", e)
        return _dtype_inspect_schema(df)


# ─────────────────────────────────────────────
# LAYER 3c — Reasoning Blocks  (pandas; unchanged)
# ─────────────────────────────────────────────

def pick_metric(metrics, query):
    q = query.lower()
    for m in metrics:
        if m.lower() in q:
            return m
    for m in metrics:
        if any(word in q for word in re.split(r'[_\s]', m.lower())):
            return m
    return metrics[0]


def build_reasoning_context(df, query, query_type, schema) -> list[dict]:
    """
    Computes deterministic pandas analytics blocks (unchanged from v1).
    These are fed to the LLM narration step as PRIMARY DATA.
    """
    metrics    = schema["metrics"]
    dimensions = schema["dimensions"]
    date_col   = schema["date_col"]

    if not metrics:
        return []

    primary_metric = pick_metric(metrics, query)
    insights: list[dict] = []

    if query_type == "explanatory":
        if date_col:
            try:
                insights.append(_block_change_detection(df, primary_metric, date_col))
                insights.append(_block_trend(df, primary_metric, date_col))
            except Exception as e:
                logger.warning("Trend/change block failed: %s", e)
        for dim in dimensions[:2]:
            try:
                insights.append(_block_top_contributors(df, primary_metric, dim))
                insights.append(_block_contribution_pct(df, primary_metric, dim))
                if date_col:
                    insights.append(_block_driver_detection(df, primary_metric, dim, date_col))
            except Exception as e:
                logger.warning("Contributor/pct block failed for %s: %s", dim, e)

    elif query_type == "comparison":
        for dim in dimensions[:3]:
            try:
                insights.append(_block_segmentation(df, primary_metric, dim))
                insights.append(_block_contribution_pct(df, primary_metric, dim))
            except Exception as e:
                logger.warning("Segmentation block failed for %s: %s", dim, e)

    elif query_type == "trend":
        if date_col:
            try:
                insights.append(_block_trend(df, primary_metric, date_col))
                insights.append(_block_change_detection(df, primary_metric, date_col))
            except Exception as e:
                logger.warning("Trend block failed: %s", e)

    logger.info(
        "Reasoning context: query_type=%s metric=%s dims=%s blocks=%d",
        query_type, primary_metric, dimensions[:3], len(insights),
    )
    return insights


# ─────────────────────────────────────────────
# LAYER 3c helpers — Reasoning Blocks  (unchanged)
# ─────────────────────────────────────────────

def _block_top_contributors(df, metric, dimension, n=5):
    grouped = df.groupby(dimension)[metric].sum().sort_values(ascending=False)
    total = grouped.sum()
    top = grouped.head(n)
    return {
        "block": "top_contributors",
        "metric": metric,
        "dimension": dimension,
        "total": clean_value(total),
        "leaders": [
            {
                "segment": clean_value(k),
                "value": clean_value(v),
                "pct": round(clean_value(v) / clean_value(total) * 100, 1) if clean_value(total) else None,
            }
            for k, v in top.items()
        ],
    }


def _block_change_detection(df, metric, date_col):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col)
    mid = len(df) // 2
    before = df.iloc[:mid][metric].sum()
    after  = df.iloc[mid:][metric].sum()
    change = after - before
    pct    = round(change / before * 100, 1) if before else None
    return {
        "block": "change_detection",
        "metric": metric,
        "date_col": date_col,
        "first_half": clean_value(before),
        "second_half": clean_value(after),
        "absolute_change": clean_value(change),
        "pct_change": pct,
        "direction": "up" if change > 0 else ("down" if change < 0 else "flat"),
    }


def _block_segmentation(df, metric, dimension):
    grouped = df.groupby(dimension)[metric].agg(["mean", "std", "count"]).reset_index()
    grouped.columns = [dimension, "mean", "std", "count"]
    grouped = grouped.sort_values("mean", ascending=False)
    return {
        "block": "segmentation",
        "metric": metric,
        "dimension": dimension,
        "segments": [
            {
                "segment": clean_value(row[dimension]),
                "mean": round(clean_value(row["mean"]), 2),
                "std":  round(clean_value(row["std"]), 2) if pd.notna(row["std"]) else None,
                "count": int(row["count"]),
            }
            for _, row in grouped.iterrows()
        ],
    }


def _block_trend(df, metric, date_col):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    monthly = (
        df.resample("ME", on=date_col)[metric]
        .sum()
        .reset_index()
        .rename(columns={date_col: "period", metric: "value"})
    )
    monthly["period"] = monthly["period"].dt.to_period("M").astype(str)
    return {
        "block": "trend",
        "metric": metric,
        "date_col": date_col,
        "periods": [
            {"period": row["period"], "value": clean_value(row["value"])}
            for _, row in monthly.iterrows()
        ],
    }


def _block_contribution_pct(df, metric, dimension):
    grouped = df.groupby(dimension)[metric].sum()
    total = grouped.sum()
    shares = (grouped / total * 100).round(1).sort_values(ascending=False)
    return {
        "block": "contribution_pct",
        "metric": metric,
        "dimension": dimension,
        "total": clean_value(total),
        "shares": [
            {"segment": clean_value(k), "pct": clean_value(v)}
            for k, v in shares.items()
        ],
    }


def _block_driver_detection(df, metric, dimension, date_col):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col)
    mid = len(df) // 2
    before = df.iloc[:mid].groupby(dimension)[metric].sum()
    after  = df.iloc[mid:].groupby(dimension)[metric].sum()
    combined = pd.DataFrame({"before": before, "after": after}).fillna(0)
    combined["change"] = combined["after"] - combined["before"]

    if combined["change"].abs().sum() == 0:
        return {"block": "driver_detection", "dimension": dimension, "drivers": [], "note": "no_change"}

    combined = combined.sort_values("change", ascending=False)
    total_change = combined["change"].abs().sum()
    threshold = 0.05 * total_change
    top_drivers = combined[combined["change"].abs() >= threshold].head(3)

    return {
        "block": "driver_detection",
        "metric": metric,
        "dimension": dimension,
        "drivers": [
            {
                "segment": clean_value(idx),
                "change": clean_value(row["change"]),
                "before": clean_value(row["before"]),
                "after": clean_value(row["after"]),
            }
            for idx, row in top_drivers.iterrows()
        ],
    }


# ─────────────────────────────────────────────
# LAYER 4 — Result Verification (NEW)
# [CHANGE 4] LLM cross-checks the interpreted result
# ─────────────────────────────────────────────

async def _llm_verify_result(query: str, meta: dict, llm) -> dict:
    """
    Cross-verify the interpreted result. Returns a dict with:
      - "ok": True/False — whether the result appears to answer the query
      - "warning": str | None — any anomaly detected (empty when expected data,
                                suspicious scalar, mismatched columns, etc.)

    This is a lightweight sanity check, not a transformation. If the LLM
    call fails we return {"ok": True, "warning": None} to avoid blocking the pipeline.
    """
    if meta["type"] == "empty":
        # Always worth flagging empty results to the LLM
        pass

    prompt = f"""You are a result validator for a data analytics pipeline.

User query: {query}

Computed result metadata:
{json.dumps(meta, indent=2)}

Check:
1. Does the result type make sense for the query? (e.g. scalar for "what is the total", table for "list all")
2. If type is "empty" — is that plausible, or does the query clearly expect data?
3. If type is "scalar" — is the value in a reasonable range given the query?
4. If type is "table" — do the column names match what the query asked for?

Return ONLY valid JSON:
{{"ok": true, "warning": null}}
OR
{{"ok": false, "warning": "short description of the issue"}}
No markdown. No explanation outside the JSON.
"""
    try:
        raw = await llm._call([
            {"role": "system", "content": "Return only JSON. No markdown."},
            {"role": "user", "content": prompt},
        ])
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        verdict = json.loads(clean)
        ok = bool(verdict.get("ok", True))
        warning = verdict.get("warning") or None
        if warning:
            logger.warning("Result verification warning: %s", warning)
        return {"ok": ok, "warning": warning}
    except Exception as e:
        logger.warning("Result verification failed (%s) — skipping", e)
        return {"ok": True, "warning": None}


# ─────────────────────────────────────────────
# HELPERS — History formatting  (unchanged)
# ─────────────────────────────────────────────

def format_history(history: list[dict]) -> str:
    if not history:
        return "None"
    lines = []
    for msg in history[-6:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"] if isinstance(msg["content"], str) else msg["content"].get("answer", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


# ─────────────────────────────────────────────
# LAYER 3d — Summary Analytics  (NEW)
# Runs ALL analytics blocks across ALL metrics & dims
# to power the comprehensive summary response.
# ─────────────────────────────────────────────

def build_summary_context(df: pd.DataFrame, schema: dict) -> dict:
    """
    Build a comprehensive analytics context for summary/insight queries.
    Runs all available blocks across metrics and dimensions.
    Returns a rich dict that the LLM uses to produce a structured report.
    """
    metrics    = schema["metrics"]
    dimensions = schema["dimensions"]
    date_col   = schema["date_col"]

    if not metrics:
        return {}

    summary = {
        "shape": {"rows": int(len(df)), "cols": int(len(df.columns))},
        "metrics": {},
        "dimensions": {},
        "trends": {},
        "correlations": [],
    }

    # ── Dataset date range ────────────────────────────────────────────────
    if date_col:
        try:
            dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
            if not dates.empty:
                summary["date_range"] = {
                    "start": str(dates.min().date()),
                    "end":   str(dates.max().date()),
                    "days":  int((dates.max() - dates.min()).days),
                }
        except Exception:
            pass

    # ── Per-metric stats ──────────────────────────────────────────────────
    for metric in metrics[:6]:  # cap at 6 metrics
        try:
            col = df[metric].dropna()
            stats = {
                "total":  clean_value(col.sum()),
                "mean":   round(float(col.mean()), 2),
                "median": round(float(col.median()), 2),
                "min":    clean_value(col.min()),
                "max":    clean_value(col.max()),
                "count":  int(col.count()),
            }
            # Per-dimension breakdown for top 2 dims
            dim_breakdown = {}
            for dim in dimensions[:2]:
                try:
                    grp = df.groupby(dim)[metric].agg(["sum", "mean", "count"]).reset_index()
                    grp.columns = [dim, "total", "mean", "count"]
                    grp = grp.sort_values("total", ascending=False)
                    dim_breakdown[dim] = [
                        {
                            "segment": clean_value(r[dim]),
                            "total":   clean_value(r["total"]),
                            "mean":    round(float(r["mean"]), 2),
                            "count":   int(r["count"]),
                            "pct":     round(float(r["total"]) / float(col.sum()) * 100, 1)
                                       if col.sum() else None,
                        }
                        for _, r in grp.head(6).iterrows()
                    ]
                except Exception:
                    pass
            stats["by_dimension"] = dim_breakdown

            # Time trend (monthly totals)
            if date_col:
                try:
                    tmp = df.copy()
                    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
                    monthly = (
                        tmp.dropna(subset=[date_col])
                        .resample("ME", on=date_col)[metric]
                        .sum()
                        .reset_index()
                    )
                    monthly["period"] = monthly[date_col].dt.to_period("M").astype(str)
                    periods = [
                        {"period": r["period"], "value": clean_value(r[metric])}
                        for _, r in monthly.iterrows()
                    ]
                    stats["trend"] = {
                        "periods": periods,
                        "peak_period": max(periods, key=lambda x: x["value"] or 0) if periods else None,
                        "low_period":  min(periods, key=lambda x: x["value"] or 0) if periods else None,
                    }
                except Exception:
                    pass

            summary["metrics"][metric] = stats
        except Exception as e:
            logger.warning("Summary metric block failed for %s: %s", metric, e)

    # ── Dimension cardinality & top values ────────────────────────────────
    for dim in dimensions[:5]:
        try:
            vc = df[dim].value_counts().reset_index()
            vc.columns = [dim, "count"]
            summary["dimensions"][dim] = {
                "unique": int(df[dim].nunique()),
                "top_values": [
                    {"value": clean_value(r[dim]), "count": int(r["count"])}
                    for _, r in vc.head(5).iterrows()
                ],
            }
        except Exception:
            pass

    # ── Numeric correlations ──────────────────────────────────────────────
    try:
        num_df = df[metrics].dropna()
        if len(metrics) >= 2 and len(num_df) > 10:
            corr = num_df.corr(numeric_only=True)
            pairs = _extract_top_correlations(corr)
            summary["correlations"] = pairs.to_dict(orient="records") if not pairs.empty else []
    except Exception:
        pass

    return summary


async def _llm_generate_summary(
    query: str,
    summary_context: dict,
    schema: dict,
    history_block: str,
    llm,
) -> str:
    """
    Generate a comprehensive structured summary report from the analytics context.
    Returns a JSON string that the frontend renders as a rich report.
    """
    metrics    = schema["metrics"]
    dimensions = schema["dimensions"]
    date_col   = schema.get("date_col")

    prompt = f"""You are a senior data analyst. The user asked: "{query}"

You have been given a comprehensive analytics context computed from their dataset.
Generate a DETAILED, STRUCTURED analysis report as a JSON object.

Analytics context:
{json.dumps(summary_context, indent=2, default=str)}

Instructions:
- Use ONLY the data provided. Do not invent numbers.
- Be specific: cite actual values, percentages, and segment names from the context.
- Format large numbers human-readably (e.g. $2.3M, 12.4K, 87.3%).
- The report should be thorough enough for both a business executive and a data analyst.
- Include ALL metrics from the context, ALL dimension breakdowns, and any trends.
- Use plain, clear English. Avoid jargon unless the data naturally suggests it.
- Adapt the depth to the data: if there's a lot of rich data, give a lot of detail.

Return ONLY a valid JSON object with this exact structure (no markdown, no extra text):
{{
  "report_type": "summary",
  "title": "Descriptive title for this analysis",
  "overview": "2-3 sentence executive overview of the entire dataset",
  "date_range": "e.g. Jan 2014 – Dec 2017 (4 years) or null",
  "key_metrics": [
    {{"label": "Total Revenue", "value": "$2.3M", "note": "optional context"}},
    ...
  ],
  "sections": [
    {{
      "heading": "Section heading (e.g. Category Performance)",
      "body": "2-4 sentences of insight for this section",
      "subsections": [
        {{
          "name": "Subsection name (e.g. Technology)",
          "stats": [
            {{"label": "Sales", "value": "$836K"}},
            {{"label": "Margin", "value": "17.4%"}}
          ],
          "note": "1 sentence key takeaway"
        }}
      ]
    }}
  ],
  "recommendations": [
    "Specific actionable recommendation 1",
    "Specific actionable recommendation 2",
    "Specific actionable recommendation 3"
  ]
}}

Generate sections for: overview metrics, then one section per dimension breakdown found in the context, then trends (if date data exists), then correlations (if any), then recommendations.
Tailor the key_metrics array to the most important 4-8 KPIs visible in the data.
"""

    try:
        raw = await llm._call([
            {"role": "system", "content": "Return ONLY valid JSON. No markdown. No preamble. No explanation."},
            {"role": "user", "content": prompt},
        ])
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        # Validate it parses
        parsed = json.loads(clean)
        parsed["report_type"] = "summary"  # ensure marker is set
        return json.dumps(parsed)
    except Exception as e:
        logger.error("LLM summary generation failed: %s", e)
        # Fallback: plain text summary
        return json.dumps({
            "report_type": "summary",
            "title": "Dataset Analysis",
            "overview": "A summary could not be fully generated. Please try a more specific question.",
            "key_metrics": [],
            "sections": [],
            "recommendations": [],
        })




class ChatService:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self._llm = LLMClient()

    async def chat(self, query: str, history: list[dict] | None = None) -> dict:
        history = history or []
        columns = list(self.df.columns)
        history_block = format_history(history)

        # ── STEP -1: OFF-TOPIC GUARD — fires before everything else ──────────
        # Two-layer: fast keyword pre-filter → LLM judgment for uncertain cases.
        # Fail-open: on LLM failure, the query is allowed through so legitimate
        # queries are never wrongly blocked.
        is_off_topic, rejection = await guard_offtopic(query, columns, self._llm)
        if is_off_topic:
            return rejection

        # ── STEP 0: CLASSIFY QUERY — LLM-driven  [CHANGE 1] ──────────────────
        query_type = await _llm_classify_query(query, columns, self._llm)
        logger.info("Query type: %s | Query: %s", query_type, query)

        sample  = self.df.head(5).to_string()
        stats   = self.df.describe(include="all").to_string()

        # ── STEP 0b: INSPECT SCHEMA — LLM-driven  [CHANGE 2] ─────────────────
        schema = await _llm_inspect_schema(self.df, self._llm)

        # ── SUMMARY PATH — fires before reasoning/code for broad queries ──────
        if query_type == "summary":
            logger.info("Summary path triggered for query: %s", query)
            summary_context = build_summary_context(self.df, schema)
            summary_json = await _llm_generate_summary(
                query=query,
                summary_context=summary_context,
                schema=schema,
                history_block=history_block,
                llm=self._llm,
            )
            return {"answer": summary_json, "table": None, "is_summary": True}

        # ── REASONING LAYER — pandas blocks + LLM narration  [CHANGE 3] ──────
        reasoning_insights: list[dict] = []
        if query_type in ("explanatory", "comparison", "trend"):
            reasoning_insights = build_reasoning_context(self.df, query, query_type, schema)

        # ── STEP 1: LLM → CODE ───────────────────────────────────────────────
        code_prompt = f"""You are a Python data analyst.

Conversation so far:
{history_block}

Current user query: {query}

Dataset columns: {columns}

Sample (first 5 rows):
{sample}

Stats:
{stats}

{_PANDAS_COMPAT_RULES}

Write pandas code to answer the current query.
Use the conversation history only to resolve references like "it", "that column", "the same period".

Additional rules:
- Use `df` as the dataframe variable.
- Store final output in a variable named `result`.
- `result` must be a flat DataFrame, Series, or scalar (never a named-index DataFrame).
- DO NOT modify `df` in-place (no df['col'] = ...). Use a copy or intermediate variable.
- For trend/direction questions, compute the actual values (e.g. monthly totals) as a DataFrame.
- Write flat code — no functions, no classes.
- Return ONLY raw Python code. No JSON. No markdown. No explanation.
- If no computation is needed, return exactly: result = None
- Do NOT use import statements — pd and np are already available.
"""

        code_response = await self._llm._call([
            {"role": "system", "content": "Return ONLY raw Python code. No JSON. No markdown. No explanation."},
            {"role": "user",   "content": code_prompt},
        ])

        logger.info("LLM code response:\n%s", code_response)

        # ── STEP 2: PARSE + EXECUTE ───────────────────────────────────────────
        code = re.sub(r"```(?:python)?|```", "", code_response).strip()

        result, exec_error = None, None
        if code:
            result, exec_error = await safe_exec(code, self.df, self._llm)

        if result is None:
            if exec_error:
                logger.error("All exec attempts failed: %s", exec_error)
                return {
                    "answer": "I wasn't able to compute that. Try rephrasing your query.",
                    "table": None,
                }
            return {"answer": "No meaningful result could be computed.", "table": None}

        # ── STEP 3: INTERPRET + VERIFY  [CHANGE 4] ────────────────────────────
        meta = interpret_result(result)
        logger.info("Result metadata: %s", meta)

        # Cross-verify with LLM — non-blocking; only used to surface warnings
        verification = await _llm_verify_result(query, meta, self._llm)
        if not verification["ok"]:
            logger.warning("Result verification failed: %s", verification["warning"])
            # Don't abort — the LLM narration will handle the situation gracefully

        # ── STEP 4: TABLE DECISION — LLM-driven  [CHANGE 5] ──────────────────
        show_table = await _llm_table_decision(query, meta, columns, self._llm)
        table = to_table_records(result, meta) if show_table else None

        # ── STEP 5: ANSWER — always LLM-driven  [CHANGE 6] ───────────────────
        # NOTE: deterministic_answer() is intentionally removed.
        # Even scalars ("The result is 123.") go through the LLM to produce
        # contextual, human-readable answers with units, context, and insight.

        if exec_error:
            answer = "There was an issue processing your query. Please try rephrasing it."
        else:
            answer = await self._llm_narrate(
                query=query,
                query_type=query_type,
                meta=meta,
                reasoning_insights=reasoning_insights,
                history_block=history_block,
                verification=verification,
            )

        logger.info("Table shown: %s | Answer: %s", show_table, answer)
        return {"answer": answer, "table": table}

    async def _llm_narrate(
        self,
        query: str,
        query_type: str,
        meta: dict,
        reasoning_insights: list[dict],
        history_block: str,
        verification: dict,
    ) -> str:
        """
        Single unified narration step for ALL result types.
        [CHANGE 6] Replaces the deterministic_answer() shortcut — every result
        (scalar, boolean, list, table) gets a contextual LLM-generated answer.
        """

        # Build the reasoning section (if any analytics blocks were computed)
        reasoning_section = ""
        if reasoning_insights:
            reasoning_section = (
                "\n\n🚨 PRIMARY ANALYTICS DATA (use this first):\n"
                + json.dumps(reasoning_insights, indent=2)
            )

        # Surface any verification warnings to the LLM
        verification_note = ""
        if not verification["ok"] and verification["warning"]:
            verification_note = (
                f"\n\n⚠️ RESULT ANOMALY DETECTED: {verification['warning']}\n"
                "Address this in your answer if relevant."
            )

        guardrail = (
            "\n\n🚨 RULES (STRICT — DO NOT VIOLATE):"
            "\n1. ALWAYS use the PRIMARY ANALYTICS DATA if present."
            "\n2. DO NOT say 'not enough information' if PRIMARY DATA exists."
            "\n3. DO NOT ignore driver_detection, trend, or change_detection blocks."
            "\n\nTREND QUERY RULES:"
            "\n- Use change_detection.direction (up/down/flat) as final answer."
            "\n- Use trend.periods to support with values."
            "\n\nEXPLANATORY QUERY RULES:"
            "\n- MUST use driver_detection."
            "\n- Explain which segments contributed to the change using 'change' values."
            "\n- If driver_detection.drivers is empty, say: 'There is no significant change.'"
            "\n\nCOMPARISON QUERY RULES:"
            "\n- Use segmentation or contribution_pct blocks to compare."
            "\n- Mention top segments explicitly."
            "\n\nSCALAR RESULT RULES:"
            "\n- NEVER just repeat the raw number. Add context: what metric, what time frame."
            "\n- Use human-readable formatting: ~1.23M instead of 1234567.89"
            "\n- If the scalar is a count, say what is being counted."
            "\n- If the scalar is a rate or percentage, say it is a percentage."
            "\n\nBOOLEAN RESULT RULES:"
            "\n- Translate to plain English: 'Yes, ...' or 'No, ...' with supporting detail."
            "\n\n🚫 NEVER:"
            "\n- speculate reasons"
            "\n- ignore provided data"
            "\n- give generic filler answers"
            "\n- say 'The result is X' with no other context"
            "\n\nOUTPUT STYLE:"
            "\n- Human-readable and concise (1–3 sentences)"
            "\n- Prefer insights over raw number dumps"
            "\n- Use 'driven by' or 'main contributors are' instead of 'because'"
            "\n- Avoid 'first half vs second half'; use 'early vs recent'"
        )

        narration_prompt = f"""You are a data analyst assistant.

Conversation so far:
{history_block}

Current user query: {query}

Computed result:
{json.dumps(meta, indent=2)}
{reasoning_section}
{verification_note}
{guardrail}

Instructions:
- Give a clear, direct answer to the query.
- Support with 1–2 key insights from the data.
- Use approximate numbers for large values (~13M, ~4.5K).
- Focus on what changed or what matters, not raw dumps.
- If PRIMARY DATA contains a direction, use it directly.
- If driver_detection exists, use it to explain WHY.

Write 1–3 concise sentences. No bullet points. No markdown.
"""
        answer = await self._llm._call([
            {"role": "system", "content": "Answer using only the provided data. Cite specific values. No speculation. No markdown."},
            {"role": "user",   "content": narration_prompt},
        ])
        return answer.strip()