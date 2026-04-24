import textwrap
import json
import re
import pandas as pd
from app.pipeline.llm_client import LLMClient
import logging
import asyncio

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# UNIVERSAL CLEANER
# ─────────────────────────────────────────────

def clean_value(val):
    if pd.isna(val):
        return None

    # FIX: handle Period (your crash issue)
    if isinstance(val, pd.Period):
        return str(val)

    # datetime
    if isinstance(val, pd.Timestamp):
        # if it's monthly data → show YYYY-MM
        return val.strftime("%Y-%m")

    # duration
    if isinstance(val, pd.Timedelta):
        return str(val)

    # numpy types (int64, float64, bool_)
    if hasattr(val, "item") and not isinstance(val, (pd.Series, pd.DataFrame)):
        return val.item()

    return val


def clean_records(df, n=3):
    return [
        {k: clean_value(v) for k, v in row.items()}
        for row in df.head(n).to_dict(orient="records")
    ]



# ─────────────────────────────────────────────
# SAFE EXEC — normalise + execute + retry
# ─────────────────────────────────────────────

_SAFE_GLOBALS = {
    "__builtins__": {},
    "pd": pd
}  # whitelist only — no builtins, no os, no sys

def _normalise_code(code: str) -> str:
    """dedent + strip so LLM indentation quirks don't crash exec."""
    return textwrap.dedent(code).strip()

async def safe_exec(code: str, df: pd.DataFrame, llm) -> tuple:
    """
    Execute LLM-generated pandas code safely with a timeout.
    Returns (result, error_string | None).
    """
    import asyncio
    
    async def _run_with_timeout(c, d, timeout=10.0):
        try:
            # Offload sync exec to a thread and enforce a time limit
            return await asyncio.wait_for(asyncio.to_thread(_run_code, c, d), timeout=timeout)
        except asyncio.TimeoutError:
            logger.error("Execution timed out after %s seconds.", timeout)
            return None, "Execution timed out. The operation took too long or got stuck in a loop."

    code = _normalise_code(code)
    result, exec_error = await _run_with_timeout(code, df)

    if exec_error and "timed out" not in exec_error:
        logger.warning("First exec failed (%s) — asking LLM to fix", exec_error)
        code = await _fix_code(code, exec_error, llm)
        if code:
            result, exec_error = await _run_with_timeout(code, df)

    return result, exec_error

def _run_code(code: str, df: pd.DataFrame):
    try:
        local_vars = {"df": df.copy()}   # always copy — never mutate original
        exec(code, _SAFE_GLOBALS, local_vars)
        return local_vars.get("result"), None
    except Exception as e:
        logger.error("Exec error: %s\nCode:\n%s", e, code)
        return None, str(e)


async def _fix_code(bad_code: str, error: str, llm) -> str:
    """One-shot LLM correction attempt."""
    retry_prompt = f"""Fix this Python pandas code.

Error:
{error}

Code:
{bad_code}

Rules:
- Use `df` as the dataframe variable.
- Store result in `result`.
- Do NOT use import statements.
- Do NOT modify df in-place (no df['col'] = ...).
- Return ONLY valid JSON: {{"code": "corrected code here"}}
"""
    try:
        resp = await llm._call([
            {"role": "system", "content": "Return valid JSON only. No markdown."},
            {"role": "user", "content": retry_prompt},
        ])
        clean = re.sub(r"```(?:json)?|```", "", resp).strip()
        fixed = json.loads(clean).get("code", "").strip()
        logger.info("LLM-fixed code:\n%s", fixed)
        return _normalise_code(fixed) if fixed else ""
    except Exception as e:
        logger.error("Fix attempt failed: %s", e)
        return ""


# ─────────────────────────────────────────────
# LAYER 1 — Result Interpreter
# ─────────────────────────────────────────────

def interpret_result(result) -> dict:
    if result is None:
        return {"type": "empty"}

    # bool must be checked BEFORE int/float — bool is a subclass of int in Python,
    # and numpy.bool_ has .item(), so they'd be misclassified as scalars otherwise.
    if isinstance(result, bool):
        return {"type": "boolean", "value": bool(result)}
    if hasattr(result, "item") and not isinstance(result, (pd.Series, pd.DataFrame)):
        # numpy.bool_ → convert via item() then re-check type
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
# LAYER 2 — Table Decision + Sorting
# ─────────────────────────────────────────────

def should_show_table(meta: dict) -> bool:
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


def to_table_records(result, meta: dict):
    """Serialise result to records. Sort by first numeric column descending."""
    if isinstance(result, pd.DataFrame):
        df = result.head(50).copy()
        # ✅ FIX: if time column exists → sort chronologically
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
            # fallback: numeric sort
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            if numeric_cols:
                df = df.sort_values(numeric_cols[0], ascending=False)
        return [{k: clean_value(v) for k, v in row.items()} for row in df.to_dict(orient="records")]

    if isinstance(result, pd.Series):
        df = result.head(50).reset_index()
        return [{k: clean_value(v) for k, v in row.items()} for row in df.to_dict(orient="records")]

    return None


# ─────────────────────────────────────────────
# LAYER 3 — Query Classifier
# Decides pipeline depth without an LLM call.
# ─────────────────────────────────────────────

EXPLANATION_TRIGGERS = ["why", "reason", "cause", "explain", "how come", "what caused"]
COMPARISON_TRIGGERS  = ["compare", "vs", "versus", "difference between", "which is better"]
TREND_TRIGGERS       = [
    "increase", "decrease", "increasing", "decreasing",
    "trend", "growth", "decline", "up", "down",
]

def classify_query(query: str) -> str:
    q = query.lower()
    if any(t in q for t in EXPLANATION_TRIGGERS):
        return "explanatory"
    if any(t in q for t in COMPARISON_TRIGGERS):
        return "comparison"
    if any(t in q for t in TREND_TRIGGERS):
        return "trend"
    return "lookup"


# ─────────────────────────────────────────────
# LAYER 3b — Schema Inspector
# Detects numeric metrics + categorical dimensions
# from column types — no hardcoding, any dataset.
# ─────────────────────────────────────────────

def inspect_schema(df: pd.DataFrame) -> dict:
    """
    Returns the dataset's structural skeleton:
      - metrics:    numeric columns suitable for aggregation
      - dimensions: low-cardinality categorical columns suitable for groupby
      - date_col:   first datetime-like column found (or None)

    Cardinality cap keeps dimensions meaningful — a column with 10k
    unique string values is an ID, not a dimension.
    """
    metrics = df.select_dtypes(include="number").columns.tolist()

    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    cardinality_cap = max(2, len(df) // 10)   # at most 10% unique values
    dimensions = [c for c in cat_cols if df[c].nunique() <= cardinality_cap]

    date_col = None
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            date_col = col
            break
        # also catch string columns that look like dates
        if df[col].dtype == object:
            try:
                parsed = pd.to_datetime(df[col], errors="coerce")
                if parsed.notna().mean() > 0.7:
                    date_col = col
                    break
            except Exception:
                pass

    return {"metrics": metrics, "dimensions": dimensions, "date_col": date_col}


# ─────────────────────────────────────────────
# LAYER 3c — Reasoning Blocks
# Five generic analysis blocks that work on
# any dataset — no domain assumptions.
# ─────────────────────────────────────────────

def _block_top_contributors(df: pd.DataFrame, metric: str, dimension: str, n: int = 5) -> dict:
    """Which segment drives the most of a metric?"""
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


def _block_change_detection(df: pd.DataFrame, metric: str, date_col: str) -> dict:
    """First-half vs second-half of the date range — simple change signal."""
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


def _block_segmentation(df: pd.DataFrame, metric: str, dimension: str) -> dict:
    """Mean + std per segment — surfaces outlier segments."""
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


def _block_trend(df: pd.DataFrame, metric: str, date_col: str) -> dict:
    """Monthly (or period-level) totals to expose trajectory."""
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


def _block_contribution_pct(df: pd.DataFrame, metric: str, dimension: str) -> dict:
    """What share (%) does each segment own of the total?"""
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

def _block_driver_detection(df: pd.DataFrame, metric: str, dimension: str, date_col: str) -> dict:
    """
    Finds which segments contributed most to the change over time.
    Generic — works for any dataset.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    # split into two halves (before vs after)
    df = df.sort_values(date_col)
    mid = len(df) // 2
    before_df = df.iloc[:mid]
    after_df = df.iloc[mid:]

    before = before_df.groupby(dimension)[metric].sum()
    after = after_df.groupby(dimension)[metric].sum()

    # align indexes
    combined = pd.DataFrame({
        "before": before,
        "after": after
    }).fillna(0)

    combined["change"] = combined["after"] - combined["before"]

    if combined["change"].abs().sum() == 0:
        return {
            "block": "driver_detection",
            "dimension": dimension,
            "drivers": [],
            "note": "no_change"
        }

    # sort by biggest positive change
    combined = combined.sort_values("change", ascending=False)

    total_change = combined["change"].abs().sum()
    threshold = 0.05 * total_change

    filtered = combined[combined["change"].abs() >= threshold]
    top_drivers = filtered.head(3)

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
        ]
    }

# ─────────────────────────────────────────────
# LAYER 3d — Reasoning Router
# Selects and runs the right blocks for each
# query type, returns structured insight payload.
# ─────────────────────────────────────────────

def pick_metric(metrics, query):
    q = query.lower()

    # exact or partial match
    for m in metrics:
        if m.lower() in q:
            return m

    # fuzzy word match
    for m in metrics:
        if any(word in q for word in re.split(r'[_\s]', m.lower())):
            return m

    return metrics[0]

def build_reasoning_context(
    df: pd.DataFrame,
    query: str,
    query_type: str,
    schema: dict,
) -> list[dict]:
    """
    Runs the appropriate reasoning blocks and returns a list of
    insight dicts that get injected into the LLM narration prompt.

    Works on any dataset — selection is driven purely by schema
    (metrics / dimensions / date_col), never by column names.
    """
    metrics    = schema["metrics"]
    dimensions = schema["dimensions"]
    date_col   = schema["date_col"]

    if not metrics:
        return []   # nothing numeric to reason about

    primary_metric = pick_metric(metrics, query)

    insights: list[dict] = []

    if query_type == "explanatory":
        
        # WHY pipeline: change + top contributors + contribution % per dimension
        if date_col:
            try:
                insights.append(_block_change_detection(df, primary_metric, date_col))
                insights.append(_block_trend(df, primary_metric, date_col))
            except Exception as e:
                logger.warning("Trend/change block failed: %s", e)

        for dim in dimensions[:2]:   # cap at 2 dims to keep prompt tight
            try:
                insights.append(_block_top_contributors(df, primary_metric, dim))
                insights.append(_block_contribution_pct(df, primary_metric, dim))

                if date_col:
                    insights.append(_block_driver_detection(df, primary_metric, dim, date_col))
                            
            except Exception as e:
                logger.warning("Contributor/pct block failed for %s: %s", dim, e)

    elif query_type == "comparison":
        # WHICH-IS-BETTER pipeline: segmentation across every dimension
        for dim in dimensions[:3]:
            try:
                insights.append(_block_segmentation(df, primary_metric, dim))
                insights.append(_block_contribution_pct(df, primary_metric, dim))
            except Exception as e:
                logger.warning("Segmentation block failed for %s: %s", dim, e)

    elif query_type == "trend":
        # TREND pipeline: trajectory + direction signal
        if date_col:
            try:
                insights.append(_block_trend(df, primary_metric, date_col))
                insights.append(_block_change_detection(df, primary_metric, date_col))
            except Exception as e:
                logger.warning("Trend block failed: %s", e)

    # lookup queries don't use this path — they go straight to exec + narrate

    logger.info(
        "Reasoning context: query_type=%s metric=%s dims=%s blocks=%d",
        query_type, primary_metric, dimensions[:3], len(insights),
    )
    return insights


# ─────────────────────────────────────────────
# LAYER 4 — Deterministic Answer
# No LLM for simple cases.
# ─────────────────────────────────────────────

def deterministic_answer(query: str, meta: dict) -> str | None:
    t = meta["type"]
    if t == "empty":
        return "No results were found for your query."
    if t == "scalar":
        v = meta["value"]
        formatted = f"{v:,.2f}" if v != int(v) else f"{int(v):,}"
        return f"The result is {formatted}."
    if t == "boolean":
        # Don't return a bare "Yes/No" — always let LLM narrate with the query
        # context so the answer is semantically meaningful (e.g. "Sales are declining.")
        return None
    if t == "list" and meta["count"] == 1:
        return f"Found 1 value: {meta['sample'][0]}."
    if t == "table" and meta["rows"] == 1 and meta["cols"] == 1:
        val = list(meta["sample"][0].values())[0]
        return f"The result is {val}."
    return None  # needs LLM narration


# ─────────────────────────────────────────────
# HELPERS — History formatting
# ─────────────────────────────────────────────

def format_history(history: list[dict]) -> str:
    """Convert [{role, content}] to a readable block for the LLM prompt."""
    if not history:
        return "None"
    lines = []
    for msg in history[-6:]:  # last 3 turns (6 messages) keeps context tight
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"] if isinstance(msg["content"], str) else msg["content"].get("answer", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


# ─────────────────────────────────────────────
# SERVICE
# ─────────────────────────────────────────────

class ChatService:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self._llm = LLMClient()

    async def chat(self, query: str, history: list[dict] | None = None) -> dict:
        """
        Flow:
          1. Classify query (lookup / comparison / explanatory)
          2. LLM → pandas code only, with conversation history for context
          3. Backend → exec → raw result
          4. interpret_result → structured metadata
          5. should_show_table → logic decision
          6. deterministic_answer → no LLM for simple cases
          7. LLM narration (fallback) — grounded strictly to metadata,
             never allowed to speculate beyond the data
        """

        history = history or []
        query_type = classify_query(query)
        logger.info("Query type: %s | Query: %s", query_type, query)

        sample = self.df.head(5).to_string()
        stats = self.df.describe(include="all").to_string()
        columns = list(self.df.columns)
        history_block = format_history(history)

        # ── REASONING LAYER (explanatory + comparison + trend only) ────
        # Builds structured insight blocks deterministically before any
        # LLM call. For lookup queries this is skipped — no overhead.
        reasoning_insights: list[dict] = []
        if query_type in ("explanatory", "comparison", "trend"):
            schema = inspect_schema(self.df)
            reasoning_insights = build_reasoning_context(
                self.df, query, query_type, schema
            )

        # ── STEP 1: LLM → CODE ────────────────────
        code_prompt = f"""You are a Python data analyst.

Conversation so far:
{history_block}

Current user query: {query}

Dataset columns: {columns}

Sample (first 5 rows):
{sample}

Stats:
{stats}

Write pandas code to answer the current query.
Use the conversation history only to resolve references like "it", "that column", "the same period".

Rules:
- Use `df` as the dataframe variable.
- Store final output in a variable named `result`.
- `result` must be a DataFrame, Series, or scalar.
- DO NOT modify `df` in-place (no df['col'] = ...). Use a copy or intermediate variable.
- For trend/direction questions, compute the actual values (e.g. monthly totals) as a DataFrame — do NOT return a bare True/False comparison.
- Write flat code — no functions, no classes, no indented blocks where avoidable.
- Return ONLY raw Python code. No JSON. No markdown. No explanation.
- If no computation is needed, return exactly: result = None
- Do NOT use import statements — pd is already available.
"""

        code_response = await self._llm._call([
            {"role": "system", "content": "Return ONLY raw Python code. No JSON. No markdown. No explanation."},
            {"role": "user", "content": code_prompt},
        ])

        logger.info("LLM code response:\n%s", code_response)

        # ── STEP 2: PARSE CODE ────────────────────
        # Strip markdown fences if the LLM wrapped anyway
        code = re.sub(r"```(?:python)?|```", "", code_response).strip()
        # Strip stray import lines the LLM emits despite instructions
        code = re.sub(r"^\s*import\s+pandas\s+as\s+pd\s*$", "", code, flags=re.MULTILINE).strip()

        # ── STEP 2b: EXECUTE (with dedent + retry) ────────────
        result, exec_error = None, None
        if code:
            result, exec_error = await safe_exec(code, self.df, self._llm)

        if result is None:
            return {"answer": "No meaningful result could be computed.", "table": None}

        # ── STEP 3: INTERPRET ─────────────────────
        meta = interpret_result(result)
        logger.info("Result metadata: %s", meta)

        # ── STEP 4: TABLE DECISION ─────────────────
        show_table = should_show_table(meta)
        table = to_table_records(result, meta) if show_table else None

        # ── STEP 5: ANSWER ────────────────────────
        if exec_error:
            answer = "There was an issue processing your query. Please try rephrasing it."
        else:
            answer = deterministic_answer(query, meta)

        # ── STEP 6: LLM NARRATION (grounded fallback) ─
        if answer is None:
            # Inject pre-computed reasoning blocks when available.
            # For lookup queries reasoning_insights is [], so nothing is added.
            reasoning_section = ""
            if not reasoning_insights and query_type in ("trend", "explanatory"):
                logger.warning("No reasoning insights generated — fallback risk")

            if reasoning_insights:
                reasoning_section = (
                    "\n\n🚨 PRIMARY DATA (USE THIS FIRST — DO NOT IGNORE):\n"
                    + json.dumps(reasoning_insights, indent=2)
                )

            guardrail = (
                "\n\n🚨 RULES (STRICT — DO NOT VIOLATE):"

                "\n1. ALWAYS use the PRIMARY DATA (structured analysis) if present."
                "\n2. DO NOT say 'not enough information' if PRIMARY DATA exists."
                "\n3. DO NOT ignore driver_detection, trend, or change_detection blocks."

                "\n\nTREND QUERY RULES:"
                "\n- Use change_detection.direction (up/down/flat) as final answer."
                "\n- Use trend.periods to support with values."

                "\n\nEXPLANATORY QUERY RULES:"
                "\n- MUST use driver_detection."
                "\n- Explain which segments contributed to the change using 'change' values (avoid causal claims)."
                "\n- If driver_detection.drivers is empty, say: 'There is no significant change in the data.'"

                "\n\nCOMPARISON QUERY RULES:"
                "\n- Use segmentation or contribution_pct blocks to compare."
                "\n- Mention top segments explicitly."

                "\n\n🚫 NEVER:"
                "\n- speculate reasons"
                "\n- ignore provided data"
                "\n- give generic answers"

                + "\n\nOUTPUT STYLE RULES:"
                + "\n- Keep answer human-readable and concise"
                + "\n- Avoid large raw numbers (use rounded or approximate values)"
                + "\n- Do NOT dump full calculations"
                + "\n- Prefer insights over exact figures"
                + "\n- Do NOT say 'because'; use 'driven by' or 'main contributors are'"
                + "\n- Avoid 'first half vs second half'; describe trend as early vs recent"
            )

            narration_prompt = f"""You are a data analyst assistant.

Conversation so far:
{history_block}

Current user query: {query}

Computed query result:
{json.dumps(meta, indent=2)}
{reasoning_section}
{guardrail}

Instructions:
- First, give a clear final answer (increase/decrease/flat OR comparison OR explanation)
- Then support it with 1–2 key insights (NOT all values)
- Use approximate numbers when large (e.g., ~13M instead of full numbers)
- Focus on what changed, not raw data dumps
- If PRIMARY DATA contains direction, use it directly
- If driver_detection exists, use it to explain WHY

Write 1–3 concise sentences.

If PRIMARY DATA is empty, ONLY then say:
"The data doesn't show enough information to answer this."

No bullet points. No markdown.
"""
            answer = await self._llm._call([
                {"role": "system", "content": "Answer using only the provided data. Cite specific values. No speculation. No markdown."},
                {"role": "user", "content": narration_prompt},
            ])
            answer = answer.strip()

        logger.info("Table shown: %s | Answer: %s", show_table, answer)

        return {"answer": answer, "table": table}