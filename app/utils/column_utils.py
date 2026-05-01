import re
import logging
import pandas as pd
 
logger = logging.getLogger(__name__)
 
# ─────────────────────────────────────────────────────────────────────────────
# SHARED UTILITY  — used by every module in this patch
# ─────────────────────────────────────────────────────────────────────────────
 
# Token patterns that indicate a column is an identifier, not a metric.
# Matched against each whitespace/underscore-split token of the lowercased name.
_ID_TOKENS = frozenset({
    "id", "number", "num", "no",
    "code", "zip", "postal", "phone", "fax",
    "key", "index", "row", "seq", "sequence",
    "line", "item",   # ← ADD THIS
    "ssn", "ein", "guid", "uuid",
})
 
# Column names that are entirely non-metric regardless of token analysis.
_ID_EXACT: frozenset[str] = frozenset({
    "transactionid", "customerid", "orderid", "invoiceid", "employeeid",
    "productid", "userid", "accountid", "rowid", "row id",
})
 
 
def _is_id_col(name: str) -> bool:
    lower = name.strip().lower()

    # Normalize camelCase → split
    lower = re.sub(r'([a-z])([A-Z])', r'\1 \2', name).lower()

    tokens = re.split(r"[_\s]+", lower)

    # Strong rules

    # 1. Ends with id
    if lower.endswith("id"):
        return True

    # 2. Contains ID token
    if any(t in _ID_TOKENS for t in tokens):
        return True

    # 3. Pattern: <entity>number / <entity>code
    if any(t in {"number", "code"} for t in tokens):
        return True

    return False
 
 
def _meaningful_numeric_cols(df: pd.DataFrame) -> list[str]:
    raw = df.select_dtypes(include="number").columns.tolist()
    meaningful = [c for c in raw if not _is_id_col(c)]

    logger.info("Meaningful numeric columns: %s", meaningful)

    return meaningful
