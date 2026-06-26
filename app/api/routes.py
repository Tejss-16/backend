from datetime import datetime, timezone
import asyncio
import io
import json
import logging
import uuid
import traceback
import re as _re
from pydantic import BaseModel

import pandas as pd
from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse
from plotly.utils import PlotlyJSONEncoder

from app.services.chart_service import ChartGenerator
from app.utils.task_manager import create_task, cancel_task, remove_task, is_cancelled
from app.services.chat_service import ChatService
from app.utils.data_store import data_store


router = APIRouter()

# ---------- LOGGER ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------- IN-MEMORY RESULT STORE ----------
# Keyed by task_id -> {"status": "running"|"completed"|"cancelled"|"error", ...}
_results: dict[str, dict] = {}

def _coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include="object").columns:
        cleaned = df[col].astype(str).str.replace(
            r"[₹$€£¥,\s%]", "", regex=True
        )
        parsed = pd.to_numeric(cleaned, errors="coerce")
        non_null = df[col].notna().sum()
        if non_null > 0 and parsed.notna().sum() / non_null >= 0.60:
            df[col] = parsed
            logger.info("Coerced column %r to numeric (dtype=%s)", col, parsed.dtype)
    return df


def _parse_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect and convert date/time columns to datetime64 ONCE at upload time.

    This is the single place where the O(rows × formats) date-parsing work
    happens.  Every downstream consumer (transformer, _apply_date_filter,
    _find_date_column, _is_time) just checks is_datetime64_any_dtype — a
    free dtype flag read — and never re-parses.

    Strategy (same format list as transformer._parse_time_inplace so behaviour
    is identical; just done once instead of once per chart):
      1. Skip columns already parsed as datetime64 by pd.read_csv/read_excel.
      2. For object columns with date-hinted names, try each explicit format.
      3. Fall back to dateutil inference if no format matches well.
      4. Only convert if ≥ 70% of non-null values parse successfully.
    """
    import warnings as _w

    _DATE_HINTS = ("date", "time", "period", "ship", "created", "updated")
    _FORMATS = [
        "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d",
        "%d-%m-%Y", "%m-%d-%Y",
        "%Y-%m-%d %H:%M:%S", "%m/%d/%Y %H:%M:%S", "%d/%m/%Y %H:%M:%S",
        "%b %d, %Y", "%B %d, %Y", "%Y%m%d",
    ]
    _THRESHOLD = 0.70

    for col in df.columns:
        # Already datetime — nothing to do
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            continue

        # Only consider object columns
        if df[col].dtype != object:
            continue

        non_null = df[col].notna().sum()
        if non_null == 0:
            continue

        # Gate 1: column name must hint at a date OR pass a quick probe
        name_lower = col.lower()
        name_hinted = any(h in name_lower for h in _DATE_HINTS)

        # Quick probe: try the most common format first before the full loop
        with _w.catch_warnings():
            _w.simplefilter("ignore", UserWarning)
            probe = pd.to_datetime(df[col], format="%Y-%m-%d", errors="coerce")
        if probe.notna().sum() / non_null >= _THRESHOLD:
            df[col] = probe
            logger.info("Parsed date column %r (format=%%Y-%%m-%%d)", col)
            continue

        # Only run the expensive format loop for name-hinted columns
        if not name_hinted:
            # Last-resort: full dateutil inference only for hinted columns
            continue

        converted = False
        for fmt in _FORMATS[1:]:   # skip "%Y-%m-%d" — already tried above
            try:
                with _w.catch_warnings():
                    _w.simplefilter("ignore", UserWarning)
                    parsed = pd.to_datetime(df[col], format=fmt, errors="coerce")
                if parsed.notna().sum() / non_null >= _THRESHOLD:
                    df[col] = parsed
                    logger.info("Parsed date column %r (format=%s)", col, fmt)
                    converted = True
                    break
            except Exception:
                continue

        if not converted:
            # Dateutil fallback — only for strongly name-hinted columns
            try:
                with _w.catch_warnings():
                    _w.simplefilter("ignore", UserWarning)
                    parsed = pd.to_datetime(df[col], errors="coerce")
                if parsed.notna().sum() / non_null >= _THRESHOLD:
                    df[col] = parsed
                    logger.info("Parsed date column %r (dateutil fallback)", col)
            except Exception:
                pass

    return df

@router.get("/health")
def health_check():
    return {"status": "successful"}

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    filename = file.filename or ""

    try:
        if filename.endswith(".csv"):
            try:
                df = pd.read_csv(io.BytesIO(contents), encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(io.BytesIO(contents), encoding="latin-1")
        elif filename.endswith(".xlsx"):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            return JSONResponse(status_code=400, content={"error": "Unsupported file format"})
    except Exception as e:
        logger.exception("File parsing failed")
        return JSONResponse(status_code=500, content={"error": "Failed to parse file"})

    df = _coerce_numeric_columns(df)
    df = _parse_date_columns(df)
    dataset_id = data_store.save(df)
    logger.info("Dataset uploaded: %s (%d rows, %d cols, numeric: %s)",
                dataset_id, *df.shape, df.select_dtypes(include="number").columns.tolist())
    return {"dataset_id": dataset_id}

@router.post("/start-analysis")
async def start_analysis(
    dataset_id: str = Form(...),
    query: str = Form(...),
):
    task_id = str(uuid.uuid4())
    logger.info("Starting analysis task %s for query: %r", task_id, query[:60])

    df = data_store.get(dataset_id)
    if df is None:
        logger.warning("Invalid dataset_id: %s", dataset_id)
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid or expired dataset_id"},
        )

    _results[task_id] = {"status": "running"}

    async def _run():
        try:
            # Pass dataset_id so ChartGenerator uses the process-level metadata
            # cache — avoids recomputing column classifications, cardinalities,
            # and the LLM profile string on every query against the same dataset.
            generator = ChartGenerator(df, dataset_id=dataset_id)
            result = await generator.generate(query, task_id)

            if is_cancelled(task_id):
                _results[task_id] = {"status": "cancelled"}
                logger.info("Task %s: LLM finished but cancel was requested — discarding result", task_id)
                return

            serialised = json.loads(json.dumps(result, cls=PlotlyJSONEncoder))
            _results[task_id] = {"status": "completed", "data": serialised}
            logger.info("Task %s completed successfully", task_id)

        except asyncio.CancelledError:
            _results[task_id] = {"status": "cancelled"}
            logger.info("Task %s pipeline aborted via cancel flag", task_id)
            raise

        except Exception as exc:
            _results[task_id] = {
                "status": "error",
                "error": str(exc),
                "trace": traceback.format_exc()
            }
            logger.exception("Task %s raised an exception", task_id)

        finally:
            remove_task(task_id)

    create_task(task_id, _run())
    return {"task_id": task_id}


@router.get("/status/{task_id}")
async def get_status(task_id: str):
    result = _results.get(task_id)
    if result is None:
        return JSONResponse(status_code=404, content={"status": "not_found"})
    return JSONResponse(content=result)


@router.post("/cancel/{task_id}")
async def cancel(task_id: str):
    if task_id not in _results:
        return JSONResponse(status_code=404, content={"error": "Unknown task_id"})
    cancelled = cancel_task(task_id)
    return {"cancelled": cancelled}


class ChatRequest(BaseModel):
    dataset_id: str
    query: str
    history: list[dict] = []

@router.post("/chat")
async def chat(req: ChatRequest):
    df = data_store.get(req.dataset_id)
    if df is None:
        return JSONResponse(status_code=400, content={"error": "Invalid dataset_id"})
    service = ChatService(df)
    result = await service.chat(req.query, history=req.history)
    return result
