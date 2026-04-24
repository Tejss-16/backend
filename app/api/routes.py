from datetime import datetime, timezone
import asyncio
import io
import json
import logging
import uuid
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
# Keyed by task_id → {"status": "running"|"completed"|"cancelled"|"error", ...}
_results: dict[str, dict] = {}

@router.get("/")
def root():
    return {"message": "API running"}
# ─────────────────────────────────────────────
# HEALTH
# ─────────────────────────────────────────────
@router.get("/healthz")
def health_check():
    return {"status": "ok"}

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload dataset ONLY (no processing)
    Returns dataset_id
    """

    contents = await file.read()
    filename = file.filename or ""

    if filename.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(contents))
    elif filename.endswith(".xlsx"):
        df = pd.read_excel(io.BytesIO(contents))
    else:
        return JSONResponse(
            status_code=400,
            content={"error": "Unsupported file format"},
        )

    dataset_id = data_store.save(df)

    logger.info("Dataset uploaded: %s", dataset_id)

    return {"dataset_id": dataset_id}

# ─────────────────────────────────────────────
# START ANALYSIS  (replaces the old /generate-code)
# ─────────────────────────────────────────────
@router.post("/start-analysis")
async def start_analysis(
    dataset_id: str = Form(...),
    query: str = Form(...),
):
    """
    Accepts dataset_id + query kicks off a background task, and immediately
    returns a task_id.  The client polls /status/{task_id} to track progress
    and may call /cancel/{task_id} at any time.
    """
    task_id = str(uuid.uuid4())
    logger.info("Starting analysis task %s for query: %r", task_id, query[:60])

    # # Read the file eagerly — the UploadFile object is not safe to pass into
    # # a background task because the request lifecycle may close it first.
    # contents = await file.read()
    # filename = file.filename or ""

    # if filename.endswith(".csv"):
    #     df = pd.read_csv(io.BytesIO(contents))
    # elif filename.endswith(".xlsx"):
    #     df = pd.read_excel(io.BytesIO(contents))
    # else:
    #     return JSONResponse(
    #         status_code=400,
    #         content={"error": "Unsupported file format. Upload a .csv or .xlsx file."},
    #     )
    
    # dataset_id = data_store.save(df)

    df = data_store.get(dataset_id)

    if df is None:
        logger.warning("Invalid dataset_id: %s", dataset_id)
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid or expired dataset_id"},
        )

    if df is None:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid dataset_id"},
        )

    # Mark as running immediately so the frontend sees a valid status on its
    # first poll even before the coroutine is scheduled.
    _results[task_id] = {"status": "running"}

    async def _run():
        try:
            generator = ChartGenerator(df)
            result = await generator.generate(query, task_id)

            # Check the flag one final time before writing 'completed' —
            # the LLM may have finished while a cancel was in-flight.
            if is_cancelled(task_id):
                _results[task_id] = {"status": "cancelled"}
                logger.info("Task %s: LLM finished but cancel was requested — discarding result", task_id)
                return

            serialised = json.loads(json.dumps(result, cls=PlotlyJSONEncoder))
            _results[task_id] = {"status": "completed", "data": serialised}
            logger.info("Task %s completed successfully", task_id)

        except asyncio.CancelledError:
            # Raised by cooperative check inside the pipeline (not task.cancel()).
            _results[task_id] = {"status": "cancelled"}
            logger.info("Task %s pipeline aborted via cancel flag", task_id)
            raise

        except Exception as exc:
            _results[task_id] = {"status": "error", "error": str(exc)}
            logger.exception("Task %s raised an exception", task_id)

        finally:
            remove_task(task_id)

    create_task(task_id, _run())

    return {"task_id": task_id }


# ─────────────────────────────────────────────
# STATUS POLL
# ─────────────────────────────────────────────
@router.get("/status/{task_id}")
async def get_status(task_id: str):
    """
    Returns the current state of the task.

    Possible responses
    ------------------
    {"status": "running"}
    {"status": "completed", "data": { ... }}
    {"status": "cancelled"}
    {"status": "error", "error": "..."}
    {"status": "not_found"}   ← unknown task_id
    """
    result = _results.get(task_id)
    if result is None:
        return JSONResponse(status_code=404, content={"status": "not_found"})
    return JSONResponse(content=result)


# ─────────────────────────────────────────────
# CANCEL
# ─────────────────────────────────────────────
@router.post("/cancel/{task_id}")
async def cancel(task_id: str):
    """
    Requests cancellation of a running task.
    If the task is already done the call is a no-op (cancelled: false).
    """
    if task_id not in _results:
        return JSONResponse(status_code=404, content={"error": "Unknown task_id"})

    cancelled = cancel_task(task_id)
    return {"cancelled": cancelled}

# ─────────────────────────────────────────────
# Chat
# ─────────────────────────────────────────────
class ChatRequest(BaseModel):
    dataset_id: str
    query: str
    history: list[dict] = []   # [{role: "user"|"assistant", content: str|dict}]

@router.post("/chat")
async def chat(req: ChatRequest):
    df = data_store.get(req.dataset_id)

    if df is None:
        return JSONResponse(status_code=400, content={"error": "Invalid dataset_id"})

    service = ChatService(df)
    result = await service.chat(req.query, history=req.history)

    return result