import asyncio
import logging

logger = logging.getLogger(__name__)

# task_id → asyncio.Task
_tasks: dict[str, asyncio.Task] = {}

# task_id → True when the user has requested cancellation.
# Checked cooperatively inside the pipeline so we can stop AFTER the LLM
# returns, without relying on asyncio.Task.cancel() which cannot interrupt
# an in-flight HTTP request.
_cancel_flags: dict[str, bool] = {}


def create_task(task_id: str, coro) -> asyncio.Task:
    """Register and schedule a coroutine as a tracked asyncio Task."""
    _cancel_flags[task_id] = False
    task = asyncio.create_task(coro)
    _tasks[task_id] = task
    logger.info("Task created: %s", task_id)
    return task


def cancel_task(task_id: str) -> bool:
    """
    Mark a task as cancelled via the cooperative flag.
    The pipeline checks this flag at every stage boundary and raises
    CancelledError itself — no asyncio.Task.cancel() needed.

    Returns True if the task was known and not already cancelled.
    """
    if task_id not in _cancel_flags:
        logger.warning("cancel_task: unknown task_id %s", task_id)
        return False
    if _cancel_flags[task_id]:
        logger.info("cancel_task: %s already flagged", task_id)
        return False
    _cancel_flags[task_id] = True
    logger.info("cancel_task: flagged %s for cancellation", task_id)
    return True


def is_cancelled(task_id: str) -> bool:
    """Pipeline stages call this to check whether they should abort."""
    return _cancel_flags.get(task_id, False)


def get_task(task_id: str) -> asyncio.Task | None:
    """Return the Task object for a given task_id, or None."""
    return _tasks.get(task_id)


def remove_task(task_id: str) -> None:
    """
    Drop a finished task from the registry.
    Call this only after the task's result has been written to _results —
    never eagerly on cancel, because the pipeline may still be finishing
    the LLM call and needs to write 'cancelled' status afterwards.
    """
    _tasks.pop(task_id, None)
    _cancel_flags.pop(task_id, None)