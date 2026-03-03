# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Log streaming API endpoints.

Provides endpoints to discover available log sources and to stream log file
contents in real-time via Server-Sent Events.

Source types:
    - application: The main app log (catch-all for non-worker logs)
    - worker: Per-class worker logs (training, inference, etc.)
    - session: Per-session logs for websocket workers (camera, robot, setup)
    - job: Per-job logs created during training/import/export runs
"""

import asyncio
import os
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from typing import Literal

import anyio
from fastapi import APIRouter, HTTPException, status
from loguru import logger
from sse_starlette import EventSourceResponse, ServerSentEvent

from core.logging.utils import VALID_SESSION_TYPES, get_job_logs_path, get_session_logs_path
from schemas.logs import LogSource
from settings import get_settings

router = APIRouter(prefix="/api/logs", tags=["Logs"])

# ---------------------------------------------------------------------------
# Static source definitions
# ---------------------------------------------------------------------------

# Source id → (display name, log filename relative to log_dir, source type)
_LogSourceType = Literal["application", "worker", "session", "job"]
_STATIC_SOURCES: dict[str, tuple[str, str, _LogSourceType]] = {
    "app": ("Application", "app.log", "application"),
    "training": ("Training", "training.log", "worker"),
    "import_export": ("Import / Export", "import_export.log", "worker"),
    "inference": ("Inference", "inference.log", "worker"),
    "teleoperate": ("Teleoperate", "teleoperate.log", "worker"),
}

# Human-readable labels for session worker types.
_SESSION_TYPE_LABELS: dict[str, str] = {
    "camera": "Camera",
    "robot": "Robot",
    "setup": "Setup",
}


# ---------------------------------------------------------------------------
# Path resolution helpers
# ---------------------------------------------------------------------------


def _get_static_log_path(source_id: str) -> str | None:
    """Resolve a static source id to an absolute log file path."""
    entry = _STATIC_SOURCES.get(source_id)
    if entry is None:
        return None
    settings = get_settings()
    return os.path.join(settings.log_dir, entry[1])


def _resolve_source_path(source_id: str) -> str | None:
    """Resolve any source id to an absolute log file path.

    Returns None if the source id is not recognized.
    """
    # Static source (application + workers)
    path = _get_static_log_path(source_id)
    if path is not None:
        return path

    # Session source (format: "session-{type}-{uuid}")
    if source_id.startswith("session-"):
        parts = source_id.split("-", 2)  # ["session", type, uuid]
        if len(parts) == 3:
            worker_type, worker_id = parts[1], parts[2]
            try:
                return get_session_logs_path(worker_type, worker_id)
            except ValueError:
                return None

    # Job source (format: "job-{type}-{uuid}")
    if source_id.startswith("job-"):
        parts = source_id.split("-", 2)  # ["job", type, uuid]
        if len(parts) == 3:
            job_type, job_id = parts[1], parts[2]
            try:
                return get_job_logs_path(job_id, job_type)
            except ValueError:
                return None

    return None


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------


def _get_file_created_at(path: str) -> datetime | None:
    """Return the creation time of a file as a UTC datetime, or None on error."""
    try:
        stat = os.stat(path)
        # Use birth time if available (Linux 4.11+ with statx), fall back to mtime.
        ts = getattr(stat, "st_birthtime", None) or stat.st_mtime
        return datetime.fromtimestamp(ts, tz=UTC)
    except OSError:
        return None


def _short_id(uuid_str: str) -> str:
    """Return the first 8 characters of a UUID string for display."""
    return uuid_str[:8]


def _discover_session_sources() -> list[LogSource]:
    """List per-session log files on disk and return them as LogSource entries."""
    settings = get_settings()
    sessions_dir = os.path.join(settings.log_dir, "sessions")
    sources: list[LogSource] = []
    if not os.path.isdir(sessions_dir):
        return sources

    for worker_type in sorted(VALID_SESSION_TYPES):
        type_dir = os.path.join(sessions_dir, worker_type)
        if not os.path.isdir(type_dir):
            continue
        label = _SESSION_TYPE_LABELS.get(worker_type, worker_type.title())
        for filename in sorted(os.listdir(type_dir)):
            if not filename.endswith(".log"):
                continue
            worker_id = filename.removesuffix(".log")
            file_path = os.path.join(type_dir, filename)
            sources.append(
                LogSource(
                    id=f"session-{worker_type}-{worker_id}",
                    name=f"{label}: {_short_id(worker_id)}",
                    type="session",
                    created_at=_get_file_created_at(file_path),
                )
            )
    return sources


def _discover_job_sources() -> list[LogSource]:
    """List per-job log files on disk and return them as LogSource entries.

    Job log files are named ``{type}_{job_id}.log`` (e.g. ``training_abc123.log``).
    """
    settings = get_settings()
    jobs_dir = os.path.join(settings.log_dir, "jobs")
    sources: list[LogSource] = []
    if not os.path.isdir(jobs_dir):
        return sources

    # Human-readable labels for job types.
    _JOB_TYPE_LABELS: dict[str, str] = {
        "training": "Train",
        "import": "Import",
        "export": "Export",
    }

    for filename in sorted(os.listdir(jobs_dir)):
        if not filename.endswith(".log"):
            continue
        file_path = os.path.join(jobs_dir, filename)
        created_at = _get_file_created_at(file_path)
        stem = filename.removesuffix(".log")
        # Expected format: {type}_{uuid}
        sep_idx = stem.find("_")
        if sep_idx == -1:
            # Legacy filename without type prefix — fall back gracefully
            job_id = stem
            sources.append(
                LogSource(
                    id=f"job-unknown-{job_id}",
                    name=f"Job: {_short_id(job_id)}",
                    type="job",
                    created_at=created_at,
                )
            )
            continue
        job_type = stem[:sep_idx]
        job_id = stem[sep_idx + 1 :]
        label = _JOB_TYPE_LABELS.get(job_type, job_type.title())
        sources.append(
            LogSource(
                id=f"job-{job_type}-{job_id}",
                name=f"{label}: {_short_id(job_id)}",
                type="job",
                created_at=created_at,
            )
        )
    return sources


# ---------------------------------------------------------------------------
# SSE streaming
# ---------------------------------------------------------------------------


async def _tail_log_file(path: str) -> AsyncGenerator[ServerSentEvent]:
    """Async generator that live-tails a log file and yields SSE events.

    Reads existing content first, then polls for new lines every 500ms.
    The generator runs until the client disconnects (the SSE connection is
    closed), which causes the framework to cancel this coroutine via
    GeneratorExit / CancelledError.
    """
    try:
        async with await anyio.open_file(path, encoding="utf-8") as f:
            while True:
                line = await f.readline()
                if not line:
                    await asyncio.sleep(0.5)
                    continue
                yield ServerSentEvent(data=line.rstrip())
    except asyncio.CancelledError:
        logger.debug(f"SSE log stream cancelled for {path}")
    except GeneratorExit:
        logger.debug(f"SSE log stream closed for {path}")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/sources")
async def get_log_sources() -> list[LogSource]:
    """Return all available log sources.

    Sources are grouped by type: application, worker, session, job.
    """
    sources: list[LogSource] = []

    # Static sources (application + workers)
    for source_id, (name, _filename, source_type) in _STATIC_SOURCES.items():
        sources.append(LogSource(id=source_id, name=name, type=source_type))

    # Dynamic: per-session logs
    sources.extend(_discover_session_sources())

    # Dynamic: per-job logs
    sources.extend(_discover_job_sources())

    return sources


@router.get("/{source_id}/stream")
async def stream_logs(source_id: str) -> EventSourceResponse:
    """Stream log lines from the given source via Server-Sent Events.

    The connection stays open and new lines are pushed as they are written
    to the log file. Close the connection from the client side to stop.
    """
    path = _resolve_source_path(source_id)
    if path is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown log source: {source_id}",
        )
    if not await anyio.Path(path).exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Log file not found for source: {source_id}",
        )
    return EventSourceResponse(_tail_log_file(path))
