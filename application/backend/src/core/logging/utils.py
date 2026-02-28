# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager, contextmanager
from uuid import UUID

from loguru import logger

from core.logging.setup import global_log_config


def _validate_uuid(value: str | UUID) -> str | UUID:
    """Validate that a value is a valid UUID (prevents path traversal).

    Args:
        value: The identifier to validate

    Returns:
        Validated value

    Raises:
        ValueError: If value is not a valid UUID
    """
    try:
        UUID(str(value))
    except ValueError as e:
        raise ValueError(
            f"Invalid id '{value}'. Only valid UUIDs are allowed.",
        ) from e
    return value


# Keep the old name as an alias so existing call-sites don't break.
_validate_job_id = _validate_uuid


# ---------------------------------------------------------------------------
# Job logs  (logs/jobs/{job_id}.log)
# ---------------------------------------------------------------------------


def get_job_logs_path(job_id: str | UUID) -> str:
    """Get the path to the log file for a specific job.

    Args:
        job_id: Unique identifier for the job

    Returns:
        str: Path to the job's log file (e.g. logs/jobs/{job_id}.log)

    Raises:
        ValueError: If job_id contains invalid characters
    """
    job_id = _validate_uuid(job_id)
    jobs_folder = os.path.join(global_log_config.log_folder, "jobs")
    try:
        os.makedirs(jobs_folder, exist_ok=True)
    except OSError as e:
        raise RuntimeError(f"Failed to create jobs log directory: {e}") from e
    return os.path.join(jobs_folder, f"{job_id}.log")


@contextmanager
def job_logging_ctx(job_id: str | UUID) -> Generator[str]:
    """Add a temporary log sink for a specific job.

    Captures all logs emitted during the context to logs/jobs/{job_id}.log.
    The sink is automatically removed on exit, but the log file persists.
    Logs also continue to go to other configured sinks.

    Args:
        job_id: Unique identifier for the job, used as the log filename

    Yields:
        str: Path to the created log file (e.g. logs/jobs/{job_id}.log)

    Raises:
        ValueError: If job_id contains invalid characters
        RuntimeError: If log directory creation or sink addition fails
    """
    job_id = _validate_uuid(job_id)

    log_file = get_job_logs_path(job_id)

    try:
        sink_id = logger.add(
            log_file,
            rotation=global_log_config.rotation,
            retention=global_log_config.retention,
            level=global_log_config.level,
            serialize=global_log_config.serialize,
            enqueue=True,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to add log sink for job {job_id}: {e}") from e

    try:
        logger.info(f"Started logging to {log_file}")
        yield log_file
    finally:
        logger.info(f"Stopped logging to {log_file}")
        logger.remove(sink_id)


# ---------------------------------------------------------------------------
# Session logs  (logs/sessions/{worker_type}/{worker_id}.log)
# ---------------------------------------------------------------------------

# Allowed worker type prefixes — keeps the directory structure predictable.
VALID_SESSION_TYPES = frozenset({"camera", "robot", "setup"})


def get_session_logs_path(worker_type: str, worker_id: str | UUID) -> str:
    """Get the path to the log file for a worker session.

    Args:
        worker_type: Short label for the worker kind (e.g. "camera", "robot", "setup").
        worker_id: Unique identifier for this session (typically a UUID).

    Returns:
        Absolute path to the session log file.

    Raises:
        ValueError: If worker_type is unknown or worker_id is not a valid UUID.
    """
    if worker_type not in VALID_SESSION_TYPES:
        raise ValueError(f"Unknown worker type '{worker_type}'. Must be one of {sorted(VALID_SESSION_TYPES)}")
    worker_id = _validate_uuid(worker_id)
    sessions_folder = os.path.join(global_log_config.log_folder, "sessions", worker_type)
    try:
        os.makedirs(sessions_folder, exist_ok=True)
    except OSError as e:
        raise RuntimeError(f"Failed to create session log directory: {e}") from e
    return os.path.join(sessions_folder, f"{worker_id}.log")


@asynccontextmanager
async def session_logging_ctx(worker_type: str, worker_id: str | UUID) -> AsyncGenerator[str]:
    """Add a temporary log sink for a websocket worker session.

    Creates a per-session log file at logs/sessions/{worker_type}/{worker_id}.log
    and sets ``logger.contextualize(worker=...)`` so the worker class-level filter
    also captures these messages.

    The sink is removed when the context exits, but the log file persists for
    later viewing through the log viewer UI.

    Args:
        worker_type: Short label (e.g. "camera", "robot", "setup").
        worker_id: Unique session identifier (UUID).

    Yields:
        Absolute path to the created log file.
    """
    log_file = get_session_logs_path(worker_type, worker_id)

    try:
        sink_id = logger.add(
            log_file,
            rotation=global_log_config.rotation,
            retention=global_log_config.retention,
            level=global_log_config.level,
            serialize=global_log_config.serialize,
            enqueue=True,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to add log sink for {worker_type} session {worker_id}: {e}") from e

    try:
        logger.info(f"Session started — logging to {log_file}")
        yield log_file
    finally:
        logger.info(f"Session ended — logging to {log_file}")
        logger.remove(sink_id)
