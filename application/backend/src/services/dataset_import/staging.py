# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Staging-path helpers for dataset import archives.

An *opaque staging identifier* (a UUID string) is persisted in the job
payload instead of the raw filesystem path.  The actual path is derived
deterministically from the id at runtime so no in-memory map is needed.

Layout::

    <cache_dir>/imports/datasets/<staging_id>.zip
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING
from uuid import uuid4

from settings import get_settings

if TYPE_CHECKING:
    from pathlib import Path

    from schemas.import_job import DatasetImportJobPayload

# Only allow plain UUID strings (hex digits + hyphens) to prevent path traversal.
_STAGING_ID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE)


def generate_staging_id() -> str:
    """Return a new random staging identifier (UUID v4 string)."""
    return str(uuid4())


def staging_path_for_id(staging_id: str) -> Path:
    """Return the absolute staging archive path for *staging_id*.

    Only computes and validates the path - does NOT create directories.
    Directory creation is the responsibility of the API upload path.

    Raises ``ValueError`` if *staging_id* is not a valid UUID string to
    prevent path-traversal attacks.
    """
    if not _STAGING_ID_RE.fullmatch(staging_id):
        raise ValueError(f"Invalid staging_id '{staging_id}': must be a UUID string")
    settings = get_settings()
    staging_dir = settings.cache_dir / "imports" / "datasets"
    return staging_dir / f"{staging_id}.zip"


def resolve_payload_archive_path(payload: DatasetImportJobPayload) -> Path | None:
    """Return the archive ``Path`` for a dataset import job payload.

    Derives the path deterministically from ``payload.archive_staging_id``.
    Returns ``None`` when the field is absent or empty.
    """
    staging_id: str | None = payload.archive_staging_id
    if staging_id:
        return staging_path_for_id(staging_id)

    return None
