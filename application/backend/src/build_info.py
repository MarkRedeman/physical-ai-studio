# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Resolve the source provenance of the running Studio application."""

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

_SOURCE_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class BuildInfo:
    """Source revision metadata safe to expose in the Studio UI and support reports."""

    git_sha: str | None = None
    git_commit_date: str | None = None
    git_tag: str | None = None
    git_dirty: bool = False


def _run_git(*args: str) -> str | None:
    """Return git command output when this is a usable source checkout."""
    git_path = shutil.which("git")
    if git_path is None:
        return None

    try:
        result = subprocess.run(  # noqa: S603 - arguments are fixed literals from this module
            [git_path, *args],
            cwd=_SOURCE_ROOT,
            capture_output=True,
            check=True,
            text=True,
            timeout=1,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None

    output = result.stdout.strip()
    return output or None


def _environment_value(name: str) -> str | None:
    value = os.environ.get(name, "").strip()
    return value if value and value.lower() != "unknown" else None


def get_build_info() -> BuildInfo:
    """Return injected build metadata, falling back to an editable git checkout.

    Container images do not retain ``.git`` and therefore receive this metadata
    through build arguments. An editable checkout, including one on ``main``,
    instead resolves its current revision directly so support reports remain
    useful without a separate build step.
    """
    git_sha = _environment_value("STUDIO_GIT_SHA") or _run_git("rev-parse", "HEAD")
    git_commit_date = _environment_value("STUDIO_GIT_COMMIT_DATE") or _run_git("show", "-s", "--format=%cI", "HEAD")
    git_tag = _environment_value("STUDIO_GIT_TAG") or _run_git("describe", "--tags", "--exact-match", "HEAD")

    injected_dirty = _environment_value("STUDIO_GIT_DIRTY")
    if injected_dirty is not None:
        git_dirty = injected_dirty.lower() in {"1", "true", "yes"}
    elif _environment_value("STUDIO_GIT_SHA") is None:
        git_dirty = _run_git("status", "--porcelain") is not None
    else:
        git_dirty = False

    return BuildInfo(
        git_sha=git_sha,
        git_commit_date=git_commit_date,
        git_tag=git_tag,
        git_dirty=git_dirty,
    )
