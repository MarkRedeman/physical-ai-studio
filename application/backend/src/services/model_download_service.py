# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import re
import tempfile
import zipfile
from pathlib import Path
from uuid import uuid4

from models.utils import load_policy
from physicalai.export import ExportablePolicyMixin
from physicalai.export import ExportBackend, get_available_backends
from schemas import Model

SNAPSHOT_DIR_PATTERN = re.compile(r"^snapshot_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")


class ModelDownloadService:
    @staticmethod
    def _validate_backend(backend: str) -> str:
        """Validate backend and return normalized value."""
        try:
            return ExportBackend(backend).value
        except ValueError as error:
            available = ", ".join(get_available_backends())
            raise ValueError(f"Unsupported backend '{backend}'. Supported backends: {available}") from error

    @staticmethod
    def _is_snapshot_path(relative_path: Path) -> bool:
        """Check if a path is inside a snapshot directory."""
        top_level = relative_path.parts[0] if relative_path.parts else ""
        return bool(SNAPSHOT_DIR_PATTERN.match(top_level))

    @staticmethod
    def _zip_directory(directory_path: Path, *, include_snapshot: bool = False) -> Path:
        """Zip a directory and preserve relative paths within it."""
        temporary_archive_path = Path(tempfile.gettempdir()) / f"model-{uuid4()}.zip"
        with zipfile.ZipFile(temporary_archive_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in directory_path.rglob("*"):
                if not path.is_file():
                    continue

                relative = path.relative_to(directory_path)

                if not include_snapshot and ModelDownloadService._is_snapshot_path(relative):
                    continue

                archive.write(path, arcname=relative)

        return temporary_archive_path

    def _ensure_export_exists(self, model: Model, model_path: Path, backend: str) -> Path:
        """Ensure backend export exists on disk, creating it if needed."""
        export_dir = model_path / "exports" / backend
        has_export = export_dir.exists() and any(export_dir.iterdir())
        if has_export:
            return export_dir

        export_dir.mkdir(parents=True, exist_ok=True)
        policy = load_policy(model)
        if not isinstance(policy, ExportablePolicyMixin):
            raise TypeError(f"Policy '{model.policy}' does not support export")
        policy.export(export_dir, backend=backend)
        return export_dir

    def _get_export_dir(self, model: Model, model_path: Path, backend: str | None) -> Path:
        """Resolve which directory should be archived."""
        if backend is None:
            return model_path

        normalized_backend = self._validate_backend(backend)
        return self._ensure_export_exists(model, model_path, normalized_backend)

    def create_model_archive(self, model: Model, *, include_snapshot: bool = False, backend: str | None = None) -> Path:
        """Create a zip archive of a model folder and return the archive path.

        :param model: Model schema containing model metadata and path.
        :param include_snapshot: When False (default), files inside snapshot_*
            directories are excluded from the archive.
        :param backend: Optional export backend. When set, only files for this
            backend are archived; an export is created first if missing.
        :return: Path to the temporary zip archive.
        """
        model_path = Path(model.path).resolve()
        if not model_path.exists() or not model_path.is_dir():
            raise FileNotFoundError(f"Model path not found or not a directory: {model_path}")

        export_dir = self._get_export_dir(model, model_path, backend)
        return self._zip_directory(export_dir, include_snapshot=include_snapshot if backend is None else False)
