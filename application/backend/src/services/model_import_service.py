"""Orchestrator service for importing models from zip archives.

Detects the archive format and delegates to the appropriate import strategy:
- Physical AI Studio exports -> PhysicalAIStudioExportImportStrategy
- HuggingFace/LeRobot models -> HuggingFaceImportStrategy
"""

import io
import logging
import zipfile
from pathlib import Path

from exceptions import ImportValidationError
from schemas import Model
from services.dataset_service import DatasetService
from services.import_strategies import (
    HuggingFaceImportStrategy,
    PhysicalAIStudioExportImportStrategy,
    SafeImportStrategy,
)
from services.import_strategies.base import ImportStrategy
from services.model_service import ModelService
from services.snapshot_service import SnapshotService
from settings import Settings, get_settings

logger = logging.getLogger(__name__)


class ModelImportService:
    """Orchestrates model import by detecting archive format and delegating to strategies."""

    def __init__(
        self,
        model_service: ModelService,
        dataset_service: DatasetService,
        snapshot_service: SnapshotService,
        settings: Settings | None = None,
    ) -> None:
        resolved_settings = settings or get_settings()

        self._strategies: dict[str, ImportStrategy] = {
            "export": SafeImportStrategy(
                PhysicalAIStudioExportImportStrategy(
                    model_service=model_service,
                    dataset_service=dataset_service,
                    snapshot_service=snapshot_service,
                    settings=resolved_settings,
                )
            ),
            "huggingface": SafeImportStrategy(
                HuggingFaceImportStrategy(
                    model_service=model_service,
                    settings=resolved_settings,
                )
            ),
        }

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def import_model(self, file_content: bytes, filename: str, name: str, project_id: str) -> Model:
        """Import a model from raw zip bytes.

        Detects the archive format and delegates to the appropriate strategy.

        Args:
            file_content: Raw bytes of the uploaded zip file.
            filename: Original filename (used for .zip validation).
            name: Display name for the imported model.
            project_id: Project to associate the model with.

        Returns:
            The created Model record.

        Raises:
            ImportValidationError: On bad archive, unrecognized format, or missing data.
            ImportConversionError: On HuggingFace conversion failure.
            ImportDependencyError: If physicalai-train is unavailable.
        """
        self._validate_filename(filename)
        zf = self._open_zip(file_content)

        with zf:
            all_names = zf.namelist()
            if not all_names:
                raise ImportValidationError("The zip archive is empty.")

            archive_format = self._detect_archive_format(all_names)

            if archive_format not in self._strategies:
                self._raise_unknown_format_error(all_names)

            strategy = self._strategies[archive_format]
            return await strategy.import_model(zf, all_names, name, project_id)

    # ------------------------------------------------------------------
    # Format detection
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_filename(filename: str | None) -> None:
        if not filename or not filename.endswith(".zip"):
            raise ImportValidationError("Uploaded file must be a .zip archive.")

    @staticmethod
    def _open_zip(content: bytes) -> zipfile.ZipFile:
        try:
            return zipfile.ZipFile(io.BytesIO(content))
        except zipfile.BadZipFile:
            raise ImportValidationError("Uploaded file is not a valid zip archive.")

    @staticmethod
    def _detect_archive_format(all_names: list[str]) -> str:
        """Detect whether a zip contains a Studio export or a HuggingFace model.

        Returns "export", "huggingface", or "unknown".
        """
        basenames: set[str] = set()
        has_exports_metadata = False

        for name in all_names:
            parts = Path(name).parts
            if name.endswith("metadata.yaml") and "exports" in parts:
                has_exports_metadata = True

            if len(parts) == 1:
                basenames.add(parts[0])
            elif len(parts) == 2:
                basenames.add(parts[1])

        if has_exports_metadata:
            return "export"

        if "config.json" in basenames and "model.safetensors" in basenames:
            return "huggingface"

        return "unknown"

    @staticmethod
    def _raise_unknown_format_error(all_names: list[str]) -> None:
        """Raise an ImportValidationError with a helpful message about the unrecognized format."""
        basenames: set[str] = set()
        for n in all_names:
            parts = Path(n).parts
            basenames.add(parts[-1] if parts else n)

        if "config.json" in basenames and "model.safetensors" not in basenames:
            raise ImportValidationError(
                "Found config.json but model.safetensors is missing. "
                "A HuggingFace model archive must contain both config.json "
                "and model.safetensors."
            )

        raise ImportValidationError(
            "Unrecognized model archive format. Expected either a "
            "Physical AI Studio export (containing exports/*/metadata.yaml) "
            "or a HuggingFace model (containing config.json and model.safetensors)."
        )
