"""Base class for model import strategies.

Each import strategy handles a specific archive format (e.g., Physical AI Studio
exports, HuggingFace/LeRobot models). The orchestrator (ModelImportService) detects
the archive format and delegates to the appropriate strategy.
"""

import zipfile
from abc import ABC, abstractmethod

from schemas import Model


class ImportStrategy(ABC):
    """Abstract base class that all model import strategies must inherit from."""

    @abstractmethod
    async def import_model(
        self,
        zf: zipfile.ZipFile,
        all_names: list[str],
        name: str,
        project_id: str,
    ) -> Model:
        """Import a model from a zip archive.

        Args:
            zf: Open zip file to read from.
            all_names: Pre-computed list of all entry names in the zip.
            name: Display name for the imported model.
            project_id: Project to associate the model with.

        Returns:
            The created Model record.

        Raises:
            ImportValidationError: On bad archive content or missing data.
            ImportConversionError: On model conversion failure.
            ImportDependencyError: If a required library is unavailable.
        """
