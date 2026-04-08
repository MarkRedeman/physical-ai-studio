from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from schemas import Dataset
from schemas.import_job import DatasetImportJobPayload, DatasetImportSource, DatasetManifestV1, ImportValidationReport

if TYPE_CHECKING:
    from pathlib import Path
    from uuid import UUID


class DatasetImportAdapter(ABC):
    source: DatasetImportSource = DatasetImportSource.UNKNOWN

    @abstractmethod
    def detect(self, archive_path: Path) -> bool:
        """Return True if this adapter can process the archive."""

    @abstractmethod
    def parse_to_draft_manifest(self, archive_path: Path, payload: DatasetImportJobPayload) -> DatasetManifestV1:
        """Parse source archive and build canonical draft manifest."""

    @abstractmethod
    def validate(
        self,
        manifest: DatasetManifestV1,
        payload: DatasetImportJobPayload,
    ) -> ImportValidationReport:
        """Validate draft/finalized payload for this source."""

    @abstractmethod
    async def commit(self, payload: DatasetImportJobPayload, project_id: UUID) -> Dataset:
        """Execute extraction and register dataset in DB."""
