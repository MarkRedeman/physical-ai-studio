from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from schemas import Dataset
from schemas.import_job import DatasetImportJobPayload, DatasetImportSource, DatasetManifest, ImportValidationReport

if TYPE_CHECKING:
    from uuid import UUID

    from services.archive_safety import SafeZipArchive


class DatasetImportAdapter(ABC):
    source: DatasetImportSource = DatasetImportSource.UNKNOWN

    @abstractmethod
    def detect(self, archive: SafeZipArchive) -> bool:
        """Return True if this adapter can process the archive."""

    @abstractmethod
    def parse_to_draft_manifest(self, archive: SafeZipArchive, payload: DatasetImportJobPayload) -> DatasetManifest:
        """Parse source archive and build canonical draft manifest."""

    @abstractmethod
    def validate_pre_finalize(
        self, manifest: DatasetManifest, payload: DatasetImportJobPayload
    ) -> ImportValidationReport:
        """Validate the draft manifest before presenting it to the user for finalization.

        At this stage ``payload.finalize_input`` is not yet available, so
        implementations must not require it.  Return a valid default report when
        there is nothing to reject.
        """

    @abstractmethod
    def validate_pre_commit(self, payload: DatasetImportJobPayload) -> ImportValidationReport:
        """Validate the finalized payload immediately before committing.

        Implementations must enforce that ``finalize_input`` and
        ``archive_staging_id`` are present and consistent.
        """

    @abstractmethod
    async def commit(self, payload: DatasetImportJobPayload, project_id: UUID, archive: SafeZipArchive) -> Dataset:
        """Execute extraction and register dataset in DB."""
