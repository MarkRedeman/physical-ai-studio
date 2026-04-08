from __future__ import annotations

import json
import zipfile
from pathlib import Path
from uuid import UUID, uuid4

from loguru import logger

from schemas import Dataset
from schemas.import_job import (
    DatasetImportJobPayload,
    DatasetImportSource,
    DatasetManifestCapture,
    DatasetManifestIdentity,
    DatasetManifestSource,
    DatasetManifestV1,
    ImportValidationIssue,
    ImportValidationReport,
)
from services.dataset_service import DatasetService
from settings import get_settings

from .dataset_base_adapter import DatasetImportAdapter


class StudioAdapter(DatasetImportAdapter):
    source = DatasetImportSource.STUDIO

    def detect(self, archive_path: Path) -> bool:
        try:
            with zipfile.ZipFile(archive_path, "r") as zf:
                names = [name.strip("/") for name in zf.namelist()]

            def _normalize(name: str) -> str:
                return name[2:] if name.startswith("./") else name

            normalized = [_normalize(name) for name in names]
            has_info = any(name.endswith("meta/info.json") or name == "meta/info.json" for name in normalized)
            has_data = any(
                name.startswith("data/") or "/data/" in name or name.endswith(".parquet") for name in normalized
            )
            return has_info and has_data
        except zipfile.BadZipFile:
            return False

    def parse_to_draft_manifest(self, archive_path: Path, payload: DatasetImportJobPayload) -> DatasetManifestV1:  # noqa: ARG002
        original_dataset_uuid = None
        source_format_version = None
        source_identifier = archive_path.name

        try:
            with zipfile.ZipFile(archive_path, "r") as zf:
                names = set(zf.namelist())
                for manifest_name in ("studio-dataset-manifest.json", "geti-dataset-manifest.json"):
                    if manifest_name in names:
                        with zf.open(manifest_name) as manifest_file:
                            manifest_raw = json.load(manifest_file)
                        source_identifier = manifest_raw.get("source_identifier", source_identifier)
                        source_format_version = manifest_raw.get("manifest_version")
                        original_uuid_raw = manifest_raw.get("original_dataset_uuid")
                        if original_uuid_raw:
                            try:
                                original_dataset_uuid = UUID(str(original_uuid_raw))
                            except (TypeError, ValueError):
                                original_dataset_uuid = None
                        break
        except Exception as e:
            logger.exception(e)

        logger.info(
            "StudioAdapter manifest draft parsed: archive='{}', source_identifier='{}', source_format_version='{}', original_dataset_uuid='{}'",  # noqa: E501
            archive_path,
            source_identifier,
            source_format_version,
            original_dataset_uuid,
        )

        return DatasetManifestV1(
            source=DatasetManifestSource(
                source_type=DatasetImportSource.STUDIO,
                adapter=self.__class__.__name__,
                source_format_version=source_format_version,
                source_identifier=source_identifier,
                original_dataset_uuid=original_dataset_uuid,
            ),
            identity=DatasetManifestIdentity(suggested_name=archive_path.stem),
            capture=DatasetManifestCapture(),
            missing_fields=["environment_id"],
        )

    def validate(self, manifest: DatasetManifestV1, payload: DatasetImportJobPayload) -> ImportValidationReport:  # noqa: ARG002
        report = ImportValidationReport(is_valid=True)

        if payload.finalize_input is None:
            report.is_valid = False
            report.required_user_inputs = ["dataset_name", "environment_id"]
            report.blocking_errors.append(
                ImportValidationIssue(code="finalize_input_missing", message="Finalize input is required before commit")
            )

        if not payload.uploaded_archive_path:
            report.is_valid = False
            report.blocking_errors.append(
                ImportValidationIssue(code="archive_missing", message="No archive path associated with import job")
            )

        return report

    async def commit(self, payload: DatasetImportJobPayload, project_id: UUID) -> Dataset:
        if payload.finalize_input is None:
            raise ValueError("Cannot commit dataset import without finalize input")

        settings = get_settings()
        settings.datasets_dir.mkdir(parents=True, exist_ok=True)

        dataset_id = uuid4()
        destination_dir = settings.datasets_dir / str(dataset_id)
        destination_dir.mkdir(parents=True, exist_ok=False)
        logger.info(
            "StudioAdapter commit destination prepared: dataset_id='{}', destination_dir='{}'",
            dataset_id,
            destination_dir,
        )

        archive_path = Path(payload.uploaded_archive_path)
        extracted_files: list[str] = []
        with zipfile.ZipFile(archive_path, "r") as zf:
            for member in zf.infolist():
                member_path = Path(member.filename)
                if member_path.is_absolute() or ".." in member_path.parts:
                    raise ValueError(f"Unsafe ZIP entry detected: {member.filename}")

            extracted_files = [member.filename for member in zf.infolist() if not member.is_dir()]
            zf.extractall(destination_dir)

        logger.info(
            "StudioAdapter extracted archive: archive='{}', destination_dir='{}', file_count={}",
            archive_path,
            destination_dir,
            len(extracted_files),
        )
        for extracted_file in extracted_files:
            logger.debug("StudioAdapter extracted file: '{}'", extracted_file)

        dataset = Dataset(
            id=dataset_id,
            name=payload.finalize_input.dataset_name,
            path=str(destination_dir),
            default_task=payload.finalize_input.default_task or "",
            project_id=project_id,
            environment_id=payload.finalize_input.environment_id,
        )

        saved = await DatasetService.create_dataset(dataset)
        logger.info(
            "StudioAdapter dataset persisted: dataset_id='{}', project_id='{}', environment_id='{}', path='{}'",
            saved.id,
            saved.project_id,
            saved.environment_id,
            saved.path,
        )

        try:
            archive_path.unlink(missing_ok=True)
            logger.info("StudioAdapter removed staged archive: '{}'", archive_path)
        except Exception:
            logger.warning("StudioAdapter could not remove staged archive: '{}'", archive_path)

        return saved


class LeRobotV2Adapter(DatasetImportAdapter):
    source = DatasetImportSource.LEROBOT_V2

    def detect(self, archive_path: Path) -> bool:  # noqa: ARG002
        return False

    def parse_to_draft_manifest(self, archive_path: Path, payload: DatasetImportJobPayload) -> DatasetManifestV1:
        raise NotImplementedError

    def validate(self, manifest: DatasetManifestV1, payload: DatasetImportJobPayload) -> ImportValidationReport:
        raise NotImplementedError

    async def commit(self, payload: DatasetImportJobPayload, project_id: UUID) -> Dataset:
        raise NotImplementedError


class LeRobotV3Adapter(DatasetImportAdapter):
    source = DatasetImportSource.LEROBOT_V3

    def detect(self, archive_path: Path) -> bool:  # noqa: ARG002
        return False

    def parse_to_draft_manifest(self, archive_path: Path, payload: DatasetImportJobPayload) -> DatasetManifestV1:
        raise NotImplementedError

    def validate(self, manifest: DatasetManifestV1, payload: DatasetImportJobPayload) -> ImportValidationReport:
        raise NotImplementedError

    async def commit(self, payload: DatasetImportJobPayload, project_id: UUID) -> Dataset:
        raise NotImplementedError


class TrossenSDKAdapter(DatasetImportAdapter):
    source = DatasetImportSource.TROSSEN_SDK

    def detect(self, archive_path: Path) -> bool:  # noqa: ARG002
        return False

    def parse_to_draft_manifest(self, archive_path: Path, payload: DatasetImportJobPayload) -> DatasetManifestV1:
        raise NotImplementedError

    def validate(self, manifest: DatasetManifestV1, payload: DatasetImportJobPayload) -> ImportValidationReport:
        raise NotImplementedError

    async def commit(self, payload: DatasetImportJobPayload, project_id: UUID) -> Dataset:
        raise NotImplementedError
