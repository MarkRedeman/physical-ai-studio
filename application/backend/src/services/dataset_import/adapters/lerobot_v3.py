from __future__ import annotations

import shutil
from io import BytesIO
from pathlib import PurePath
from uuid import UUID, uuid4

import pandas as pd
from loguru import logger

from exceptions import InvalidArchiveError
from schemas import Dataset
from schemas.dataset_import_job import (
    DatasetImportJobPayload,
    DatasetImportSource,
    DatasetManifest,
    DatasetManifestStatistics,
    ImportValidationReport,
)
from services.archive_safety import SafeZipArchive, check_disk_headroom, flatten_single_root_directory
from services.dataset_service import DatasetService
from settings import get_settings

from .base import DatasetImportAdapter
from .recording_schema import extract_recording_schema


class LeRobotV3Adapter(DatasetImportAdapter):
    source = DatasetImportSource.LEROBOT_V3

    def _load_info(self, archive: SafeZipArchive, report: ImportValidationReport) -> dict:
        raw_info = archive.read_json("meta/info.json")
        if raw_info is None:
            report.add_error("Could not read required metadata file 'meta/info.json'.")
            return {}
        return raw_info

    def _load_episode_counts(self, archive: SafeZipArchive, report: ImportValidationReport) -> tuple[int, int]:
        episode_count = 0
        frame_count = 0

        episodes_bytes = archive.read_bytes("meta/episodes/chunk-000/file-000.parquet")
        if episodes_bytes is None:
            report.add_error("No episode parquet found under 'meta/episodes/chunk-000/file-000.parquet'.")
            return episode_count, frame_count

        episodes_df = pd.read_parquet(BytesIO(episodes_bytes))
        episode_count = len(episodes_df)
        if "length" in episodes_df.columns:
            frame_count = int(episodes_df["length"].fillna(0).sum())
        else:
            report.add_warning("Episode parquet is missing 'length' column; frame count may be incomplete.")

        return episode_count, frame_count

    def detect(self, archive: SafeZipArchive) -> bool:
        """Return True only for LeRobot v3 archives.

        v3 markers (all required):
          - ``meta/info.json``
          - ``meta/tasks.parquet``         (v3 task index; v2 uses tasks.jsonl)
          - ``data/chunk-*/file-*.parquet`` (v3 data layout; v2 uses episode_*.parquet)

        Explicitly reject if v2-specific markers are found:
          - ``meta/episodes.jsonl``  (v2 only)
          - ``meta/tasks.jsonl``     (v2 only)
        """
        has_info = False
        has_tasks_parquet = False
        has_v3_data = False

        for name in archive.iter_normalized_names():
            # Reject v2-only markers immediately (handle both flat and nested archives)
            if name == "meta/episodes.jsonl" or name.endswith("/meta/episodes.jsonl"):
                return False
            if name == "meta/tasks.jsonl" or name.endswith("/meta/tasks.jsonl"):
                return False

            if not has_info and (name == "meta/info.json" or name.endswith("/meta/info.json")):
                has_info = True
            if not has_tasks_parquet and (name == "meta/tasks.parquet" or name.endswith("/meta/tasks.parquet")):
                has_tasks_parquet = True
            if (
                not has_v3_data
                and ("data/" in name or name.startswith("data/"))
                and "/file-" in name
                and name.endswith(".parquet")
                and "episode_" not in name
            ):
                has_v3_data = True

            if has_info and has_tasks_parquet and has_v3_data:
                return True

        return False

    def build_draft(
        self,
        archive: SafeZipArchive,
        payload: DatasetImportJobPayload,
    ) -> tuple[DatasetManifest, ImportValidationReport]:
        report = ImportValidationReport()
        info: dict = {}
        episode_count = 0
        frame_count = 0

        try:
            info = self._load_info(archive=archive, report=report)
            episode_count, frame_count = self._load_episode_counts(archive=archive, report=report)

            if archive.read_json("meta/stats.json") is None:
                report.add_warning("No global stats metadata found in 'meta/stats.json'.")

            if archive.read_bytes("meta/tasks.parquet") is None:
                report.add_warning("No tasks index found in 'meta/tasks.parquet'.")

        except (ValueError, InvalidArchiveError) as error:
            logger.debug("Could not read LeRobot v3 info from '{}': {}", archive.path, error)
            report.add_error(f"Unable to parse core dataset metadata from archive ('{type(error).__name__}').")

        if episode_count == 0:
            report.add_warning("Detected 0 episodes from metadata.")
        if frame_count == 0 and episode_count > 0:
            report.add_warning("Detected episodes but total frame count is 0.")

        recording_schema = extract_recording_schema(info)
        uploaded_name = payload.uploaded_archive_name
        suggested_name = PurePath(uploaded_name).stem if uploaded_name else archive.path.stem

        if not recording_schema.cameras:
            report.add_warning("No camera streams inferred from dataset metadata.")
        if not recording_schema.robots:
            report.add_warning("No robot schema inferred from dataset metadata.")

        logger.info(
            "LeRobotV3Adapter manifest draft parsed: archive='{}'",
            archive.path,
        )

        manifest = DatasetManifest(
            source_type=DatasetImportSource.LEROBOT_V3,
            suggested_name=suggested_name,
            statistics=DatasetManifestStatistics(
                episode_count=episode_count,
                frame_count=frame_count,
            ),
            dataset_schema=recording_schema,
        )

        return manifest, report

    def validate_pre_commit(self, payload: DatasetImportJobPayload) -> ImportValidationReport:
        report = ImportValidationReport()

        if payload.finalize_input is None:
            report.add_error("Finalize input is required before commit")

        return report

    async def commit(self, payload: DatasetImportJobPayload, project_id: UUID, archive: SafeZipArchive) -> Dataset:
        if payload.finalize_input is None:
            raise ValueError("Cannot commit dataset import without finalize input")

        settings = get_settings()
        settings.datasets_dir.mkdir(parents=True, exist_ok=True)

        dataset_id = uuid4()
        destination_dir = settings.datasets_dir / str(dataset_id)
        destination_dir.mkdir(parents=True, exist_ok=False)
        logger.info(
            "LeRobotV3Adapter commit destination prepared: dataset_id='{}', destination_dir='{}'",
            dataset_id,
            destination_dir,
        )

        check_disk_headroom(
            settings.datasets_dir,
            required_bytes=archive.estimated_uncompressed_size(),
            min_free_bytes=settings.data_import_min_free_bytes,
        )

        try:
            extracted_count = archive.extract_to(
                destination_dir,
                min_free_bytes=settings.data_import_min_free_bytes,
            )

            # Allow users to upload a zip with the dataset either at archive root
            # or inside a single top-level folder.
            flatten_single_root_directory(destination_dir)

            logger.info(
                "LeRobotV3Adapter extracted archive: archive='{}', destination_dir='{}', file_count={}",
                archive.path,
                destination_dir,
                extracted_count,
            )

            dataset = Dataset(
                id=dataset_id,
                name=payload.finalize_input.dataset_name,
                path=str(destination_dir),
                default_task=payload.finalize_input.default_task,
                project_id=project_id,
                environment_id=payload.finalize_input.environment_id,
            )

            saved = await DatasetService.create_dataset(dataset)
            logger.info(
                "LeRobotV3Adapter dataset persisted: dataset_id='{}', project_id='{}', environment_id='{}', path='{}'",
                saved.id,
                saved.project_id,
                saved.environment_id,
                saved.path,
            )

            return saved
        except Exception:
            shutil.rmtree(destination_dir, ignore_errors=True)
            raise
