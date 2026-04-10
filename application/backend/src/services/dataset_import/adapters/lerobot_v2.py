from __future__ import annotations

import json
import shutil
from pathlib import PurePath
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import cv2
import pandas as pd
from loguru import logger

from exceptions import InvalidArchiveError
from schemas import Dataset
from schemas.import_job import (
    DatasetImportJobPayload,
    DatasetImportSource,
    DatasetManifest,
    DatasetManifestCapture,
    ImportValidationIssue,
    ImportValidationReport,
)
from services.archive_safety import SafeZipArchive, check_disk_headroom, flatten_single_root_directory
from services.dataset_service import DatasetService
from settings import get_settings

from .base import DatasetImportAdapter
from .recording_schema import extract_recording_schema

if TYPE_CHECKING:
    from pathlib import Path


class LeRobotV2Adapter(DatasetImportAdapter):
    """Adapter for legacy LeRobot v2.x datasets.

    Current implementation targets a practical v2.1->v3-like migration for
    ingestion into our internal v3-oriented layout.
    """

    source = DatasetImportSource.LEROBOT_V2

    def detect(self, archive: SafeZipArchive) -> bool:
        has_info = False
        has_legacy_episodes = False
        has_legacy_tasks = False
        has_episode_parquet = False

        for name in archive.iter_normalized_names():
            if not has_info and name == "meta/info.json":
                has_info = True
            if not has_legacy_episodes and name == "meta/episodes.jsonl":
                has_legacy_episodes = True
            if not has_legacy_tasks and name == "meta/tasks.jsonl":
                has_legacy_tasks = True
            if (
                not has_episode_parquet
                and name.startswith("data/")
                and "episode_" in name
                and name.endswith(".parquet")
            ):
                has_episode_parquet = True

            if has_info and has_legacy_episodes and has_legacy_tasks and has_episode_parquet:
                return True

        return False

    def parse_to_draft_manifest(self, archive: SafeZipArchive, payload: DatasetImportJobPayload) -> DatasetManifest:
        _ = payload
        fps: int | None = None
        episode_count: int | None = None
        frame_count: int | None = None
        info: dict = {}

        try:
            raw_info = archive.read_json("meta/info.json")
            if raw_info is not None:
                info = raw_info
                fps_raw = info.get("fps")
                try:
                    fps = int(float(fps_raw)) if fps_raw is not None else None
                except (TypeError, ValueError):
                    fps = None

            episodes_rows = archive.read_jsonl("meta/episodes.jsonl")
            if episodes_rows:
                episode_count = len(episodes_rows)
                frame_count = sum(int(row.get("length", 0)) for row in episodes_rows if isinstance(row, dict))
        except (ValueError, InvalidArchiveError) as error:
            logger.debug("Could not read LeRobot v2 info from '{}': {}", archive.path, error)

        recording_schema = extract_recording_schema(info)
        uploaded_name = payload.uploaded_archive_name if isinstance(payload.uploaded_archive_name, str) else None
        suggested_name = PurePath(uploaded_name).stem if uploaded_name else archive.path.stem

        return DatasetManifest(
            source_type=DatasetImportSource.LEROBOT_V2,
            suggested_name=suggested_name,
            capture=DatasetManifestCapture(
                fps=fps,
                episode_count=episode_count,
                frame_count=frame_count,
            ),
            schema_=recording_schema,
        )

    def validate_pre_finalize(
        self, manifest: DatasetManifest, payload: DatasetImportJobPayload
    ) -> ImportValidationReport:
        _ = manifest
        _ = payload
        return ImportValidationReport(is_valid=True)

    def validate_pre_commit(self, payload: DatasetImportJobPayload) -> ImportValidationReport:
        report = ImportValidationReport(is_valid=True)

        if payload.finalize_input is None:
            report.is_valid = False
            report.required_user_inputs = ["dataset_name", "environment_id"]
            report.blocking_errors.append(
                ImportValidationIssue(code="finalize_input_missing", message="Finalize input is required before commit")
            )

        if not payload.archive_staging_id:
            report.is_valid = False
            report.blocking_errors.append(
                ImportValidationIssue(code="archive_missing", message="No archive path associated with import job")
            )

        return report

    async def commit(self, payload: DatasetImportJobPayload, project_id: UUID, archive: SafeZipArchive) -> Dataset:
        if payload.finalize_input is None:
            raise ValueError("Cannot commit dataset import without finalize input")
        if not payload.archive_staging_id:
            raise ValueError("No archive path associated with dataset import job")

        settings = get_settings()
        settings.datasets_dir.mkdir(parents=True, exist_ok=True)

        dataset_id = uuid4()
        destination_dir = settings.datasets_dir / str(dataset_id)
        destination_dir.mkdir(parents=True, exist_ok=False)

        check_disk_headroom(
            settings.datasets_dir,
            required_bytes=archive.estimated_uncompressed_size(),
            min_free_bytes=settings.data_import_min_free_bytes,
        )

        try:
            archive.extract_to(
                destination_dir,
                min_free_bytes=settings.data_import_min_free_bytes,
            )

            # Allow users to upload a zip with the dataset either at archive root
            # or inside a single top-level folder.
            flatten_single_root_directory(destination_dir)

            self._migrate_v21_to_v30_layout(destination_dir)

            dataset = Dataset(
                id=dataset_id,
                name=payload.finalize_input.dataset_name,
                path=str(destination_dir),
                default_task=payload.finalize_input.default_task or "",
                project_id=project_id,
                environment_id=payload.finalize_input.environment_id,
            )
            saved = await DatasetService.create_dataset(dataset)
            return saved  # noqa: RET504
        except Exception:
            shutil.rmtree(destination_dir, ignore_errors=True)
            raise

    def _migrate_v21_to_v30_layout(self, root: Path) -> None:
        info = self._load_info(root)
        self._convert_tasks(root)

        data_index = self._convert_data_files(root)
        video_index = self._convert_video_files(root)
        self._write_episodes_parquet(root, data_index, video_index)
        self._cleanup_legacy_meta_files(root)

        # Compute real sizes from migrated output trees after conversion
        data_size_bytes = sum(f.stat().st_size for f in (root / "data").rglob("*.parquet") if f.is_file())
        video_size_bytes = sum(f.stat().st_size for f in (root / "videos").rglob("*.mp4") if f.is_file())
        data_files_size_in_mb = round(data_size_bytes / (1024 * 1024), 3)
        video_files_size_in_mb = round(video_size_bytes / (1024 * 1024), 3)

        self._write_info_v30(
            root, info, data_files_size_in_mb=data_files_size_in_mb, video_files_size_in_mb=video_files_size_in_mb
        )

    def _load_info(self, root: Path) -> dict:
        info_path = root / "meta" / "info.json"
        if not info_path.exists():
            raise ValueError("LeRobot import requires 'meta/info.json'")
        return json.loads(info_path.read_text())

    def _write_info_v30(
        self, root: Path, info: dict, *, data_files_size_in_mb: float = 0.0, video_files_size_in_mb: float = 0.0
    ) -> None:
        info["codebase_version"] = "v3.0"
        info.pop("total_chunks", None)
        info.pop("total_videos", None)
        info["data_files_size_in_mb"] = data_files_size_in_mb
        info["video_files_size_in_mb"] = video_files_size_in_mb
        info["data_path"] = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
        info["video_path"] = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"

        fps = int(info.get("fps", 0) or 0)
        info["fps"] = fps
        features = info.get("features", {})
        for feature_name, feature_info in features.items():
            if isinstance(feature_info, dict) and feature_info.get("dtype") != "video":
                feature_info["fps"] = fps
                features[feature_name] = feature_info
        info["features"] = features

        meta_dir = root / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        (meta_dir / "info.json").write_text(json.dumps(info, indent=2))

    def _convert_tasks(self, root: Path) -> None:
        tasks_jsonl = root / "meta" / "tasks.jsonl"
        if not tasks_jsonl.exists():
            return

        rows = self._read_jsonl(tasks_jsonl)
        rows = sorted(rows, key=lambda item: int(item.get("task_index", 0)))
        task_names = [str(item.get("task", "")) for item in rows]
        task_indices = [int(item.get("task_index", idx)) for idx, item in enumerate(rows)]
        tasks_df = pd.DataFrame({"task_index": task_indices}, index=pd.Index(task_names, name="task"))
        tasks_df.to_parquet(root / "meta" / "tasks.parquet")

    def _convert_data_files(self, root: Path) -> dict[int, dict[str, int]]:
        source_data_root = root / "data"
        target_data_root = root / "data_v3"
        shutil.rmtree(target_data_root, ignore_errors=True)
        target_data_root.mkdir(parents=True, exist_ok=True)

        episode_paths = sorted(source_data_root.glob("chunk-*/episode_*.parquet"), key=self._episode_index_from_path)
        if not episode_paths:
            raise ValueError("No legacy episode parquet files found under 'data/chunk-*/episode_*.parquet'")

        data_index: dict[int, dict[str, int]] = {}
        global_frame_index = 0
        for file_counter, episode_path in enumerate(episode_paths):
            episode_index = self._episode_index_from_path(episode_path)
            chunk_index = file_counter // 1000
            file_index = file_counter % 1000

            out_dir = target_data_root / f"chunk-{chunk_index:03d}"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"file-{file_index:03d}.parquet"

            frame_df = pd.read_parquet(episode_path)
            frame_count = len(frame_df)
            frame_df.to_parquet(out_path, index=False)

            data_index[episode_index] = {
                "data/chunk_index": chunk_index,
                "data/file_index": file_index,
                "dataset_from_index": global_frame_index,
                "dataset_to_index": global_frame_index + frame_count,
            }

            global_frame_index += frame_count

        shutil.rmtree(source_data_root, ignore_errors=True)
        target_data_root.rename(source_data_root)
        return data_index

    def _convert_video_files(self, root: Path) -> dict[int, dict[str, int | float]]:
        source_videos_root = root / "videos"
        if not source_videos_root.exists():
            return {}

        camera_names = sorted({path.parent.name for path in source_videos_root.glob("chunk-*/**/episode_*.mp4")})
        if not camera_names:
            return {}

        target_videos_root = root / "videos_v3"
        shutil.rmtree(target_videos_root, ignore_errors=True)
        target_videos_root.mkdir(parents=True, exist_ok=True)

        video_index: dict[int, dict[str, int | float]] = {}
        for camera_name in camera_names:
            episode_video_paths = sorted(
                source_videos_root.glob(f"chunk-*/{camera_name}/episode_*.mp4"),
                key=self._episode_index_from_path,
            )

            for file_counter, episode_video_path in enumerate(episode_video_paths):
                episode_index = self._episode_index_from_path(episode_video_path)
                chunk_index = file_counter // 1000
                file_index = file_counter % 1000

                target_dir = target_videos_root / camera_name / f"chunk-{chunk_index:03d}"
                target_dir.mkdir(parents=True, exist_ok=True)
                target_path = target_dir / f"file-{file_index:03d}.mp4"
                shutil.copy2(episode_video_path, target_path)

                duration_seconds = self._video_duration_seconds(target_path)
                entry = video_index.setdefault(episode_index, {})
                entry[f"videos/{camera_name}/chunk_index"] = chunk_index
                entry[f"videos/{camera_name}/file_index"] = file_index
                entry[f"videos/{camera_name}/from_timestamp"] = 0.0
                entry[f"videos/{camera_name}/to_timestamp"] = duration_seconds

        shutil.rmtree(source_videos_root, ignore_errors=True)
        target_videos_root.rename(source_videos_root)
        return video_index

    def _write_episodes_parquet(
        self,
        root: Path,
        data_index: dict[int, dict[str, int]],
        video_index: dict[int, dict[str, int | float]],
    ) -> None:
        episodes_rows = self._read_jsonl(root / "meta" / "episodes.jsonl")
        stats_rows = self._read_jsonl(root / "meta" / "episodes_stats.jsonl")

        legacy_episode_map = {int(row["episode_index"]): row for row in episodes_rows if "episode_index" in row}
        legacy_stats_map = {
            int(row["episode_index"]): row.get("stats", {}) for row in stats_rows if "episode_index" in row
        }

        rows: list[dict] = []
        for episode_index in sorted(data_index.keys()):
            legacy_episode = legacy_episode_map.get(episode_index, {})
            row: dict = {
                "episode_index": episode_index,
                "tasks": legacy_episode.get("tasks", []),
                "length": int(legacy_episode.get("length", 0)),
                "meta/episodes/chunk_index": 0,
                "meta/episodes/file_index": 0,
            }
            row.update(data_index[episode_index])
            row.update(video_index.get(episode_index, {}))
            row.update(self._flatten_stats(legacy_stats_map.get(episode_index, {})))
            rows.append(row)

        episodes_df = pd.DataFrame(rows)
        episodes_out = root / "meta" / "episodes" / "chunk-000"
        shutil.rmtree(root / "meta" / "episodes", ignore_errors=True)
        episodes_out.mkdir(parents=True, exist_ok=True)
        episodes_df.to_parquet(episodes_out / "file-000.parquet", index=False)

    def _cleanup_legacy_meta_files(self, root: Path) -> None:
        for filename in ("tasks.jsonl", "episodes.jsonl", "episodes_stats.jsonl"):
            (root / "meta" / filename).unlink(missing_ok=True)

    def _flatten_stats(self, stats: dict) -> dict[str, float | int]:
        flattened: dict[str, float | int] = {}

        def _flatten_node(node: dict, prefix: str) -> None:
            for key, value in node.items():
                next_prefix = f"{prefix}/{key}" if prefix else key
                if isinstance(value, dict):
                    _flatten_node(value, next_prefix)
                elif isinstance(value, int | float):
                    flattened[next_prefix] = value

        _flatten_node(stats, "stats")
        return flattened

    def _read_jsonl(self, path: Path) -> list[dict]:
        if not path.exists():
            return []

        rows: list[dict] = []
        for raw_line in path.read_text().splitlines():
            normalized_line = raw_line.strip()
            if not normalized_line:
                continue
            rows.append(json.loads(normalized_line))
        return rows

    def _episode_index_from_path(self, path: Path) -> int:
        stem = path.stem
        if "episode_" not in stem:
            raise ValueError(f"Cannot infer episode index from '{path}'")
        return int(stem.split("episode_")[-1])

    def _video_duration_seconds(self, video_path: Path) -> float:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return 0.0
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if fps and fps > 0:
                return float(frame_count / fps)
            return 0.0
        finally:
            cap.release()
