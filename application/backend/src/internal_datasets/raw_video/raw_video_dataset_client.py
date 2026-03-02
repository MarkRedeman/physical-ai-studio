# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Raw-video dataset client implementing the :class:`DatasetClient` ABC.

This module provides :class:`RawVideoDatasetClient`, which handles CRUD
operations for the raw-video dataset format — creating datasets, recording
episodes (writing JSONL telemetry and piping camera frames to MP4 via ffmpeg),
listing episodes, deleting episodes, and managing the recording mutation
lifecycle.

On-disk layout managed by this client::

    dataset_root/
    ├── manifest.json
    ├── ep_000_<uuid>/
    │   ├── data.jsonl
    │   ├── cam_top.mp4
    │   └── cam_gripper.mp4
    ├── ep_001_<uuid>/
    │   └── ...
    └── .cache/
        └── stats.json
"""

from __future__ import annotations

import base64
import json
import shutil
import time
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from loguru import logger

from internal_datasets.dataset_client import DatasetClient
from internal_datasets.mutations.recording_mutation import RecordingMutation
from internal_datasets.raw_video.manifest import (
    CameraConfig,
    DatasetManifest,
    EpisodeEntry,
    load_manifest,
    save_manifest,
)
from internal_datasets.raw_video.video_writer import VideoWriter
from schemas import Episode, EpisodeVideo
from settings import get_settings


class RawVideoDatasetClient(DatasetClient):
    """Dataset client for the raw-video format.

    Implements all 13 abstract methods of :class:`DatasetClient` to support
    dataset creation, episode recording, episode querying, deletion, and the
    cache-copy mutation pattern used during teleoperation.
    """

    type: str = "raw_video"
    path: Path

    # Manifest — loaded on init if the dataset exists on disk.
    _manifest: DatasetManifest | None

    # Recording state — populated between start_recording / save / discard.
    _episode_dir: Path | None
    _video_writers: dict[str, VideoWriter]
    _jsonl_buffer: list[dict]
    _frame_count: int
    _recording_start_time: float

    def __init__(self, dataset_path: Path) -> None:
        self.path = dataset_path
        self._manifest = None
        self._episode_dir = None
        self._video_writers = {}
        self._jsonl_buffer = []
        self._frame_count = 0
        self._recording_start_time = 0.0

        if self._check_exists():
            self._manifest = load_manifest(self.path)

    # ------------------------------------------------------------------
    # DatasetClient class attributes
    # ------------------------------------------------------------------

    @property
    def exists_on_disk(self) -> bool:
        """Check if the dataset exists (manifest.json is present)."""
        return self._check_exists()

    @property
    def has_episodes(self) -> bool:
        """Whether the dataset has at least one episode."""
        return self._manifest is not None and len(self._manifest.episodes) > 0

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------

    def create(self, fps: int, features: dict, robot_type: str) -> None:
        """Create a new raw-video dataset from environment features.

        The *features* dict is expected to have the structure returned by
        :func:`build_raw_video_manifest_args`:

        .. code-block:: python

            {
                "state_names": ["shoulder_pan.pos", ...],
                "action_names": ["shoulder_pan.pos", ...],
                "cameras": [{"name": "top", "width": 640, "height": 480}, ...],
            }
        """
        if self._check_exists():
            raise RuntimeError(f"Dataset already exists at {self.path}")

        state_names: list[str] = features["state_names"]
        action_names: list[str] = features["action_names"]
        cameras_raw: list[dict] = features["cameras"]

        manifest = DatasetManifest(
            name=self.path.name,
            fps=fps,
            state_dim=len(state_names),
            action_dim=len(action_names),
            state_names=state_names,
            action_names=action_names,
            robot_type=robot_type,
            cameras=[CameraConfig(name=c["name"]) for c in cameras_raw],
            episodes=[],
        )

        self.path.mkdir(parents=True, exist_ok=True)
        save_manifest(self.path, manifest)
        self._manifest = manifest
        logger.info("Created raw-video dataset at {}", self.path)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_episodes(self) -> list[Episode]:
        """Return all episodes as :class:`Episode` schema objects."""
        if self._manifest is None:
            return []

        result: list[Episode] = []
        for idx, ep in enumerate(self._manifest.episodes):
            ep_dir = self.path / ep.episode_dir

            # Read JSONL to get frame count and action data
            data_path = ep_dir / ep.data_file
            rows = _read_jsonl(data_path)
            length = len(rows)
            actions = [row["action"] for row in rows]

            # Build video metadata
            videos: dict[str, EpisodeVideo] = {}
            for cam_name, video_filename in ep.video_files.items():
                video_path = ep_dir / video_filename
                videos[cam_name] = EpisodeVideo(
                    start=0.0,
                    end=length / self._manifest.fps if length > 0 else 0.0,
                    path=str(video_path.relative_to(self.path)),
                )

            # Build thumbnail from first frame of first camera
            thumbnail = self._build_thumbnail(ep_dir, ep) if ep.video_files else None

            result.append(
                Episode(
                    episode_index=idx,
                    length=length,
                    fps=self._manifest.fps,
                    tasks=[self._manifest.task_description] if self._manifest.task_description else [],
                    actions=actions,
                    action_keys=self._manifest.action_names,
                    videos=videos,
                    thumbnail=thumbnail,
                )
            )

        return result

    def get_tasks(self) -> list[str]:
        """Return task descriptions in the dataset."""
        if self._manifest is None:
            return []
        if self._manifest.task_description:
            return [self._manifest.task_description]
        return []

    def get_video_path(self, episode: int, camera: str) -> Path:
        """Get the absolute path to a video file for a given episode and camera."""
        if self._manifest is None:
            raise RuntimeError("Dataset not loaded")
        if episode < 0 or episode >= len(self._manifest.episodes):
            raise IndexError(f"Episode {episode} out of range (0..{len(self._manifest.episodes) - 1})")

        ep = self._manifest.episodes[episode]
        if camera not in ep.video_files:
            raise KeyError(f"Camera {camera!r} not found in episode {episode}")

        return self.path / ep.episode_dir / ep.video_files[camera]

    # ------------------------------------------------------------------
    # Write (recording)
    # ------------------------------------------------------------------

    def prepare_for_writing(self) -> None:
        """Prepare the dataset for recording.

        For raw-video datasets this creates a new episode directory and
        starts ffmpeg video writers for each camera.
        """
        if self._manifest is None:
            raise RuntimeError("Dataset not created — call create() first")

        ep_index = len(self._manifest.episodes)
        short_uuid = uuid4().hex[:8]
        ep_dir_name = f"ep_{ep_index:03d}_{short_uuid}"
        self._episode_dir = self.path / ep_dir_name
        self._episode_dir.mkdir(parents=True, exist_ok=True)

        # Start one ffmpeg process per camera.
        self._video_writers = {}
        for cam_info in self._manifest.cameras:
            cam_name = cam_info.name
            video_filename = f"cam_{cam_name}.mp4"
            writer = VideoWriter(
                output_path=self._episode_dir / video_filename,
                fps=self._manifest.fps,
                # Dimensions will be set on first frame
                width=0,
                height=0,
            )
            # Don't start yet — we need the frame dimensions from the first write.
            self._video_writers[cam_name] = writer

        self._jsonl_buffer = []
        self._frame_count = 0
        self._recording_start_time = time.perf_counter()
        logger.info("Prepared episode directory: {}", self._episode_dir)

    def add_frame(self, obs: dict, act: dict, task: str) -> None:
        """Add a single frame to the recording buffer.

        Args:
            obs: Observation dict mapping camera names to RGB ``np.ndarray``
                 frames (HWC uint8), and joint names to float values.
            act: Action dict mapping joint names to float values.
            task: Task description string (stored in manifest on save).
        """
        if self._manifest is None or self._episode_dir is None:
            raise RuntimeError("Not in a recording session — call prepare_for_writing() first")

        timestamp = time.perf_counter() - self._recording_start_time

        # Extract state and action vectors in the order defined by the manifest.
        state_vector = [float(obs.get(name, 0.0)) for name in self._manifest.state_names]
        action_vector = [float(act.get(name, 0.0)) for name in self._manifest.action_names]

        # Buffer the JSONL row
        self._jsonl_buffer.append(
            {
                "timestamp": round(timestamp, 6),
                "state": state_vector,
                "action": action_vector,
            }
        )

        # Write camera frames to video
        for cam_config in self._manifest.cameras:
            cam_name = cam_config.name
            frame = obs.get(cam_name)
            if frame is None:
                raise ValueError(f"Missing camera frame for {cam_name!r} in observations")

            writer = self._video_writers[cam_name]

            # Lazy start: set dimensions from the first frame
            if not writer.is_running:
                h, w = frame.shape[:2]
                writer._width = w
                writer._height = h
                writer.start()

            writer.write_frame(frame)

        self._frame_count += 1

    def save_episode(self, task: str) -> Episode:
        """Flush the recording buffer as a completed episode.

        Closes all video writers, writes the JSONL data file, and updates
        the manifest.

        Returns:
            The newly created :class:`Episode`.
        """
        if self._manifest is None or self._episode_dir is None:
            raise RuntimeError("No episode to save")

        # Update task description
        if task and not self._manifest.task_description:
            self._manifest = self._manifest.model_copy(update={"task_description": task})

        # Close all video writers
        for cam_name, writer in self._video_writers.items():
            if writer.is_running:
                writer.close()
            logger.debug("Closed video writer for camera {}", cam_name)

        # Write JSONL data file
        data_path = self._episode_dir / "data.jsonl"
        with data_path.open("w", encoding="utf-8") as fh:
            for row in self._jsonl_buffer:
                fh.write(json.dumps(row) + "\n")

        # Build the episode entry for the manifest
        ep_dir_name = self._episode_dir.name
        video_files = {cam.name: f"cam_{cam.name}.mp4" for cam in self._manifest.cameras}
        episode_entry = EpisodeEntry(
            episode_dir=ep_dir_name,
            data_file="data.jsonl",
            video_files=video_files,
        )

        # Update manifest
        new_episodes = list(self._manifest.episodes) + [episode_entry]
        self._manifest = self._manifest.model_copy(update={"episodes": new_episodes})
        save_manifest(self.path, self._manifest)

        # Build the Episode return object
        episode_index = len(self._manifest.episodes) - 1
        actions = [row["action"] for row in self._jsonl_buffer]
        duration = self._frame_count / self._manifest.fps if self._frame_count > 0 else 0.0

        videos: dict[str, EpisodeVideo] = {}
        for cam_name, video_filename in video_files.items():
            videos[cam_name] = EpisodeVideo(
                start=0.0,
                end=duration,
                path=f"{ep_dir_name}/{video_filename}",
            )

        thumbnail = self._build_thumbnail_from_buffer()

        episode = Episode(
            episode_index=episode_index,
            length=self._frame_count,
            fps=self._manifest.fps,
            tasks=[task] if task else [],
            actions=actions,
            action_keys=self._manifest.action_names,
            videos=videos,
            thumbnail=thumbnail,
        )

        logger.info(
            "Saved episode {} ({} frames) to {}",
            episode_index,
            self._frame_count,
            self._episode_dir,
        )

        # Reset recording state (but keep manifest loaded for next episode)
        self._episode_dir = None
        self._video_writers = {}
        self._jsonl_buffer = []
        self._frame_count = 0

        return episode

    def discard_buffer(self) -> None:
        """Discard the current recording buffer without saving."""
        if self._episode_dir is None:
            return

        # Abort video writers
        for writer in self._video_writers.values():
            writer.abort()

        # Remove the episode directory
        if self._episode_dir.is_dir():
            shutil.rmtree(self._episode_dir)
            logger.info("Discarded episode directory: {}", self._episode_dir)

        self._episode_dir = None
        self._video_writers = {}
        self._jsonl_buffer = []
        self._frame_count = 0

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete_episodes(self, episode_indices: list[int], output_path: Path) -> DatasetClient:
        """Copy dataset to *output_path* excluding the given episodes.

        Returns a new :class:`RawVideoDatasetClient` pointing at *output_path*.
        """
        if self._manifest is None:
            raise RuntimeError("Dataset not loaded")

        indices_to_remove = set(episode_indices)
        remaining_episodes: list[EpisodeEntry] = []

        output_path.mkdir(parents=True, exist_ok=True)

        for idx, ep in enumerate(self._manifest.episodes):
            if idx in indices_to_remove:
                continue
            # Copy episode directory
            src_ep_dir = self.path / ep.episode_dir
            dst_ep_dir = output_path / ep.episode_dir
            if src_ep_dir.is_dir():
                shutil.copytree(src_ep_dir, dst_ep_dir)
            remaining_episodes.append(ep)

        # Write new manifest
        new_manifest = self._manifest.model_copy(update={"episodes": remaining_episodes})
        save_manifest(output_path, new_manifest)

        return RawVideoDatasetClient(output_path)

    def delete(self) -> None:
        """Delete the entire dataset from disk."""
        if self.path.is_dir():
            shutil.rmtree(self.path)
        self._manifest = None
        logger.info("Deleted dataset at {}", self.path)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def teardown(self) -> None:
        """Finalize the dataset or delete if no episodes exist."""
        if self._manifest is None or len(self._manifest.episodes) == 0:
            logger.info("Removing dataset since it has no episodes")
            self.delete()
        else:
            logger.info("Finalizing raw-video dataset")
            self.finalize()

    def finalize(self) -> None:
        """Finalize the dataset.

        For raw-video datasets, finalization ensures any in-progress video
        writers are closed and the manifest is up to date.  Invalidates the
        stats cache so it gets recomputed on next training run.
        """
        # Close any dangling video writers (safety net)
        for writer in self._video_writers.values():
            if writer.is_running:
                writer.close()
        self._video_writers = {}

        # Invalidate stats cache
        cache_dir = self.path / ".cache"
        stats_file = cache_dir / "stats.json"
        if stats_file.is_file():
            stats_file.unlink()
            logger.debug("Invalidated stats cache")

        logger.info("Finalized raw-video dataset at {}", self.path)

    def overwrite(self, source: DatasetClient) -> None:
        """Overwrite this dataset with the contents of *source*."""
        if not isinstance(source, RawVideoDatasetClient):
            raise ValueError(f"Cannot overwrite raw-video dataset with {source.__class__.__name__}")

        if self.path.is_dir():
            shutil.rmtree(self.path)

        shutil.copytree(source.path, self.path)
        self._manifest = load_manifest(self.path)
        logger.info("Overwrote dataset at {} from {}", self.path, source.path)

    def start_recording_mutation(self, fps: int, features: dict, robot_type: str) -> RecordingMutation:
        """Create a recording mutation using a cache dataset.

        If the dataset already exists, the entire dataset is copied to a
        cache directory.  Otherwise, a new dataset is created in the cache.
        Recording happens in the cache; on teardown the cache overwrites the
        source.
        """
        settings = get_settings()
        cache_dir = settings.cache_dir / str(uuid4())

        logger.info("Creating cache dataset at {}", cache_dir)
        if self.exists_on_disk:
            shutil.copytree(self.path, cache_dir)
            cache_dataset = RawVideoDatasetClient(cache_dir)
        else:
            cache_dataset = RawVideoDatasetClient(cache_dir)
            cache_dataset.create(fps, features, robot_type)

        return RecordingMutation(self, cache_dataset)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_exists(self) -> bool:
        """Check whether manifest.json exists at the dataset path."""
        return (self.path / "manifest.json").is_file()

    def _build_thumbnail(self, ep_dir: Path, ep: EpisodeEntry) -> str | None:
        """Build a base64 JPEG thumbnail from the first frame of the first camera."""
        if not ep.video_files:
            return None

        thumbnail_size = (320, 240)
        first_cam = next(iter(ep.video_files))
        video_path = ep_dir / ep.video_files[first_cam]

        if not video_path.is_file():
            return None

        try:
            cap = cv2.VideoCapture(str(video_path))
            ret, frame = cap.read()
            cap.release()
            if not ret or frame is None:
                return None
            thumbnail = cv2.resize(frame, thumbnail_size)
            _, imagebytes = cv2.imencode(".jpg", thumbnail)
            return base64.b64encode(imagebytes).decode()
        except Exception:
            logger.warning("Failed to build thumbnail from {}", video_path)
            return None

    def _build_thumbnail_from_buffer(self) -> str | None:
        """Build a thumbnail from the last frame in the JSONL buffer.

        Since the raw frames are already written to ffmpeg and not stored in
        memory, this returns ``None``.  A future optimisation could cache the
        last frame during recording.
        """
        # TODO: Cache last frame during add_frame() for thumbnail generation
        return None


def _read_jsonl(path: Path) -> list[dict]:
    """Read all rows from a JSONL file."""
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


__all__ = ["RawVideoDatasetClient"]
