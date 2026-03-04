# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Bidirectional converters between LeRobot v3 and raw-video dataset formats.

Provides two converter classes:

- :class:`LeRobotToRawVideoConverter` — reads a LeRobot v3 dataset and writes
  a raw-video dataset (``manifest.json`` + per-episode JSONL + video files).
- :class:`RawVideoToLeRobotConverter` — reads a raw-video dataset and writes
  a LeRobot v3 dataset using the ``LeRobotDataset.create()`` API.

Both converters are designed to be used programmatically or via the CLI
commands in ``converter_cli.py``.
"""

from __future__ import annotations

import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from uuid import uuid4

import click
import numpy as np
from loguru import logger

from .manifest import CameraConfig, DatasetManifest, EpisodeEntry, load_manifest
from .stats import _compute_single_episode_stats, _save_episode_stats
from .video_decode import get_video_info


class LeRobotToRawVideoConverter:
    """Convert a LeRobot v3 dataset to the raw-video format.

    Reads an existing LeRobot v3 dataset on disk (with videos) and writes a
    raw-video dataset consisting of ``manifest.json``, per-episode JSONL data
    files, and per-episode per-camera MP4 video files.

    Video segments are extracted using ffmpeg stream copy for speed.  If the
    stream copy produces an unexpected frame count (due to keyframe alignment
    issues), the converter falls back to re-encoding with libx264.

    Args:
        source: Path to the LeRobot v3 dataset root directory (must contain
            ``meta/info.json``).
        dest: Path to write the raw-video dataset.  Must not exist yet.

    Raises:
        FileNotFoundError: If *source* does not contain a LeRobot dataset.
        FileExistsError: If *dest* already exists.
    """

    def __init__(self, source: Path, dest: Path) -> None:
        self._source = Path(source)
        self._dest = Path(dest)

    def convert(self, *, max_workers: int | None = None) -> None:
        """Run the conversion.

        Args:
            max_workers: Maximum number of threads for parallel episode
                conversion.  Defaults to ``min(num_episodes, os.cpu_count() or 4)``.
                Set to ``1`` to disable parallelism.
        """
        self._validate_paths()

        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        click.echo(f"Loading LeRobot dataset from {self._source} ...")
        dataset = LeRobotDataset(str(uuid4()), root=self._source)

        fps = dataset.meta.fps
        features = dataset.meta.features

        # Derive dimensions and camera names from the features dict.
        state_dim = features["observation.state"]["shape"][0]
        action_dim = features["action"]["shape"][0]

        # Preserve per-element joint names (e.g. ["shoulder_pan.pos", ...]).
        state_names: list[str] = features["observation.state"].get("names") or [f"state_{i}" for i in range(state_dim)]
        action_names: list[str] = features["action"].get("names") or [f"action_{i}" for i in range(action_dim)]

        robot_type: str = getattr(dataset.meta, "robot_type", "unknown") or "unknown"

        camera_names: list[str] = []
        for key in sorted(dataset.meta.video_keys + dataset.meta.image_keys):
            # Strip "observation.images." prefix.
            if key.startswith("observation.images."):
                cam_name = key[len("observation.images.") :]
                camera_names.append(cam_name)

        self._dest.mkdir(parents=True, exist_ok=False)

        num_episodes = dataset.meta.total_episodes

        # Extract task descriptions from the LeRobot task table.
        # .to_dict()["task_index"] returns {task_string: integer_index}.
        tasks: list[str] = []
        task_description = ""
        if dataset.meta.tasks is not None and len(dataset.meta.tasks) > 0:
            tasks_dict = dataset.meta.tasks.to_dict()["task_index"]
            # Sort by integer index (values) to preserve ordering.
            tasks = [str(k) for k, _ in sorted(tasks_dict.items(), key=lambda x: int(x[1]))]
            if tasks:
                task_description = tasks[0]

        # Build a manifest skeleton (without episodes) for per-episode stats
        # computation.  _compute_single_episode_stats only uses state_dim,
        # action_dim, and cameras from the manifest.
        cam_configs = [CameraConfig(name=c) for c in camera_names]
        manifest_skeleton = DatasetManifest(
            name=self._source.name,
            fps=fps,
            state_dim=state_dim,
            action_dim=action_dim,
            state_names=state_names,
            action_names=action_names,
            robot_type=robot_type,
            cameras=cam_configs,
            episodes=[],
            task_description=task_description,
            tasks=tasks,
        )

        workers = max_workers if max_workers is not None else min(num_episodes, os.cpu_count() or 4)
        workers = max(1, workers)

        if workers == 1 or num_episodes <= 1:
            # Sequential path — simpler, avoids thread overhead for small datasets.
            episodes: list[EpisodeEntry] = []
            for ep_idx in range(num_episodes):
                click.echo(f"  Converting episode {ep_idx + 1}/{num_episodes} ...")
                entry = self._convert_single_episode(
                    dataset,
                    ep_idx,
                    camera_names,
                    fps,
                    manifest_skeleton,
                )
                episodes.append(entry)
        else:
            # Parallel path — convert episodes concurrently.
            click.echo(f"  Converting {num_episodes} episodes with {workers} workers ...")
            episodes_by_idx: dict[int, EpisodeEntry] = {}
            with ThreadPoolExecutor(max_workers=workers) as pool:
                future_to_idx = {
                    pool.submit(
                        self._convert_single_episode,
                        dataset,
                        ep_idx,
                        camera_names,
                        fps,
                        manifest_skeleton,
                    ): ep_idx
                    for ep_idx in range(num_episodes)
                }
                for future in as_completed(future_to_idx):
                    ep_idx = future_to_idx[future]
                    entry = future.result()  # propagates exceptions
                    episodes_by_idx[ep_idx] = entry
                    click.echo(f"  Converted episode {ep_idx + 1}/{num_episodes}")

            # Restore original order.
            episodes = [episodes_by_idx[i] for i in range(num_episodes)]

        # ---- Write manifest.json ----
        manifest = manifest_skeleton.model_copy(update={"episodes": episodes})
        manifest_path = self._dest / "manifest.json"
        manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

        click.echo(f"Conversion complete: {num_episodes} episodes written to {self._dest}")

    def _convert_single_episode(
        self,
        dataset: object,
        ep_idx: int,
        camera_names: list[str],
        fps: int,
        manifest_skeleton: DatasetManifest,
    ) -> EpisodeEntry:
        """Convert a single episode: extract video, write JSONL, compute stats.

        This method is safe to call from multiple threads — each episode writes
        to its own directory and reads from disjoint ranges of the HuggingFace
        dataset.

        Returns:
            The :class:`EpisodeEntry` for this episode.
        """
        ep_dir_name = f"episode_{ep_idx:03d}"
        ep_dir = self._dest / ep_dir_name
        ep_dir.mkdir(exist_ok=True)

        ep_meta = dataset.meta.episodes[ep_idx]  # type: ignore[attr-defined]

        # ---- Extract video segments ----
        video_files: dict[str, str] = {}
        for cam_name in camera_names:
            vid_key = f"observation.images.{cam_name}"
            src_video = self._source / dataset.meta.get_video_file_path(ep_idx, vid_key)  # type: ignore[attr-defined]
            from_ts = ep_meta[f"videos/{vid_key}/from_timestamp"]
            to_ts = ep_meta[f"videos/{vid_key}/to_timestamp"]
            dest_video_name = f"{cam_name}.mp4"
            dest_video = ep_dir / dest_video_name

            expected_frames = ep_meta["length"]
            self._extract_video_segment(
                src_video,
                dest_video,
                from_ts,
                to_ts,
                expected_frames,
                fps,
            )
            video_files[cam_name] = dest_video_name

        # ---- Write JSONL data (batch HF dataset access) ----
        from_frame = ep_meta["dataset_from_index"]
        to_frame = ep_meta["dataset_to_index"]
        data_path = ep_dir / "data.jsonl"

        hf_dataset = dataset.hf_dataset  # type: ignore[attr-defined]
        # Check if the HF dataset has a task_index column.
        has_task_index = "task_index" in (getattr(hf_dataset, "column_names", None) or [])
        # Try batch slice access first (much faster for Arrow-backed datasets);
        # fall back to row-by-row if the dataset doesn't support slicing well.
        try:
            batch = hf_dataset[from_frame:to_frame]
            rows_written = self._write_jsonl_from_batch(data_path, batch, to_frame - from_frame, has_task_index)
        except Exception:
            rows_written = self._write_jsonl_row_by_row(data_path, hf_dataset, from_frame, to_frame, has_task_index)

        if rows_written == 0:
            logger.warning("Episode %s has 0 data rows", ep_dir_name)

        episode_entry = EpisodeEntry(
            episode_dir=ep_dir_name,
            data_file="data.jsonl",
            video_files=video_files,
        )

        # ---- Compute per-episode stats (async-friendly) ----
        # Pre-populate the per-episode stats cache so the first training
        # run doesn't need to decode video frames for stats.
        try:
            ep_stats = _compute_single_episode_stats(
                manifest_skeleton,
                episode_entry,
                self._dest,
                image_sample_count=100,
            )
            _save_episode_stats(ep_dir, ep_stats)
        except Exception:
            logger.warning(
                "Failed to compute per-episode stats for %s; will be computed at training time",
                ep_dir_name,
            )

        return episode_entry

    @staticmethod
    def _write_jsonl_from_batch(
        data_path: Path,
        batch: dict,
        num_rows: int,
        has_task_index: bool = False,
    ) -> int:
        """Write JSONL from a batch dict returned by HF dataset slice access.

        Args:
            data_path: Output file path.
            batch: Dict of lists/tensors from ``hf_dataset[from:to]``.
            num_rows: Expected number of rows in the batch.
            has_task_index: Whether to include ``task_index`` from the batch.

        Returns:
            Number of rows written.
        """
        timestamps = batch["timestamp"]
        states = batch["observation.state"]
        actions = batch["action"]
        task_indices = batch.get("task_index") if has_task_index else None

        with data_path.open("w", encoding="utf-8") as fh:
            for i in range(num_rows):
                ts = timestamps[i]
                timestamp = float(ts.item()) if hasattr(ts, "item") else float(ts)

                state = states[i]
                if hasattr(state, "tolist"):
                    state = state.tolist()
                elif hasattr(state, "numpy"):
                    state = state.numpy().tolist()

                action = actions[i]
                if hasattr(action, "tolist"):
                    action = action.tolist()
                elif hasattr(action, "numpy"):
                    action = action.numpy().tolist()

                row: dict = {"timestamp": timestamp, "state": state, "action": action}
                if task_indices is not None:
                    ti = task_indices[i]
                    row["task_index"] = int(ti.item()) if hasattr(ti, "item") else int(ti)
                fh.write(json.dumps(row) + "\n")

        return num_rows

    @staticmethod
    def _write_jsonl_row_by_row(
        data_path: Path,
        hf_dataset: object,
        from_frame: int,
        to_frame: int,
        has_task_index: bool = False,
    ) -> int:
        """Fallback: write JSONL by accessing the HF dataset row by row.

        Returns:
            Number of rows written.
        """
        count = 0
        with data_path.open("w", encoding="utf-8") as fh:
            for global_idx in range(from_frame, to_frame):
                item = hf_dataset[global_idx]  # type: ignore[index]
                timestamp = (
                    float(item["timestamp"].item()) if hasattr(item["timestamp"], "item") else float(item["timestamp"])
                )
                state = item["observation.state"]
                action = item["action"]

                if hasattr(state, "tolist"):
                    state = state.tolist()
                elif hasattr(state, "numpy"):
                    state = state.numpy().tolist()
                if hasattr(action, "tolist"):
                    action = action.tolist()
                elif hasattr(action, "numpy"):
                    action = action.numpy().tolist()

                row: dict = {"timestamp": timestamp, "state": state, "action": action}
                if has_task_index and "task_index" in item:
                    ti = item["task_index"]
                    row["task_index"] = int(ti.item()) if hasattr(ti, "item") else int(ti)
                fh.write(json.dumps(row) + "\n")
                count += 1
        return count

    def _validate_paths(self) -> None:
        """Check that source exists and dest does not."""
        info_path = self._source / "meta" / "info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"Source does not look like a LeRobot dataset (missing {info_path})")
        if self._dest.exists():
            raise FileExistsError(f"Destination already exists: {self._dest}")

    @staticmethod
    def _extract_video_segment(
        src: Path,
        dest: Path,
        from_ts: float,
        to_ts: float,
        expected_frames: int,
        fps: int,
    ) -> None:
        """Extract a video segment, trying stream copy first with fallback to re-encode.

        Args:
            src: Source video file (may contain multiple episodes).
            dest: Destination video file for the extracted segment.
            from_ts: Start timestamp in seconds within *src*.
            to_ts: End timestamp in seconds within *src*.
            expected_frames: Expected number of frames in the output.
            fps: Target FPS for fallback re-encoding.
        """
        # Attempt 1: stream copy (fast, no quality loss).
        _ffmpeg_extract(src, dest, from_ts, to_ts, reencode=False)

        # Verify frame count.
        try:
            info = get_video_info(dest)
            if info.num_frames == expected_frames:
                return  # Success.
            logger.warning(
                "Stream copy produced %d frames (expected %d) for %s; falling back to re-encode.",
                info.num_frames,
                expected_frames,
                dest,
            )
        except Exception:
            logger.warning(
                "Could not verify stream-copy output for %s; falling back to re-encode.",
                dest,
            )

        # Attempt 2: re-encode with libx264.
        dest.unlink(missing_ok=True)
        _ffmpeg_extract(src, dest, from_ts, to_ts, reencode=True, fps=fps)


class RawVideoToLeRobotConverter:
    """Convert a raw-video dataset to LeRobot v3 format.

    Reads a raw-video dataset (``manifest.json`` + JSONL + video files) and
    writes a LeRobot v3 dataset using the ``LeRobotDataset.create()`` /
    ``add_frame()`` / ``save_episode()`` / ``finalize()`` API.

    This necessarily involves decoding every video frame (to supply NumPy
    arrays to ``add_frame()``) and re-encoding them to the LeRobot video
    format.  For large datasets this may be slow.

    Args:
        source: Path to the raw-video dataset root (must contain
            ``manifest.json``).
        dest: Path to write the LeRobot v3 dataset.  Must not exist yet.
        robot_type: Override for the ``robot_type`` field in the LeRobot
            metadata.  If not provided (or ``None``), the value stored in
            the manifest is used.  The CLI exposes this so you can
            correct a missing/wrong robot type without editing the
            manifest.

    Raises:
        FileNotFoundError: If *source* does not contain a ``manifest.json``.
        FileExistsError: If *dest* already exists.
    """

    def __init__(self, source: Path, dest: Path, *, robot_type: str | None = None) -> None:
        self._source = Path(source)
        self._dest = Path(dest)
        self._robot_type_override = robot_type

    def convert(self) -> None:
        """Run the conversion."""
        self._validate_paths()

        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        manifest = load_manifest(self._source)

        click.echo(f"Loading raw-video dataset from {self._source} ...")

        # Probe image resolution from the first episode's first camera.
        first_ep = manifest.episodes[0]
        cam_shapes: dict[str, tuple[int, int, int]] = {}
        for cam in manifest.cameras:
            video_path = self._source / first_ep.episode_dir / first_ep.video_files[cam.name]
            info = get_video_info(video_path)
            cam_shapes[cam.name] = (info.height, info.width, 3)  # (H, W, C) for LeRobot

        # Build LeRobot features dict, preserving per-element joint names.
        features: dict[str, dict] = {
            "observation.state": {
                "dtype": "float32",
                "shape": (manifest.state_dim,),
                "names": manifest.state_names,
            },
            "action": {
                "dtype": "float32",
                "shape": (manifest.action_dim,),
                "names": manifest.action_names,
            },
        }
        for cam in manifest.cameras:
            h, w, c = cam_shapes[cam.name]
            features[f"observation.images.{cam.name}"] = {
                "dtype": "video",
                "shape": (h, w, c),
                "names": ["height", "width", "channels"],
            }

        # Use CLI override if provided, otherwise use the manifest value.
        robot_type = self._robot_type_override or manifest.robot_type

        click.echo("Creating LeRobot dataset ...")
        lr_dataset = LeRobotDataset.create(
            repo_id=str(uuid4()),
            root=self._dest,
            fps=manifest.fps,
            features=features,
            robot_type=robot_type,
            use_videos=True,
            image_writer_threads=4,
        )

        # Build task lookup: use manifest.tasks if available, else fallback.
        tasks = (
            manifest.tasks
            if manifest.tasks
            else ([manifest.task_description] if manifest.task_description else ["default"])
        )
        num_episodes = len(manifest.episodes)

        for ep_idx, episode in enumerate(manifest.episodes):
            click.echo(f"  Converting episode {ep_idx + 1}/{num_episodes} ...")
            ep_dir = self._source / episode.episode_dir

            # Read JSONL data.
            data_path = ep_dir / episode.data_file
            rows = _read_jsonl(data_path)

            # Decode all video frames for each camera.
            cam_frames: dict[str, np.ndarray] = {}
            for cam in manifest.cameras:
                video_path = ep_dir / episode.video_files[cam.name]
                cam_frames[cam.name] = _decode_all_frames_as_numpy(video_path)

            # Determine frame count from the video (authoritative source).
            first_cam = manifest.cameras[0].name
            num_frames = len(cam_frames[first_cam])

            for frame_idx in range(num_frames):
                # Map video frame to nearest data row (handles FPS mismatch).
                data_row_idx = _nearest_data_row(frame_idx, num_frames, len(rows))

                row = rows[data_row_idx]
                # Resolve per-frame task from task_index; default to first task.
                task_idx = row.get("task_index", 0)
                task_str = tasks[task_idx] if 0 <= task_idx < len(tasks) else tasks[0]

                frame: dict = {
                    "task": task_str,
                    "observation.state": np.array(row["state"], dtype=np.float32),
                    "action": np.array(row["action"], dtype=np.float32),
                }
                for cam in manifest.cameras:
                    frame[f"observation.images.{cam.name}"] = cam_frames[cam.name][frame_idx]

                lr_dataset.add_frame(frame)

            lr_dataset.save_episode()

        click.echo("Finalizing LeRobot dataset ...")
        lr_dataset.stop_image_writer()
        lr_dataset.finalize()

        click.echo(f"Conversion complete: {num_episodes} episodes written to {self._dest}")

    def _validate_paths(self) -> None:
        """Check that source has a manifest and dest does not exist."""
        manifest_path = self._source / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Source does not contain a manifest.json: {self._source}")
        if self._dest.exists():
            raise FileExistsError(f"Destination already exists: {self._dest}")


# ======================================================================
# Internal helpers
# ======================================================================


def _ffmpeg_extract(
    src: Path,
    dest: Path,
    from_ts: float,
    to_ts: float,
    *,
    reencode: bool = False,
    fps: int | None = None,
) -> None:
    """Run ffmpeg to extract a video segment.

    Args:
        src: Input video file.
        dest: Output video file.
        from_ts: Start time in seconds.
        to_ts: End time in seconds.
        reencode: If ``True``, re-encode with libx264 instead of stream copy.
        fps: Output FPS (only used when *reencode* is ``True``).
    """
    cmd = ["ffmpeg", "-y", "-loglevel", "error"]

    if not reencode:
        # Stream copy: seek before input for speed.
        cmd += ["-ss", str(from_ts), "-to", str(to_ts), "-i", str(src), "-c", "copy"]
    else:
        # Re-encode: seek then transcode.
        cmd += [
            "-ss",
            str(from_ts),
            "-to",
            str(to_ts),
            "-i",
            str(src),
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-pix_fmt",
            "yuv420p",
        ]
        if fps is not None:
            cmd += ["-r", str(fps)]

    cmd.append(str(dest))

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed (exit {result.returncode}) for {src} -> {dest}: {result.stderr}")


def _read_jsonl(path: Path) -> list[dict]:
    """Read all rows from a JSONL file.

    Returns:
        A list of dicts, each with at least ``"timestamp"``, ``"state"``, ``"action"`` keys.
    """
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _decode_all_frames_as_numpy(video_path: Path) -> np.ndarray:
    """Decode all frames of a video file into a numpy array of shape ``(N, H, W, 3)`` uint8.

    Uses our existing :func:`.video_decode.decode_frames` backend which
    returns float32 ``(N, C, H, W)`` in ``[0, 1]``.  We convert to
    ``(N, H, W, C)`` uint8 for compatibility with ``LeRobotDataset.add_frame()``.
    """
    from .video_decode import decode_frames, get_video_info

    info = get_video_info(video_path)
    if info.num_frames == 0:
        return np.empty((0, info.height, info.width, 3), dtype=np.uint8)

    indices = list(range(info.num_frames))
    # decode_frames returns (N, C, H, W) float32 in [0, 1].
    tensor = decode_frames(video_path, indices)
    # Convert to (N, H, W, C) uint8.
    np_frames = (tensor.permute(0, 2, 3, 1).numpy() * 255).clip(0, 255).astype(np.uint8)
    return np_frames


def _nearest_data_row(frame_idx: int, num_frames: int, num_rows: int) -> int:
    """Map a video frame index to the nearest JSONL data row index.

    Implements the same nearest-neighbour resampling as
    :func:`.frame_index._nearest_neighbor_resample` for a single index.
    """
    if num_frames <= 1 or num_rows <= 1:
        return 0
    scale = (num_rows - 1) / (num_frames - 1)
    return int(round(frame_idx * scale))


__all__ = [
    "LeRobotToRawVideoConverter",
    "RawVideoToLeRobotConverter",
]
