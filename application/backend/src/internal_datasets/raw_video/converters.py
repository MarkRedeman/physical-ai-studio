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
import logging
import shutil
import subprocess
from pathlib import Path
from uuid import uuid4

import click
import numpy as np

from .manifest import CameraConfig, DatasetManifest, EpisodeEntry, load_manifest
from .video_decode import get_video_info

logger = logging.getLogger(__name__)


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

    def convert(self) -> None:
        """Run the conversion."""
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
        episodes: list[EpisodeEntry] = []

        # Extract task description from the first task if available.
        task_description = ""
        if dataset.meta.tasks is not None and len(dataset.meta.tasks) > 0:
            task_description = str(list(dataset.meta.tasks.to_dict()["task_index"].keys())[0])

        for ep_idx in range(num_episodes):
            click.echo(f"  Converting episode {ep_idx + 1}/{num_episodes} ...")
            ep_dir_name = f"episode_{ep_idx:03d}"
            ep_dir = self._dest / ep_dir_name
            ep_dir.mkdir()

            ep_meta = dataset.meta.episodes[ep_idx]

            # ---- Extract video segments ----
            video_files: dict[str, str] = {}
            for cam_name in camera_names:
                vid_key = f"observation.images.{cam_name}"
                src_video = self._source / dataset.meta.get_video_file_path(ep_idx, vid_key)
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

            # ---- Write JSONL data ----
            from_frame = ep_meta["dataset_from_index"]
            to_frame = ep_meta["dataset_to_index"]
            data_path = ep_dir / "data.jsonl"

            with data_path.open("w", encoding="utf-8") as fh:
                for global_idx in range(from_frame, to_frame):
                    item = dataset.hf_dataset[global_idx]
                    timestamp = (
                        float(item["timestamp"].item())
                        if hasattr(item["timestamp"], "item")
                        else float(item["timestamp"])
                    )
                    state = item["observation.state"]
                    action = item["action"]

                    # Convert to plain lists.
                    if hasattr(state, "tolist"):
                        state = state.tolist()
                    elif hasattr(state, "numpy"):
                        state = state.numpy().tolist()
                    if hasattr(action, "tolist"):
                        action = action.tolist()
                    elif hasattr(action, "numpy"):
                        action = action.numpy().tolist()

                    row = {"timestamp": timestamp, "state": state, "action": action}
                    fh.write(json.dumps(row) + "\n")

            episodes.append(
                EpisodeEntry(
                    episode_dir=ep_dir_name,
                    data_file="data.jsonl",
                    video_files=video_files,
                )
            )

        # ---- Write manifest.json ----
        manifest = DatasetManifest(
            name=self._source.name,
            fps=fps,
            state_dim=state_dim,
            action_dim=action_dim,
            state_names=state_names,
            action_names=action_names,
            robot_type=robot_type,
            cameras=[CameraConfig(name=c) for c in camera_names],
            episodes=episodes,
            task_description=task_description,
        )
        manifest_path = self._dest / "manifest.json"
        manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

        click.echo(f"Conversion complete: {num_episodes} episodes written to {self._dest}")

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

        task = manifest.task_description or "default"
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
                frame: dict = {
                    "task": task,
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

    result = subprocess.run(cmd, capture_output=True, text=True)
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
