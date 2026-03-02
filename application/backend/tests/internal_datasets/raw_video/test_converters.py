# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for bidirectional LeRobot v3 / raw-video dataset converters.

Tests cover:
- LeRobotToRawVideoConverter: reads a mocked LeRobot dataset, writes raw-video format
- RawVideoToLeRobotConverter: reads a synthetic raw-video dataset, writes via mocked LeRobot API
- Helper functions: _read_jsonl, _nearest_data_row, _ffmpeg_extract
- Path validation and error handling
- CLI entry points
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest
import torch
from click.testing import CliRunner

from internal_datasets.raw_video.converters import (
    LeRobotToRawVideoConverter,
    RawVideoToLeRobotConverter,
    _nearest_data_row,
    _read_jsonl,
)
from internal_datasets.raw_video.manifest import load_manifest
from internal_datasets.raw_video.video_decode import VideoInfo


# ============================================================================
# Fixtures & Helpers
# ============================================================================


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write a list of dicts as newline-delimited JSON."""
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _make_data_rows(n: int, state_dim: int = 6, action_dim: int = 6) -> list[dict]:
    """Generate *n* dummy JSONL data rows."""
    return [
        {
            "timestamp": float(i) / 30.0,
            "state": [float(i) * 0.1] * state_dim,
            "action": [float(i) * 0.01] * action_dim,
        }
        for i in range(n)
    ]


def _make_raw_dataset(
    tmp_path: Path,
    *,
    num_episodes: int = 2,
    frames_per_episode: int = 5,
    cameras: list[str] | None = None,
    state_dim: int = 6,
    action_dim: int = 6,
    state_names: list[str] | None = None,
    action_names: list[str] | None = None,
    robot_type: str = "so100",
    fps: int = 30,
    task_description: str = "pick up cube",
) -> Path:
    """Create a synthetic raw-video dataset on disk.

    Note: video files are empty (0 bytes) — tests must mock video decoding.
    """
    if cameras is None:
        cameras = ["top"]
    if state_names is None:
        state_names = [f"state_{i}" for i in range(state_dim)]
    if action_names is None:
        action_names = [f"action_{i}" for i in range(action_dim)]

    dataset_dir = tmp_path / "raw_dataset"
    dataset_dir.mkdir()

    episodes = []
    for i in range(num_episodes):
        ep_dir_name = f"episode_{i:03d}"
        ep_dir = dataset_dir / ep_dir_name
        ep_dir.mkdir()

        _write_jsonl(ep_dir / "data.jsonl", _make_data_rows(frames_per_episode, state_dim, action_dim))

        video_files = {}
        for cam in cameras:
            video_name = f"{cam}.mp4"
            (ep_dir / video_name).touch()
            video_files[cam] = video_name

        episodes.append(
            {
                "episode_dir": ep_dir_name,
                "data_file": "data.jsonl",
                "video_files": video_files,
            }
        )

    manifest = {
        "name": "test_dataset",
        "fps": fps,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "state_names": state_names,
        "action_names": action_names,
        "robot_type": robot_type,
        "cameras": [{"name": c} for c in cameras],
        "episodes": episodes,
        "task_description": task_description,
    }
    (dataset_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return dataset_dir


def _make_mock_lerobot_dataset(
    source_path: Path,
    *,
    num_episodes: int = 2,
    frames_per_episode: int = 5,
    cameras: list[str] | None = None,
    state_dim: int = 6,
    action_dim: int = 6,
    state_names: list[str] | None = None,
    action_names: list[str] | None = None,
    robot_type: str = "so100",
    fps: int = 30,
) -> MagicMock:
    """Create a mock LeRobotDataset with realistic metadata and hf_dataset."""
    if cameras is None:
        cameras = ["top"]
    if state_names is None:
        state_names = [f"joint_{i}.pos" for i in range(state_dim)]
    if action_names is None:
        action_names = [f"joint_{i}.pos" for i in range(action_dim)]

    total_frames = num_episodes * frames_per_episode
    video_keys = [f"observation.images.{c}" for c in cameras]
    image_keys: list[str] = []

    # Build features dict.
    features = {
        "observation.state": {"dtype": "float32", "shape": (state_dim,), "names": state_names},
        "action": {"dtype": "float32", "shape": (action_dim,), "names": action_names},
    }
    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channels"],
        }

    # Build episodes metadata.
    episodes_list = []
    for ep_idx in range(num_episodes):
        from_idx = ep_idx * frames_per_episode
        to_idx = from_idx + frames_per_episode
        ep = {
            "episode_index": ep_idx,
            "length": frames_per_episode,
            "dataset_from_index": from_idx,
            "dataset_to_index": to_idx,
            "tasks": ["pick up cube"],
            "data/chunk_index": 0,
            "data/file_index": 0,
        }
        for vid_key in video_keys:
            ep[f"videos/{vid_key}/chunk_index"] = 0
            ep[f"videos/{vid_key}/file_index"] = 0
            ep[f"videos/{vid_key}/from_timestamp"] = float(ep_idx * frames_per_episode) / fps
            ep[f"videos/{vid_key}/to_timestamp"] = float((ep_idx + 1) * frames_per_episode) / fps
        episodes_list.append(ep)

    # Build hf_dataset entries.
    hf_data = []
    for ep_idx in range(num_episodes):
        for frame_idx in range(frames_per_episode):
            global_idx = ep_idx * frames_per_episode + frame_idx
            item = {
                "timestamp": torch.tensor(float(frame_idx) / fps),
                "episode_index": torch.tensor(ep_idx),
                "frame_index": torch.tensor(frame_idx),
                "index": torch.tensor(global_idx),
                "task_index": torch.tensor(0),
                "observation.state": torch.tensor([float(global_idx) * 0.1] * state_dim),
                "action": torch.tensor([float(global_idx) * 0.01] * action_dim),
            }
            hf_data.append(item)

    mock_hf = MagicMock()
    mock_hf.__getitem__ = MagicMock(side_effect=lambda idx: hf_data[idx])
    mock_hf.__len__ = MagicMock(return_value=total_frames)

    # Build tasks.
    mock_tasks = MagicMock()
    mock_tasks.to_dict.return_value = {"task_index": {"pick up cube": 0}}
    mock_tasks.__len__ = MagicMock(return_value=1)

    # Build meta.
    mock_meta = MagicMock()
    mock_meta.fps = fps
    mock_meta.features = features
    mock_meta.video_keys = video_keys
    mock_meta.image_keys = image_keys
    mock_meta.camera_keys = video_keys + image_keys
    mock_meta.total_episodes = num_episodes
    mock_meta.total_frames = total_frames
    mock_meta.tasks = mock_tasks
    mock_meta.robot_type = robot_type
    mock_meta.root = source_path
    mock_meta.episodes = episodes_list

    def get_video_file_path(ep_idx, vid_key):
        return Path(f"videos/{vid_key}/chunk-000/file-000.mp4")

    mock_meta.get_video_file_path = MagicMock(side_effect=get_video_file_path)

    # Create the actual video file paths on disk so ffmpeg source validation passes.
    for vid_key in video_keys:
        video_dir = source_path / "videos" / vid_key / "chunk-000"
        video_dir.mkdir(parents=True, exist_ok=True)
        (video_dir / "file-000.mp4").touch()

    # Also create meta/info.json for path validation.
    (source_path / "meta").mkdir(parents=True, exist_ok=True)
    (source_path / "meta" / "info.json").write_text("{}", encoding="utf-8")

    # Build the dataset mock.
    mock_dataset = MagicMock()
    mock_dataset.meta = mock_meta
    mock_dataset.hf_dataset = mock_hf
    mock_dataset.num_episodes = num_episodes
    mock_dataset.features = features

    return mock_dataset


# ============================================================================
# _read_jsonl tests
# ============================================================================


class TestReadJsonl:
    def test_read_jsonl_basic(self, tmp_path: Path):
        rows = _make_data_rows(3)
        path = tmp_path / "data.jsonl"
        _write_jsonl(path, rows)

        result = _read_jsonl(path)
        assert len(result) == 3
        assert result[0]["timestamp"] == 0.0
        assert len(result[0]["state"]) == 6

    def test_read_jsonl_empty_lines(self, tmp_path: Path):
        """Empty lines in JSONL should be skipped."""
        path = tmp_path / "data.jsonl"
        path.write_text(
            '{"timestamp": 0.0, "state": [1], "action": [2]}\n\n{"timestamp": 0.1, "state": [3], "action": [4]}\n'
        )

        result = _read_jsonl(path)
        assert len(result) == 2


# ============================================================================
# _nearest_data_row tests
# ============================================================================


class TestNearestDataRow:
    def test_identity(self):
        """When frames == rows, mapping is identity."""
        for i in range(5):
            assert _nearest_data_row(i, 5, 5) == i

    def test_single_frame(self):
        assert _nearest_data_row(0, 1, 10) == 0

    def test_single_row(self):
        assert _nearest_data_row(5, 10, 1) == 0

    def test_downsample(self):
        # 10 frames, 5 rows: scale = 4/9
        assert _nearest_data_row(0, 10, 5) == 0
        assert _nearest_data_row(9, 10, 5) == 4


# ============================================================================
# LeRobotToRawVideoConverter tests
# ============================================================================


class TestLeRobotToRawVideoConverter:
    """Test conversion from LeRobot v3 format to raw-video format."""

    @patch("internal_datasets.raw_video.converters.get_video_info")
    @patch("subprocess.run")
    def test_basic_conversion(self, mock_subprocess, mock_video_info, tmp_path: Path):
        """Convert a 2-episode, 1-camera LeRobot dataset to raw-video format."""
        source = tmp_path / "lerobot_source"
        source.mkdir()
        dest = tmp_path / "raw_output"

        mock_dataset = _make_mock_lerobot_dataset(
            source,
            num_episodes=2,
            frames_per_episode=5,
            cameras=["top"],
        )

        # Mock ffmpeg to succeed (stream copy produces correct frame count).
        mock_subprocess.return_value = MagicMock(returncode=0)
        mock_video_info.return_value = VideoInfo(
            num_frames=5,
            fps=30.0,
            width=640,
            height=480,
            duration_s=5 / 30.0,
            codec="h264",
        )

        with patch("lerobot.datasets.lerobot_dataset.LeRobotDataset", return_value=mock_dataset):
            converter = LeRobotToRawVideoConverter(source=source, dest=dest)
            converter.convert()

        # Since subprocess.run is mocked, ffmpeg won't actually create video
        # files.  Touch them so load_manifest's disk validation passes.
        for i in range(2):
            (dest / f"episode_{i:03d}" / "top.mp4").touch()

        # Verify manifest.
        manifest = load_manifest(dest)
        assert manifest.fps == 30
        assert manifest.state_dim == 6
        assert manifest.action_dim == 6
        assert manifest.state_names == [f"joint_{i}.pos" for i in range(6)]
        assert manifest.action_names == [f"joint_{i}.pos" for i in range(6)]
        assert manifest.robot_type == "so100"
        assert len(manifest.episodes) == 2
        assert len(manifest.cameras) == 1
        assert manifest.cameras[0].name == "top"
        assert manifest.task_description == "pick up cube"

        # Verify JSONL files exist and have correct row count.
        for i in range(2):
            data_path = dest / f"episode_{i:03d}" / "data.jsonl"
            assert data_path.is_file()
            rows = _read_jsonl(data_path)
            assert len(rows) == 5

        # Verify ffmpeg was called twice (once per episode).
        assert mock_subprocess.call_count == 2

    @patch("internal_datasets.raw_video.converters.get_video_info")
    @patch("subprocess.run")
    def test_multi_camera_conversion(self, mock_subprocess, mock_video_info, tmp_path: Path):
        """Ensure all cameras are extracted."""
        source = tmp_path / "lerobot_source"
        source.mkdir()
        dest = tmp_path / "raw_output"

        mock_dataset = _make_mock_lerobot_dataset(
            source,
            num_episodes=1,
            frames_per_episode=3,
            cameras=["top", "gripper"],
        )

        mock_subprocess.return_value = MagicMock(returncode=0)
        mock_video_info.return_value = VideoInfo(
            num_frames=3,
            fps=30.0,
            width=640,
            height=480,
            duration_s=0.1,
            codec="h264",
        )

        with patch("lerobot.datasets.lerobot_dataset.LeRobotDataset", return_value=mock_dataset):
            converter = LeRobotToRawVideoConverter(source=source, dest=dest)
            converter.convert()

        # Touch video files that ffmpeg would have created (mocked out).
        for cam in ["gripper", "top"]:
            (dest / "episode_000" / f"{cam}.mp4").touch()

        manifest = load_manifest(dest)
        assert len(manifest.cameras) == 2
        camera_names = {c.name for c in manifest.cameras}
        assert camera_names == {"top", "gripper"}

        # ffmpeg called once per camera per episode = 2.
        assert mock_subprocess.call_count == 2

    @patch("internal_datasets.raw_video.converters.get_video_info")
    @patch("subprocess.run")
    def test_stream_copy_fallback_to_reencode(self, mock_subprocess, mock_video_info, tmp_path: Path):
        """If stream copy gives wrong frame count, falls back to re-encode."""
        source = tmp_path / "lerobot_source"
        source.mkdir()
        dest = tmp_path / "raw_output"

        mock_dataset = _make_mock_lerobot_dataset(
            source,
            num_episodes=1,
            frames_per_episode=5,
            cameras=["top"],
        )

        mock_subprocess.return_value = MagicMock(returncode=0)

        # First call to get_video_info returns wrong frame count (stream copy),
        # then subsequent calls return correct count (after re-encode).
        call_count = [0]

        def video_info_side_effect(path):
            call_count[0] += 1
            if call_count[0] == 1:
                # Stream copy produced wrong count.
                return VideoInfo(num_frames=7, fps=30.0, width=640, height=480, duration_s=0.23, codec="h264")
            return VideoInfo(num_frames=5, fps=30.0, width=640, height=480, duration_s=5 / 30.0, codec="h264")

        mock_video_info.side_effect = video_info_side_effect

        with patch("lerobot.datasets.lerobot_dataset.LeRobotDataset", return_value=mock_dataset):
            converter = LeRobotToRawVideoConverter(source=source, dest=dest)
            converter.convert()

        # ffmpeg should have been called twice: once stream copy, once re-encode.
        assert mock_subprocess.call_count == 2
        # First call should be stream copy (no -c:v libx264).
        first_call_cmd = mock_subprocess.call_args_list[0][0][0]
        assert "-c" in first_call_cmd and "copy" in first_call_cmd
        # Second call should be re-encode.
        second_call_cmd = mock_subprocess.call_args_list[1][0][0]
        assert "-c:v" in second_call_cmd and "libx264" in second_call_cmd

    def test_dest_already_exists(self, tmp_path: Path):
        """Should raise FileExistsError if dest already exists."""
        source = tmp_path / "lerobot_source"
        source.mkdir()
        (source / "meta").mkdir()
        (source / "meta" / "info.json").write_text("{}")

        dest = tmp_path / "raw_output"
        dest.mkdir()

        converter = LeRobotToRawVideoConverter(source=source, dest=dest)
        with pytest.raises(FileExistsError, match="Destination already exists"):
            converter.convert()

    def test_source_not_lerobot(self, tmp_path: Path):
        """Should raise FileNotFoundError if source has no meta/info.json."""
        source = tmp_path / "not_lerobot"
        source.mkdir()
        dest = tmp_path / "output"

        converter = LeRobotToRawVideoConverter(source=source, dest=dest)
        with pytest.raises(FileNotFoundError, match="missing"):
            converter.convert()

    @patch("internal_datasets.raw_video.converters.get_video_info")
    @patch("subprocess.run")
    def test_jsonl_data_values(self, mock_subprocess, mock_video_info, tmp_path: Path):
        """Verify JSONL data values match the source dataset."""
        source = tmp_path / "lerobot_source"
        source.mkdir()
        dest = tmp_path / "raw_output"

        mock_dataset = _make_mock_lerobot_dataset(
            source,
            num_episodes=1,
            frames_per_episode=3,
            cameras=["top"],
            state_dim=2,
            action_dim=2,
        )

        mock_subprocess.return_value = MagicMock(returncode=0)
        mock_video_info.return_value = VideoInfo(
            num_frames=3,
            fps=30.0,
            width=640,
            height=480,
            duration_s=0.1,
            codec="h264",
        )

        with patch("lerobot.datasets.lerobot_dataset.LeRobotDataset", return_value=mock_dataset):
            converter = LeRobotToRawVideoConverter(source=source, dest=dest)
            converter.convert()

        rows = _read_jsonl(dest / "episode_000" / "data.jsonl")
        # Frame 0 has global_idx=0, state=[0.0, 0.0], action=[0.0, 0.0].
        np.testing.assert_allclose(rows[0]["state"], [0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(rows[0]["action"], [0.0, 0.0], atol=1e-6)
        # Frame 2 has global_idx=2, state=[0.2, 0.2], action=[0.02, 0.02].
        np.testing.assert_allclose(rows[2]["state"], [0.2, 0.2], atol=1e-6)
        np.testing.assert_allclose(rows[2]["action"], [0.02, 0.02], atol=1e-6)


# ============================================================================
# RawVideoToLeRobotConverter tests
# ============================================================================


class TestRawVideoToLeRobotConverter:
    """Test conversion from raw-video format to LeRobot v3 format."""

    @patch("internal_datasets.raw_video.converters._decode_all_frames_as_numpy")
    @patch("internal_datasets.raw_video.converters.get_video_info")
    def test_basic_conversion(self, mock_video_info, mock_decode, tmp_path: Path):
        """Convert a 2-episode raw-video dataset to LeRobot format."""
        raw_dir = _make_raw_dataset(tmp_path, num_episodes=2, frames_per_episode=5, cameras=["top"])
        dest = tmp_path / "lerobot_output"

        mock_video_info.return_value = VideoInfo(
            num_frames=5,
            fps=30.0,
            width=640,
            height=480,
            duration_s=5 / 30.0,
            codec="h264",
        )
        # Return dummy uint8 frames: (N, H, W, 3).
        mock_decode.return_value = np.random.randint(0, 255, (5, 480, 640, 3), dtype=np.uint8)

        mock_lr_dataset = MagicMock()

        with patch("lerobot.datasets.lerobot_dataset.LeRobotDataset") as MockLRDataset:
            MockLRDataset.create.return_value = mock_lr_dataset
            converter = RawVideoToLeRobotConverter(source=raw_dir, dest=dest, robot_type="so100")
            converter.convert()

        # Verify LeRobotDataset.create was called correctly.
        create_call = MockLRDataset.create.call_args
        assert create_call.kwargs["fps"] == 30
        assert create_call.kwargs["robot_type"] == "so100"
        assert create_call.kwargs["use_videos"] is True
        assert "observation.images.top" in create_call.kwargs["features"]
        assert create_call.kwargs["features"]["observation.images.top"]["dtype"] == "video"

        # Verify joint names are preserved in features.
        assert create_call.kwargs["features"]["observation.state"]["names"] == [f"state_{i}" for i in range(6)]
        assert create_call.kwargs["features"]["action"]["names"] == [f"action_{i}" for i in range(6)]

        # add_frame should have been called 10 times total (5 per episode).
        assert mock_lr_dataset.add_frame.call_count == 10

        # save_episode should have been called twice.
        assert mock_lr_dataset.save_episode.call_count == 2

        # stop_image_writer and finalize must be called.
        mock_lr_dataset.stop_image_writer.assert_called_once()
        mock_lr_dataset.finalize.assert_called_once()

    @patch("internal_datasets.raw_video.converters._decode_all_frames_as_numpy")
    @patch("internal_datasets.raw_video.converters.get_video_info")
    def test_robot_type_from_manifest(self, mock_video_info, mock_decode, tmp_path: Path):
        """robot_type should come from the manifest when no CLI override is given."""
        raw_dir = _make_raw_dataset(tmp_path, num_episodes=1, frames_per_episode=2, robot_type="so101_follower")
        dest = tmp_path / "lerobot_output"

        mock_video_info.return_value = VideoInfo(
            num_frames=2, fps=30.0, width=640, height=480, duration_s=2 / 30.0, codec="h264"
        )
        mock_decode.return_value = np.random.randint(0, 255, (2, 480, 640, 3), dtype=np.uint8)

        mock_lr_dataset = MagicMock()

        with patch("lerobot.datasets.lerobot_dataset.LeRobotDataset") as MockLRDataset:
            MockLRDataset.create.return_value = mock_lr_dataset
            # No robot_type override — should use the manifest value.
            converter = RawVideoToLeRobotConverter(source=raw_dir, dest=dest)
            converter.convert()

        assert MockLRDataset.create.call_args.kwargs["robot_type"] == "so101_follower"

    @patch("internal_datasets.raw_video.converters._decode_all_frames_as_numpy")
    @patch("internal_datasets.raw_video.converters.get_video_info")
    def test_robot_type_cli_override(self, mock_video_info, mock_decode, tmp_path: Path):
        """CLI robot_type override should take precedence over the manifest value."""
        raw_dir = _make_raw_dataset(tmp_path, num_episodes=1, frames_per_episode=2, robot_type="so101_follower")
        dest = tmp_path / "lerobot_output"

        mock_video_info.return_value = VideoInfo(
            num_frames=2, fps=30.0, width=640, height=480, duration_s=2 / 30.0, codec="h264"
        )
        mock_decode.return_value = np.random.randint(0, 255, (2, 480, 640, 3), dtype=np.uint8)

        mock_lr_dataset = MagicMock()

        with patch("lerobot.datasets.lerobot_dataset.LeRobotDataset") as MockLRDataset:
            MockLRDataset.create.return_value = mock_lr_dataset
            converter = RawVideoToLeRobotConverter(source=raw_dir, dest=dest, robot_type="custom_arm")
            converter.convert()

        assert MockLRDataset.create.call_args.kwargs["robot_type"] == "custom_arm"

    @patch("internal_datasets.raw_video.converters._decode_all_frames_as_numpy")
    @patch("internal_datasets.raw_video.converters.get_video_info")
    def test_task_description_propagated(self, mock_video_info, mock_decode, tmp_path: Path):
        """Task description from manifest should be used as the task string."""
        raw_dir = _make_raw_dataset(
            tmp_path,
            num_episodes=1,
            frames_per_episode=2,
            task_description="stack the blocks",
        )
        dest = tmp_path / "lerobot_output"

        mock_video_info.return_value = VideoInfo(
            num_frames=2,
            fps=30.0,
            width=640,
            height=480,
            duration_s=2 / 30.0,
            codec="h264",
        )
        mock_decode.return_value = np.random.randint(0, 255, (2, 480, 640, 3), dtype=np.uint8)

        mock_lr_dataset = MagicMock()

        with patch("lerobot.datasets.lerobot_dataset.LeRobotDataset") as MockLRDataset:
            MockLRDataset.create.return_value = mock_lr_dataset
            converter = RawVideoToLeRobotConverter(source=raw_dir, dest=dest)
            converter.convert()

        # Every add_frame call should have task="stack the blocks".
        for frame_call in mock_lr_dataset.add_frame.call_args_list:
            frame = frame_call[0][0]
            assert frame["task"] == "stack the blocks"

    @patch("internal_datasets.raw_video.converters._decode_all_frames_as_numpy")
    @patch("internal_datasets.raw_video.converters.get_video_info")
    def test_default_task_when_empty(self, mock_video_info, mock_decode, tmp_path: Path):
        """If no task description, should use 'default'."""
        raw_dir = _make_raw_dataset(
            tmp_path,
            num_episodes=1,
            frames_per_episode=2,
            task_description="",
        )
        dest = tmp_path / "lerobot_output"

        mock_video_info.return_value = VideoInfo(
            num_frames=2,
            fps=30.0,
            width=640,
            height=480,
            duration_s=2 / 30.0,
            codec="h264",
        )
        mock_decode.return_value = np.random.randint(0, 255, (2, 480, 640, 3), dtype=np.uint8)

        mock_lr_dataset = MagicMock()

        with patch("lerobot.datasets.lerobot_dataset.LeRobotDataset") as MockLRDataset:
            MockLRDataset.create.return_value = mock_lr_dataset
            converter = RawVideoToLeRobotConverter(source=raw_dir, dest=dest)
            converter.convert()

        first_frame = mock_lr_dataset.add_frame.call_args_list[0][0][0]
        assert first_frame["task"] == "default"

    def test_dest_already_exists(self, tmp_path: Path):
        """Should raise FileExistsError if dest already exists."""
        raw_dir = _make_raw_dataset(tmp_path, num_episodes=1, frames_per_episode=2)
        dest = tmp_path / "lerobot_output"
        dest.mkdir()

        converter = RawVideoToLeRobotConverter(source=raw_dir, dest=dest)
        with pytest.raises(FileExistsError, match="Destination already exists"):
            converter.convert()

    def test_source_no_manifest(self, tmp_path: Path):
        """Should raise FileNotFoundError if no manifest.json."""
        source = tmp_path / "no_manifest"
        source.mkdir()
        dest = tmp_path / "output"

        converter = RawVideoToLeRobotConverter(source=source, dest=dest)
        with pytest.raises(FileNotFoundError, match="manifest.json"):
            converter.convert()

    @patch("internal_datasets.raw_video.converters._decode_all_frames_as_numpy")
    @patch("internal_datasets.raw_video.converters.get_video_info")
    def test_frame_data_types(self, mock_video_info, mock_decode, tmp_path: Path):
        """Verify add_frame receives correct numpy data types."""
        raw_dir = _make_raw_dataset(
            tmp_path,
            num_episodes=1,
            frames_per_episode=2,
            state_dim=3,
            action_dim=3,
            cameras=["top"],
        )
        dest = tmp_path / "lerobot_output"

        mock_video_info.return_value = VideoInfo(
            num_frames=2,
            fps=30.0,
            width=640,
            height=480,
            duration_s=2 / 30.0,
            codec="h264",
        )
        dummy_frames = np.random.randint(0, 255, (2, 480, 640, 3), dtype=np.uint8)
        mock_decode.return_value = dummy_frames

        mock_lr_dataset = MagicMock()

        with patch("lerobot.datasets.lerobot_dataset.LeRobotDataset") as MockLRDataset:
            MockLRDataset.create.return_value = mock_lr_dataset
            converter = RawVideoToLeRobotConverter(source=raw_dir, dest=dest)
            converter.convert()

        first_frame = mock_lr_dataset.add_frame.call_args_list[0][0][0]
        assert isinstance(first_frame["observation.state"], np.ndarray)
        assert first_frame["observation.state"].dtype == np.float32
        assert isinstance(first_frame["action"], np.ndarray)
        assert first_frame["action"].dtype == np.float32
        assert isinstance(first_frame["observation.images.top"], np.ndarray)
        assert first_frame["observation.images.top"].dtype == np.uint8

    @patch("internal_datasets.raw_video.converters._decode_all_frames_as_numpy")
    @patch("internal_datasets.raw_video.converters.get_video_info")
    def test_image_shape_in_features(self, mock_video_info, mock_decode, tmp_path: Path):
        """Verify image features use (H, W, C) shape order for LeRobot."""
        raw_dir = _make_raw_dataset(tmp_path, num_episodes=1, frames_per_episode=2, cameras=["cam"])
        dest = tmp_path / "lerobot_output"

        mock_video_info.return_value = VideoInfo(
            num_frames=2,
            fps=30.0,
            width=320,
            height=240,
            duration_s=2 / 30.0,
            codec="h264",
        )
        mock_decode.return_value = np.random.randint(0, 255, (2, 240, 320, 3), dtype=np.uint8)

        mock_lr_dataset = MagicMock()

        with patch("lerobot.datasets.lerobot_dataset.LeRobotDataset") as MockLRDataset:
            MockLRDataset.create.return_value = mock_lr_dataset
            converter = RawVideoToLeRobotConverter(source=raw_dir, dest=dest)
            converter.convert()

        features = MockLRDataset.create.call_args.kwargs["features"]
        assert features["observation.images.cam"]["shape"] == (240, 320, 3)
        assert features["observation.images.cam"]["names"] == ["height", "width", "channels"]


# ============================================================================
# CLI tests
# ============================================================================


class TestConverterCLI:
    """Test the click CLI entry points."""

    def test_convert_to_raw_help(self):
        from converter_cli import cli as converter_cli

        runner = CliRunner()
        result = runner.invoke(converter_cli, ["convert-to-raw", "--help"])
        assert result.exit_code == 0
        assert "--from" in result.output
        assert "--to" in result.output

    def test_convert_to_lerobot_help(self):
        from converter_cli import cli as converter_cli

        runner = CliRunner()
        result = runner.invoke(converter_cli, ["convert-to-lerobot", "--help"])
        assert result.exit_code == 0
        assert "--from" in result.output
        assert "--to" in result.output
        assert "--robot-type" in result.output

    def test_convert_to_raw_missing_source(self):
        from converter_cli import cli as converter_cli

        runner = CliRunner()
        result = runner.invoke(
            converter_cli,
            [
                "convert-to-raw",
                "--from",
                "/nonexistent/path",
                "--to",
                "/tmp/out",
            ],
        )
        assert result.exit_code != 0

    def test_convert_to_lerobot_missing_source(self):
        from converter_cli import cli as converter_cli

        runner = CliRunner()
        result = runner.invoke(
            converter_cli,
            [
                "convert-to-lerobot",
                "--from",
                "/nonexistent/path",
                "--to",
                "/tmp/out",
            ],
        )
        assert result.exit_code != 0
