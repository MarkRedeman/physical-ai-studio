# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for RawVideoDatasetClient, VideoWriter, save_manifest, and format auto-detection.

Tests cover:
- save_manifest(): atomic writes, round-trip with load_manifest
- VideoWriter: lifecycle (start/write/close), validation, abort, lazy start
- RawVideoDatasetClient: all 13 DatasetClient ABC methods
- get_internal_dataset(): format auto-detection based on sentinel files
- build_raw_video_manifest_args(): building features dict from environment
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from internal_datasets.raw_video.manifest import (
    CameraConfig,
    DatasetManifest,
    EpisodeEntry,
    load_manifest,
    save_manifest,
)
from internal_datasets.raw_video.raw_video_dataset_client import RawVideoDatasetClient, _read_jsonl
from internal_datasets.raw_video.video_writer import VideoWriter

# ============================================================================
# Helpers
# ============================================================================


def _make_manifest(
    *,
    name: str = "test_ds",
    fps: int = 30,
    state_dim: int = 3,
    action_dim: int = 3,
    cameras: list[str] | None = None,
    num_episodes: int = 0,
    robot_type: str = "test_robot",
    task_description: str = "",
) -> DatasetManifest:
    """Build a DatasetManifest for testing."""
    if cameras is None:
        cameras = ["top"]
    cam_configs = [CameraConfig(name=c) for c in cameras]
    episodes = []
    for i in range(num_episodes):
        ep_dir = f"ep_{i:03d}_abcdef{i:02d}"
        video_files = {c: f"cam_{c}.mp4" for c in cameras}
        episodes.append(EpisodeEntry(episode_dir=ep_dir, data_file="data.jsonl", video_files=video_files))
    return DatasetManifest(
        name=name,
        fps=fps,
        state_dim=state_dim,
        action_dim=action_dim,
        state_names=[f"joint_{i}" for i in range(state_dim)],
        action_names=[f"joint_{i}" for i in range(action_dim)],
        robot_type=robot_type,
        cameras=cam_configs,
        episodes=episodes,
        task_description=task_description,
    )


def _create_dataset_on_disk(
    root: Path,
    manifest: DatasetManifest,
    *,
    num_frames: int = 5,
) -> None:
    """Create a valid raw-video dataset on disk from a manifest.

    Writes manifest.json, episode directories, data.jsonl files, and
    dummy MP4 files (empty files — tests that need real video mock cv2).
    """
    root.mkdir(parents=True, exist_ok=True)
    save_manifest(root, manifest)

    for ep in manifest.episodes:
        ep_dir = root / ep.episode_dir
        ep_dir.mkdir(parents=True, exist_ok=True)

        # Write JSONL
        rows = []
        for t in range(num_frames):
            rows.append(
                {
                    "timestamp": round(t / manifest.fps, 6),
                    "state": [float(t + j) for j in range(manifest.state_dim)],
                    "action": [float(t + j + 0.1) for j in range(manifest.action_dim)],
                }
            )
        data_path = ep_dir / ep.data_file
        with data_path.open("w") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")

        # Write dummy video files
        for cam_name, video_filename in ep.video_files.items():
            (ep_dir / video_filename).write_bytes(b"\x00" * 100)


def _make_features(
    *,
    state_names: list[str] | None = None,
    action_names: list[str] | None = None,
    cameras: list[dict] | None = None,
) -> dict:
    """Build a features dict as returned by build_raw_video_manifest_args."""
    if state_names is None:
        state_names = ["joint_0", "joint_1", "joint_2"]
    if action_names is None:
        action_names = ["joint_0", "joint_1", "joint_2"]
    if cameras is None:
        cameras = [{"name": "top", "width": 640, "height": 480}]
    return {
        "state_names": state_names,
        "action_names": action_names,
        "cameras": cameras,
    }


# ============================================================================
# Tests: save_manifest
# ============================================================================


class TestSaveManifest:
    """Tests for the save_manifest() function."""

    def test_save_manifest_creates_file(self, tmp_path: Path) -> None:
        """save_manifest writes manifest.json to the dataset root."""
        manifest = _make_manifest()
        save_manifest(tmp_path, manifest)
        assert (tmp_path / "manifest.json").is_file()

    def test_save_manifest_creates_parent_dirs(self, tmp_path: Path) -> None:
        """save_manifest creates parent directories if they don't exist."""
        nested = tmp_path / "a" / "b" / "c"
        manifest = _make_manifest()
        save_manifest(nested, manifest)
        assert (nested / "manifest.json").is_file()

    def test_save_manifest_roundtrip(self, tmp_path: Path) -> None:
        """save_manifest output can be read back by load_manifest."""
        manifest = _make_manifest(num_episodes=1)
        # Create the episode dir and files so load_manifest validation passes
        _create_dataset_on_disk(tmp_path, manifest)

        loaded = load_manifest(tmp_path)
        assert loaded.name == manifest.name
        assert loaded.fps == manifest.fps
        assert loaded.state_dim == manifest.state_dim
        assert loaded.action_dim == manifest.action_dim
        assert loaded.state_names == manifest.state_names
        assert loaded.action_names == manifest.action_names
        assert loaded.robot_type == manifest.robot_type
        assert len(loaded.episodes) == len(manifest.episodes)
        assert len(loaded.cameras) == len(manifest.cameras)

    def test_save_manifest_is_valid_json(self, tmp_path: Path) -> None:
        """The written file is valid JSON with the expected keys."""
        manifest = _make_manifest()
        save_manifest(tmp_path, manifest)
        data = json.loads((tmp_path / "manifest.json").read_text())
        assert data["name"] == "test_ds"
        assert data["fps"] == 30
        assert "state_names" in data
        assert "action_names" in data
        assert "cameras" in data
        assert "episodes" in data

    def test_save_manifest_overwrites_existing(self, tmp_path: Path) -> None:
        """save_manifest overwrites an existing manifest.json."""
        m1 = _make_manifest(name="first")
        save_manifest(tmp_path, m1)
        m2 = _make_manifest(name="second")
        save_manifest(tmp_path, m2)
        data = json.loads((tmp_path / "manifest.json").read_text())
        assert data["name"] == "second"

    def test_save_manifest_no_temp_files_left(self, tmp_path: Path) -> None:
        """No temporary files should remain after save_manifest completes."""
        manifest = _make_manifest()
        save_manifest(tmp_path, manifest)
        files = list(tmp_path.iterdir())
        assert len(files) == 1
        assert files[0].name == "manifest.json"


# ============================================================================
# Tests: VideoWriter
# ============================================================================


class TestVideoWriter:
    """Tests for the VideoWriter class."""

    def test_init_properties(self, tmp_path: Path) -> None:
        """VideoWriter stores properties correctly after init."""
        out = tmp_path / "test.mp4"
        writer = VideoWriter(output_path=out, fps=30, width=640, height=480)
        assert writer.output_path == out
        assert writer.frame_count == 0
        assert not writer.is_running

    def test_start_requires_parent_dir(self, tmp_path: Path) -> None:
        """start() raises FileNotFoundError if parent dir doesn't exist."""
        out = tmp_path / "nonexistent" / "test.mp4"
        writer = VideoWriter(output_path=out, fps=30, width=640, height=480)
        with pytest.raises(FileNotFoundError, match="Output directory does not exist"):
            writer.start()

    def test_start_twice_raises(self, tmp_path: Path) -> None:
        """Calling start() twice raises RuntimeError."""
        out = tmp_path / "test.mp4"
        writer = VideoWriter(output_path=out, fps=30, width=640, height=480)
        writer.start()
        try:
            with pytest.raises(RuntimeError, match="already started"):
                writer.start()
        finally:
            writer.abort()

    def test_write_frame_before_start_raises(self, tmp_path: Path) -> None:
        """write_frame() before start() raises RuntimeError."""
        out = tmp_path / "test.mp4"
        writer = VideoWriter(output_path=out, fps=30, width=640, height=480)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="not started"):
            writer.write_frame(frame)

    def test_write_frame_wrong_shape_raises(self, tmp_path: Path) -> None:
        """write_frame() with wrong shape raises ValueError."""
        out = tmp_path / "test.mp4"
        writer = VideoWriter(output_path=out, fps=30, width=640, height=480)
        writer.start()
        try:
            wrong_frame = np.zeros((100, 100, 3), dtype=np.uint8)
            with pytest.raises(ValueError, match="Frame shape"):
                writer.write_frame(wrong_frame)
        finally:
            writer.abort()

    def test_write_frame_wrong_dtype_raises(self, tmp_path: Path) -> None:
        """write_frame() with wrong dtype raises ValueError."""
        out = tmp_path / "test.mp4"
        writer = VideoWriter(output_path=out, fps=30, width=640, height=480)
        writer.start()
        try:
            bad_frame = np.zeros((480, 640, 3), dtype=np.float32)
            with pytest.raises(ValueError, match="dtype must be uint8"):
                writer.write_frame(bad_frame)
        finally:
            writer.abort()

    def test_write_and_close_produces_file(self, tmp_path: Path) -> None:
        """Writing frames and closing produces an MP4 file on disk."""
        out = tmp_path / "test.mp4"
        writer = VideoWriter(output_path=out, fps=30, width=64, height=48)
        writer.start()
        for _ in range(10):
            frame = np.random.randint(0, 255, (48, 64, 3), dtype=np.uint8)
            writer.write_frame(frame)
        writer.close()
        assert out.is_file()
        assert out.stat().st_size > 0
        assert writer.frame_count == 10
        assert not writer.is_running

    def test_close_without_start_is_noop(self, tmp_path: Path) -> None:
        """close() on an unstarted writer does nothing."""
        out = tmp_path / "test.mp4"
        writer = VideoWriter(output_path=out, fps=30, width=640, height=480)
        writer.close()  # should not raise
        assert not writer.is_running

    def test_abort_without_start_is_noop(self, tmp_path: Path) -> None:
        """abort() on an unstarted writer does nothing."""
        out = tmp_path / "test.mp4"
        writer = VideoWriter(output_path=out, fps=30, width=640, height=480)
        writer.abort()  # should not raise
        assert not writer.is_running

    def test_abort_kills_process(self, tmp_path: Path) -> None:
        """abort() terminates the ffmpeg process."""
        out = tmp_path / "test.mp4"
        writer = VideoWriter(output_path=out, fps=30, width=64, height=48)
        writer.start()
        assert writer.is_running
        writer.abort()
        assert not writer.is_running

    def test_is_running_reflects_process_state(self, tmp_path: Path) -> None:
        """is_running tracks the ffmpeg subprocess lifecycle."""
        out = tmp_path / "test.mp4"
        writer = VideoWriter(output_path=out, fps=30, width=64, height=48)
        assert not writer.is_running
        writer.start()
        assert writer.is_running
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        writer.write_frame(frame)
        writer.close()
        assert not writer.is_running


# ============================================================================
# Tests: _read_jsonl helper
# ============================================================================


class TestReadJsonl:
    """Tests for the _read_jsonl helper function."""

    def test_read_jsonl_basic(self, tmp_path: Path) -> None:
        """Read a simple JSONL file."""
        path = tmp_path / "data.jsonl"
        rows = [{"a": 1}, {"a": 2}, {"a": 3}]
        with path.open("w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        result = _read_jsonl(path)
        assert result == rows

    def test_read_jsonl_empty_lines_skipped(self, tmp_path: Path) -> None:
        """Empty lines in JSONL are skipped."""
        path = tmp_path / "data.jsonl"
        path.write_text('{"a": 1}\n\n{"a": 2}\n\n')
        result = _read_jsonl(path)
        assert len(result) == 2

    def test_read_jsonl_empty_file(self, tmp_path: Path) -> None:
        """An empty JSONL file returns an empty list."""
        path = tmp_path / "data.jsonl"
        path.write_text("")
        result = _read_jsonl(path)
        assert result == []


# ============================================================================
# Tests: RawVideoDatasetClient
# ============================================================================


class TestRawVideoDatasetClientInit:
    """Tests for RawVideoDatasetClient initialization."""

    def test_init_nonexistent_path(self, tmp_path: Path) -> None:
        """Client initializes without error for non-existent path."""
        ds_path = tmp_path / "nonexistent"
        client = RawVideoDatasetClient(ds_path)
        assert not client.exists_on_disk
        assert not client.has_episodes

    def test_init_existing_dataset(self, tmp_path: Path) -> None:
        """Client loads manifest when initialized with existing dataset."""
        manifest = _make_manifest(num_episodes=1)
        _create_dataset_on_disk(tmp_path, manifest)
        client = RawVideoDatasetClient(tmp_path)
        assert client.exists_on_disk
        assert client.has_episodes

    def test_init_existing_empty_dataset(self, tmp_path: Path) -> None:
        """Client loads manifest for dataset with zero episodes."""
        manifest = _make_manifest(num_episodes=0)
        save_manifest(tmp_path, manifest)
        client = RawVideoDatasetClient(tmp_path)
        assert client.exists_on_disk
        assert not client.has_episodes


class TestRawVideoDatasetClientCreate:
    """Tests for the create() method."""

    def test_create_new_dataset(self, tmp_path: Path) -> None:
        """create() builds manifest.json and creates dataset directory."""
        ds_path = tmp_path / "new_ds"
        client = RawVideoDatasetClient(ds_path)
        features = _make_features()
        client.create(fps=30, features=features, robot_type="test_robot")

        assert client.exists_on_disk
        assert not client.has_episodes
        assert (ds_path / "manifest.json").is_file()

        # Verify manifest content
        data = json.loads((ds_path / "manifest.json").read_text())
        assert data["fps"] == 30
        assert data["state_dim"] == 3
        assert data["action_dim"] == 3
        assert data["robot_type"] == "test_robot"
        assert data["state_names"] == ["joint_0", "joint_1", "joint_2"]
        assert len(data["cameras"]) == 1
        assert data["cameras"][0]["name"] == "top"

    def test_create_already_exists_raises(self, tmp_path: Path) -> None:
        """create() raises if dataset already exists."""
        manifest = _make_manifest()
        save_manifest(tmp_path, manifest)
        client = RawVideoDatasetClient(tmp_path)
        with pytest.raises(RuntimeError, match="already exists"):
            client.create(fps=30, features=_make_features(), robot_type="test_robot")

    def test_create_multiple_cameras(self, tmp_path: Path) -> None:
        """create() handles multiple cameras."""
        ds_path = tmp_path / "multi_cam"
        client = RawVideoDatasetClient(ds_path)
        features = _make_features(
            cameras=[
                {"name": "top", "width": 640, "height": 480},
                {"name": "gripper", "width": 320, "height": 240},
            ]
        )
        client.create(fps=30, features=features, robot_type="test_robot")
        data = json.loads((ds_path / "manifest.json").read_text())
        assert len(data["cameras"]) == 2
        cam_names = [c["name"] for c in data["cameras"]]
        assert "top" in cam_names
        assert "gripper" in cam_names


class TestRawVideoDatasetClientRead:
    """Tests for get_episodes(), get_tasks(), get_video_path()."""

    def test_get_episodes_empty_dataset(self, tmp_path: Path) -> None:
        """get_episodes() returns empty list for dataset with no episodes."""
        manifest = _make_manifest(num_episodes=0)
        save_manifest(tmp_path, manifest)
        client = RawVideoDatasetClient(tmp_path)
        assert client.get_episodes() == []

    def test_get_episodes_returns_correct_count(self, tmp_path: Path) -> None:
        """get_episodes() returns one Episode per manifest entry."""
        manifest = _make_manifest(num_episodes=3)
        _create_dataset_on_disk(tmp_path, manifest, num_frames=5)
        # Patch cv2.VideoCapture to avoid real video decode for thumbnails
        with patch("internal_datasets.raw_video.raw_video_dataset_client.cv2") as mock_cv2:
            mock_cap = MagicMock()
            mock_cap.read.return_value = (False, None)
            mock_cv2.VideoCapture.return_value = mock_cap
            client = RawVideoDatasetClient(tmp_path)
            episodes = client.get_episodes()
        assert len(episodes) == 3

    def test_get_episodes_content(self, tmp_path: Path) -> None:
        """get_episodes() returns Episode objects with correct attributes."""
        manifest = _make_manifest(num_episodes=1, task_description="pick up block")
        _create_dataset_on_disk(tmp_path, manifest, num_frames=10)
        with patch("internal_datasets.raw_video.raw_video_dataset_client.cv2") as mock_cv2:
            mock_cap = MagicMock()
            mock_cap.read.return_value = (False, None)
            mock_cv2.VideoCapture.return_value = mock_cap
            client = RawVideoDatasetClient(tmp_path)
            episodes = client.get_episodes()

        ep = episodes[0]
        assert ep.episode_index == 0
        assert ep.length == 10
        assert ep.fps == 30
        assert ep.tasks == ["pick up block"]
        assert len(ep.actions) == 10
        assert ep.action_keys == ["joint_0", "joint_1", "joint_2"]
        assert "top" in ep.videos

    def test_get_episodes_no_manifest(self, tmp_path: Path) -> None:
        """get_episodes() returns empty list when manifest is None."""
        client = RawVideoDatasetClient(tmp_path / "nonexistent")
        assert client.get_episodes() == []

    def test_get_tasks_with_description(self, tmp_path: Path) -> None:
        """get_tasks() returns list with task_description."""
        manifest = _make_manifest(task_description="pick up block")
        save_manifest(tmp_path, manifest)
        client = RawVideoDatasetClient(tmp_path)
        assert client.get_tasks() == ["pick up block"]

    def test_get_tasks_without_description(self, tmp_path: Path) -> None:
        """get_tasks() returns empty list when no task_description."""
        manifest = _make_manifest(task_description="")
        save_manifest(tmp_path, manifest)
        client = RawVideoDatasetClient(tmp_path)
        assert client.get_tasks() == []

    def test_get_tasks_no_manifest(self, tmp_path: Path) -> None:
        """get_tasks() returns empty list when manifest is None."""
        client = RawVideoDatasetClient(tmp_path / "nonexistent")
        assert client.get_tasks() == []

    def test_get_video_path(self, tmp_path: Path) -> None:
        """get_video_path() returns the correct absolute path."""
        manifest = _make_manifest(num_episodes=1, cameras=["top", "gripper"])
        _create_dataset_on_disk(tmp_path, manifest)
        client = RawVideoDatasetClient(tmp_path)
        path = client.get_video_path(0, "top")
        assert path == tmp_path / manifest.episodes[0].episode_dir / "cam_top.mp4"

    def test_get_video_path_invalid_episode(self, tmp_path: Path) -> None:
        """get_video_path() raises IndexError for out-of-range episode."""
        manifest = _make_manifest(num_episodes=1)
        _create_dataset_on_disk(tmp_path, manifest)
        client = RawVideoDatasetClient(tmp_path)
        with pytest.raises(IndexError, match="out of range"):
            client.get_video_path(5, "top")

    def test_get_video_path_invalid_camera(self, tmp_path: Path) -> None:
        """get_video_path() raises KeyError for unknown camera."""
        manifest = _make_manifest(num_episodes=1)
        _create_dataset_on_disk(tmp_path, manifest)
        client = RawVideoDatasetClient(tmp_path)
        with pytest.raises(KeyError, match="not found"):
            client.get_video_path(0, "nonexistent_cam")

    def test_get_video_path_no_manifest(self, tmp_path: Path) -> None:
        """get_video_path() raises RuntimeError when no dataset loaded."""
        client = RawVideoDatasetClient(tmp_path / "nonexistent")
        with pytest.raises(RuntimeError, match="not loaded"):
            client.get_video_path(0, "top")


class TestRawVideoDatasetClientRecording:
    """Tests for the recording flow: prepare_for_writing, add_frame, save_episode, discard_buffer."""

    def _setup_client(self, tmp_path: Path, cameras: list[str] | None = None) -> RawVideoDatasetClient:
        """Create a client with a dataset ready for recording."""
        if cameras is None:
            cameras = ["top"]
        ds_path = tmp_path / "ds"
        client = RawVideoDatasetClient(ds_path)
        features = _make_features(cameras=[{"name": c, "width": 64, "height": 48} for c in cameras])
        client.create(fps=30, features=features, robot_type="test_robot")
        return client

    def test_prepare_for_writing_creates_episode_dir(self, tmp_path: Path) -> None:
        """prepare_for_writing() creates a new episode directory."""
        client = self._setup_client(tmp_path)
        client.prepare_for_writing()
        # Episode dir should exist
        assert client._episode_dir is not None
        assert client._episode_dir.is_dir()
        assert client._episode_dir.name.startswith("ep_000_")

    def test_prepare_for_writing_no_manifest_raises(self, tmp_path: Path) -> None:
        """prepare_for_writing() raises when no manifest loaded."""
        client = RawVideoDatasetClient(tmp_path / "nonexistent")
        with pytest.raises(RuntimeError, match="not created"):
            client.prepare_for_writing()

    def test_add_frame_no_session_raises(self, tmp_path: Path) -> None:
        """add_frame() raises if not in a recording session."""
        client = self._setup_client(tmp_path)
        obs = {"top": np.zeros((48, 64, 3), dtype=np.uint8), "joint_0": 0.1, "joint_1": 0.2, "joint_2": 0.3}
        act = {"joint_0": 0.1, "joint_1": 0.2, "joint_2": 0.3}
        with pytest.raises(RuntimeError, match="Not in a recording session"):
            client.add_frame(obs, act, "test")

    def test_add_frame_missing_camera_raises(self, tmp_path: Path) -> None:
        """add_frame() raises if a camera frame is missing from observations."""
        client = self._setup_client(tmp_path)
        client.prepare_for_writing()
        # obs missing 'top' camera frame
        obs: dict = {"joint_0": 0.1, "joint_1": 0.2, "joint_2": 0.3}
        act = {"joint_0": 0.1, "joint_1": 0.2, "joint_2": 0.3}
        try:
            with pytest.raises(ValueError, match="Missing camera frame"):
                client.add_frame(obs, act, "test")
        finally:
            client.discard_buffer()

    def test_full_recording_cycle(self, tmp_path: Path) -> None:
        """Record frames, save, verify episode on disk."""
        client = self._setup_client(tmp_path)
        client.prepare_for_writing()

        # Add 5 frames
        for i in range(5):
            frame = np.random.randint(0, 255, (48, 64, 3), dtype=np.uint8)
            obs = {
                "top": frame,
                "joint_0": float(i),
                "joint_1": float(i + 1),
                "joint_2": float(i + 2),
            }
            act = {
                "joint_0": float(i + 0.1),
                "joint_1": float(i + 1.1),
                "joint_2": float(i + 2.1),
            }
            client.add_frame(obs, act, "pick block")

        episode = client.save_episode("pick block")

        # Verify returned Episode
        assert episode.episode_index == 0
        assert episode.length == 5
        assert episode.fps == 30
        assert episode.tasks == ["pick block"]
        assert len(episode.actions) == 5
        assert "top" in episode.videos

        # Verify on disk
        assert client.has_episodes
        manifest = load_manifest(client.path)
        assert len(manifest.episodes) == 1
        assert manifest.task_description == "pick block"

        # Verify JSONL
        ep_dir = client.path / manifest.episodes[0].episode_dir
        rows = _read_jsonl(ep_dir / "data.jsonl")
        assert len(rows) == 5
        assert rows[0]["state"] == [0.0, 1.0, 2.0]
        assert len(rows[0]["action"]) == 3

        # Verify video file exists
        video_path = ep_dir / "cam_top.mp4"
        assert video_path.is_file()
        assert video_path.stat().st_size > 0

    def test_recording_multiple_episodes(self, tmp_path: Path) -> None:
        """Record two episodes sequentially."""
        client = self._setup_client(tmp_path)

        # Episode 0
        client.prepare_for_writing()
        for i in range(3):
            frame = np.random.randint(0, 255, (48, 64, 3), dtype=np.uint8)
            obs = {"top": frame, "joint_0": 0.0, "joint_1": 0.0, "joint_2": 0.0}
            act = {"joint_0": 0.0, "joint_1": 0.0, "joint_2": 0.0}
            client.add_frame(obs, act, "task A")
        ep0 = client.save_episode("task A")
        assert ep0.episode_index == 0

        # Episode 1
        client.prepare_for_writing()
        for i in range(4):
            frame = np.random.randint(0, 255, (48, 64, 3), dtype=np.uint8)
            obs = {"top": frame, "joint_0": 1.0, "joint_1": 1.0, "joint_2": 1.0}
            act = {"joint_0": 1.0, "joint_1": 1.0, "joint_2": 1.0}
            client.add_frame(obs, act, "task A")
        ep1 = client.save_episode("task A")
        assert ep1.episode_index == 1

        # Verify manifest has 2 episodes
        manifest = load_manifest(client.path)
        assert len(manifest.episodes) == 2

    def test_recording_multiple_cameras(self, tmp_path: Path) -> None:
        """Record with multiple camera streams."""
        client = self._setup_client(tmp_path, cameras=["top", "gripper"])
        client.prepare_for_writing()

        for i in range(3):
            frame_top = np.random.randint(0, 255, (48, 64, 3), dtype=np.uint8)
            frame_grip = np.random.randint(0, 255, (48, 64, 3), dtype=np.uint8)
            obs = {
                "top": frame_top,
                "gripper": frame_grip,
                "joint_0": 0.0,
                "joint_1": 0.0,
                "joint_2": 0.0,
            }
            act = {"joint_0": 0.0, "joint_1": 0.0, "joint_2": 0.0}
            client.add_frame(obs, act, "multi cam test")

        episode = client.save_episode("multi cam test")
        assert "top" in episode.videos
        assert "gripper" in episode.videos

        # Both video files should exist
        manifest = load_manifest(client.path)
        ep_dir = client.path / manifest.episodes[0].episode_dir
        assert (ep_dir / "cam_top.mp4").is_file()
        assert (ep_dir / "cam_gripper.mp4").is_file()

    def test_discard_buffer_removes_episode_dir(self, tmp_path: Path) -> None:
        """discard_buffer() removes the episode directory."""
        client = self._setup_client(tmp_path)
        client.prepare_for_writing()
        ep_dir = client._episode_dir
        assert ep_dir is not None and ep_dir.is_dir()

        # Add a frame so video writers are started
        frame = np.random.randint(0, 255, (48, 64, 3), dtype=np.uint8)
        obs = {"top": frame, "joint_0": 0.0, "joint_1": 0.0, "joint_2": 0.0}
        act = {"joint_0": 0.0, "joint_1": 0.0, "joint_2": 0.0}
        client.add_frame(obs, act, "test")

        client.discard_buffer()
        assert not ep_dir.is_dir()
        assert client._episode_dir is None
        assert not client.has_episodes

    def test_discard_buffer_noop_when_no_session(self, tmp_path: Path) -> None:
        """discard_buffer() is a no-op when there's no active session."""
        client = self._setup_client(tmp_path)
        client.discard_buffer()  # should not raise

    def test_save_episode_no_session_raises(self, tmp_path: Path) -> None:
        """save_episode() raises when not in recording session."""
        client = self._setup_client(tmp_path)
        with pytest.raises(RuntimeError, match="No episode to save"):
            client.save_episode("test")

    def test_state_action_vector_ordering(self, tmp_path: Path) -> None:
        """State and action vectors follow manifest name ordering."""
        ds_path = tmp_path / "ds"
        client = RawVideoDatasetClient(ds_path)
        features = _make_features(
            state_names=["alpha", "beta", "gamma"],
            action_names=["alpha", "beta", "gamma"],
        )
        client.create(fps=30, features=features, robot_type="test_robot")
        client.prepare_for_writing()

        frame = np.random.randint(0, 255, (48, 64, 3), dtype=np.uint8)
        obs = {"top": frame, "alpha": 1.0, "beta": 2.0, "gamma": 3.0}
        act = {"gamma": 30.0, "alpha": 10.0, "beta": 20.0}  # out of order
        client.add_frame(obs, act, "test")
        episode = client.save_episode("test")

        # Actions should be in manifest order [alpha, beta, gamma]
        assert episode.actions[0] == [10.0, 20.0, 30.0]

    def test_missing_state_key_defaults_to_zero(self, tmp_path: Path) -> None:
        """Missing observation keys default to 0.0 in the state vector."""
        client = self._setup_client(tmp_path)
        client.prepare_for_writing()

        frame = np.random.randint(0, 255, (48, 64, 3), dtype=np.uint8)
        # Only provide joint_0, missing joint_1 and joint_2
        obs: dict[str, Any] = {"top": frame, "joint_0": 5.0}
        act: dict[str, Any] = {"joint_0": 5.0}
        client.add_frame(obs, act, "test")
        episode = client.save_episode("test")

        # State should be [5.0, 0.0, 0.0]
        manifest = load_manifest(client.path)
        ep_dir = client.path / manifest.episodes[0].episode_dir
        rows = _read_jsonl(ep_dir / "data.jsonl")
        assert rows[0]["state"] == [5.0, 0.0, 0.0]
        assert rows[0]["action"] == [5.0, 0.0, 0.0]


class TestRawVideoDatasetClientDelete:
    """Tests for delete_episodes() and delete()."""

    def test_delete_episodes_copies_remaining(self, tmp_path: Path) -> None:
        """delete_episodes() copies dataset excluding specified episodes."""
        manifest = _make_manifest(num_episodes=3)
        _create_dataset_on_disk(tmp_path / "src", manifest, num_frames=5)
        client = RawVideoDatasetClient(tmp_path / "src")

        output_path = tmp_path / "dst"
        new_client = client.delete_episodes([1], output_path)

        assert isinstance(new_client, RawVideoDatasetClient)
        new_manifest = load_manifest(output_path)
        assert len(new_manifest.episodes) == 2
        # Episode dirs from original indices 0 and 2 should be present
        ep_dirs = {ep.episode_dir for ep in new_manifest.episodes}
        assert manifest.episodes[0].episode_dir in ep_dirs
        assert manifest.episodes[2].episode_dir in ep_dirs
        assert manifest.episodes[1].episode_dir not in ep_dirs

    def test_delete_episodes_all(self, tmp_path: Path) -> None:
        """delete_episodes() with all indices results in empty dataset."""
        manifest = _make_manifest(num_episodes=2)
        _create_dataset_on_disk(tmp_path / "src", manifest, num_frames=3)
        client = RawVideoDatasetClient(tmp_path / "src")

        output_path = tmp_path / "dst"
        new_client = client.delete_episodes([0, 1], output_path)
        assert not new_client.has_episodes

    def test_delete_episodes_no_manifest_raises(self, tmp_path: Path) -> None:
        """delete_episodes() raises when manifest is None."""
        client = RawVideoDatasetClient(tmp_path / "nonexistent")
        with pytest.raises(RuntimeError, match="not loaded"):
            client.delete_episodes([0], tmp_path / "dst")

    def test_delete_removes_directory(self, tmp_path: Path) -> None:
        """delete() removes the entire dataset directory."""
        manifest = _make_manifest(num_episodes=1)
        _create_dataset_on_disk(tmp_path, manifest)
        client = RawVideoDatasetClient(tmp_path)
        assert tmp_path.is_dir()

        client.delete()
        assert not tmp_path.is_dir()
        assert not client.exists_on_disk

    def test_delete_nonexistent_is_noop(self, tmp_path: Path) -> None:
        """delete() on non-existent path doesn't raise."""
        client = RawVideoDatasetClient(tmp_path / "nonexistent")
        client.delete()  # should not raise


class TestRawVideoDatasetClientLifecycle:
    """Tests for teardown(), finalize(), overwrite(), start_recording_mutation()."""

    def test_teardown_deletes_empty_dataset(self, tmp_path: Path) -> None:
        """teardown() deletes dataset if it has no episodes."""
        manifest = _make_manifest(num_episodes=0)
        save_manifest(tmp_path, manifest)
        client = RawVideoDatasetClient(tmp_path)
        client.teardown()
        assert not tmp_path.is_dir()

    def test_teardown_finalizes_nonempty_dataset(self, tmp_path: Path) -> None:
        """teardown() calls finalize() if dataset has episodes."""
        manifest = _make_manifest(num_episodes=1)
        _create_dataset_on_disk(tmp_path, manifest)
        client = RawVideoDatasetClient(tmp_path)
        client.teardown()
        # Dataset should still exist
        assert tmp_path.is_dir()
        assert (tmp_path / "manifest.json").is_file()

    def test_finalize_invalidates_stats_cache(self, tmp_path: Path) -> None:
        """finalize() removes .cache/stats.json if it exists."""
        manifest = _make_manifest(num_episodes=1)
        _create_dataset_on_disk(tmp_path, manifest)
        cache_dir = tmp_path / ".cache"
        cache_dir.mkdir()
        stats_file = cache_dir / "stats.json"
        stats_file.write_text("{}")

        client = RawVideoDatasetClient(tmp_path)
        client.finalize()
        assert not stats_file.is_file()

    def test_finalize_no_stats_cache_no_error(self, tmp_path: Path) -> None:
        """finalize() doesn't error if there's no stats cache."""
        manifest = _make_manifest()
        save_manifest(tmp_path, manifest)
        client = RawVideoDatasetClient(tmp_path)
        client.finalize()  # should not raise

    def test_overwrite_copies_source(self, tmp_path: Path) -> None:
        """overwrite() replaces this dataset with the source dataset."""
        # Create source dataset
        src_manifest = _make_manifest(name="source", num_episodes=1)
        _create_dataset_on_disk(tmp_path / "src", src_manifest)
        src_client = RawVideoDatasetClient(tmp_path / "src")

        # Create target dataset (initially different)
        tgt_manifest = _make_manifest(name="target", num_episodes=0)
        save_manifest(tmp_path / "tgt", tgt_manifest)
        tgt_client = RawVideoDatasetClient(tmp_path / "tgt")

        tgt_client.overwrite(src_client)

        # Target should now match source
        new_manifest = load_manifest(tmp_path / "tgt")
        assert new_manifest.name == "source"
        assert len(new_manifest.episodes) == 1

    def test_overwrite_wrong_type_raises(self, tmp_path: Path) -> None:
        """overwrite() raises if source is not RawVideoDatasetClient."""
        manifest = _make_manifest()
        save_manifest(tmp_path, manifest)
        client = RawVideoDatasetClient(tmp_path)
        mock_source = MagicMock(spec=[])
        mock_source.__class__.__name__ = "FakeClient"
        with pytest.raises(ValueError, match="Cannot overwrite"):
            client.overwrite(mock_source)

    def test_start_recording_mutation_new_dataset(self, tmp_path: Path) -> None:
        """start_recording_mutation() creates cache dataset for new dataset."""
        ds_path = tmp_path / "ds"
        client = RawVideoDatasetClient(ds_path)
        features = _make_features()

        with patch("internal_datasets.raw_video.raw_video_dataset_client.get_settings") as mock_settings:
            settings = MagicMock()
            settings.cache_dir = tmp_path / "cache"
            mock_settings.return_value = settings

            mutation = client.start_recording_mutation(fps=30, features=features, robot_type="test_robot")

        from internal_datasets.mutations.recording_mutation import RecordingMutation

        assert isinstance(mutation, RecordingMutation)
        assert mutation.source_dataset is client
        assert isinstance(mutation.cache_dataset, RawVideoDatasetClient)
        assert mutation.cache_dataset.exists_on_disk

    def test_start_recording_mutation_existing_dataset(self, tmp_path: Path) -> None:
        """start_recording_mutation() copies existing dataset to cache."""
        manifest = _make_manifest(num_episodes=1)
        ds_path = tmp_path / "ds"
        _create_dataset_on_disk(ds_path, manifest)
        client = RawVideoDatasetClient(ds_path)
        features = _make_features()

        with patch("internal_datasets.raw_video.raw_video_dataset_client.get_settings") as mock_settings:
            settings = MagicMock()
            settings.cache_dir = tmp_path / "cache"
            mock_settings.return_value = settings

            mutation = client.start_recording_mutation(fps=30, features=features, robot_type="test_robot")

        cache_client = mutation.cache_dataset
        assert isinstance(cache_client, RawVideoDatasetClient)
        # Cache should have the existing episode
        assert cache_client.has_episodes


# ============================================================================
# Tests: Format auto-detection (get_internal_dataset)
# ============================================================================


class TestFormatAutoDetection:
    """Tests for get_internal_dataset() format auto-detection."""

    def test_detects_raw_video_format(self, tmp_path: Path) -> None:
        """Detects raw-video format when manifest.json is present."""
        from internal_datasets.utils import get_internal_dataset
        from schemas import Dataset

        manifest = _make_manifest()
        save_manifest(tmp_path, manifest)

        dataset = MagicMock(spec=Dataset)
        dataset.path = str(tmp_path)

        result = get_internal_dataset(dataset)
        assert isinstance(result, RawVideoDatasetClient)

    def test_detects_lerobot_format(self, tmp_path: Path) -> None:
        """Detects LeRobot format when meta/info.json is present."""
        from internal_datasets.utils import get_internal_dataset
        from schemas import Dataset

        meta_dir = tmp_path / "meta"
        meta_dir.mkdir()
        (meta_dir / "info.json").write_text("{}")

        dataset = MagicMock(spec=Dataset)
        dataset.path = str(tmp_path)

        # Patch InternalLeRobotDataset to avoid actual LeRobot initialization
        with patch("internal_datasets.utils.InternalLeRobotDataset") as MockLerobot:
            mock_instance = MagicMock()
            MockLerobot.return_value = mock_instance
            result = get_internal_dataset(dataset)
            MockLerobot.assert_called_once_with(tmp_path)
            assert result is mock_instance

    def test_defaults_to_raw_video_for_new_path(self, tmp_path: Path) -> None:
        """Defaults to RawVideoDatasetClient for non-existent path."""
        from internal_datasets.utils import get_internal_dataset
        from schemas import Dataset

        dataset = MagicMock(spec=Dataset)
        dataset.path = str(tmp_path / "new_dataset")

        result = get_internal_dataset(dataset)
        assert isinstance(result, RawVideoDatasetClient)

    def test_raw_video_takes_precedence(self, tmp_path: Path) -> None:
        """If both sentinel files exist, manifest.json (raw-video) wins."""
        from internal_datasets.utils import get_internal_dataset
        from schemas import Dataset

        manifest = _make_manifest()
        save_manifest(tmp_path, manifest)
        meta_dir = tmp_path / "meta"
        meta_dir.mkdir()
        (meta_dir / "info.json").write_text("{}")

        dataset = MagicMock(spec=Dataset)
        dataset.path = str(tmp_path)

        result = get_internal_dataset(dataset)
        assert isinstance(result, RawVideoDatasetClient)


# ============================================================================
# Tests: build_raw_video_manifest_args
# ============================================================================


class TestBuildRawVideoManifestArgs:
    """Tests for build_raw_video_manifest_args()."""

    def test_basic_single_robot_single_camera(self) -> None:
        """Build features for one robot and one camera."""
        import asyncio

        from utils.raw_video_dataset import build_raw_video_manifest_args

        # Mock environment
        robot_mock = MagicMock()
        robot_mock.robot = MagicMock()

        camera_mock = MagicMock()
        camera_mock.name = "Top"
        camera_mock.payload = MagicMock()
        camera_mock.payload.width = 640
        camera_mock.payload.height = 480

        env = MagicMock()
        env.robots = [robot_mock]
        env.cameras = [camera_mock]

        # Mock robot client — features() is sync, build() is async
        robot_client = MagicMock()
        robot_client.features.return_value = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow.pos"]

        robot_factory = MagicMock()
        robot_factory.build = AsyncMock(return_value=robot_client)

        result = asyncio.run(build_raw_video_manifest_args(env, robot_factory))

        assert result["state_names"] == ["shoulder_pan.pos", "shoulder_lift.pos", "elbow.pos"]
        assert result["action_names"] == ["shoulder_pan.pos", "shoulder_lift.pos", "elbow.pos"]
        assert len(result["cameras"]) == 1
        assert result["cameras"][0]["name"] == "top"  # lowercased
        assert result["cameras"][0]["width"] == 640
        assert result["cameras"][0]["height"] == 480

    def test_multiple_cameras(self) -> None:
        """Build features with multiple cameras."""
        import asyncio

        from utils.raw_video_dataset import build_raw_video_manifest_args

        robot_mock = MagicMock()
        robot_mock.robot = MagicMock()

        cam1 = MagicMock()
        cam1.name = "Top"
        cam1.payload = MagicMock()
        cam1.payload.width = 640
        cam1.payload.height = 480

        cam2 = MagicMock()
        cam2.name = "Gripper"
        cam2.payload = MagicMock()
        cam2.payload.width = 320
        cam2.payload.height = 240

        env = MagicMock()
        env.robots = [robot_mock]
        env.cameras = [cam1, cam2]

        robot_client = MagicMock()
        robot_client.features.return_value = ["joint_0"]
        robot_factory = MagicMock()
        robot_factory.build = AsyncMock(return_value=robot_client)

        result = asyncio.run(build_raw_video_manifest_args(env, robot_factory))
        assert len(result["cameras"]) == 2
        assert result["cameras"][0]["name"] == "top"
        assert result["cameras"][1]["name"] == "gripper"

    def test_multiple_robots_raises(self) -> None:
        """Raises ValueError with multiple robots."""
        import asyncio

        from utils.raw_video_dataset import build_raw_video_manifest_args

        env = MagicMock()
        env.robots = [MagicMock(), MagicMock()]
        robot_factory = MagicMock()

        with pytest.raises(ValueError, match="multiple robots"):
            asyncio.run(build_raw_video_manifest_args(env, robot_factory))

    def test_camera_no_payload_raises(self) -> None:
        """Raises ValueError if camera has no payload."""
        import asyncio

        from utils.raw_video_dataset import build_raw_video_manifest_args

        robot_mock = MagicMock()
        robot_mock.robot = MagicMock()

        cam = MagicMock()
        cam.name = "Top"
        cam.payload = None

        env = MagicMock()
        env.robots = [robot_mock]
        env.cameras = [cam]

        robot_client = MagicMock()
        robot_client.features.return_value = ["joint_0"]
        robot_factory = MagicMock()
        robot_factory.build = AsyncMock(return_value=robot_client)

        with pytest.raises(ValueError, match="no payload"):
            asyncio.run(build_raw_video_manifest_args(env, robot_factory))

    def test_camera_missing_dimensions_raises(self) -> None:
        """Raises ValueError if camera payload lacks width/height."""
        import asyncio

        from utils.raw_video_dataset import build_raw_video_manifest_args

        robot_mock = MagicMock()
        robot_mock.robot = MagicMock()

        cam = MagicMock()
        cam.name = "Top"
        cam.payload = MagicMock(spec=[])  # no width/height attributes

        env = MagicMock()
        env.robots = [robot_mock]
        env.cameras = [cam]

        robot_client = MagicMock()
        robot_client.features.return_value = ["joint_0"]
        robot_factory = MagicMock()
        robot_factory.build = AsyncMock(return_value=robot_client)

        with pytest.raises(ValueError, match="missing width/height"):
            asyncio.run(build_raw_video_manifest_args(env, robot_factory))
