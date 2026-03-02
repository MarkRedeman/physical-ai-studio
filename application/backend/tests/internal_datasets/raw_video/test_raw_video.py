# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the raw-video dataset adapter and its supporting modules.

Tests cover manifest parsing, frame-index construction, dataset statistics,
nearest-neighbour resampling, and the full ``RawVideoDatasetAdapter``.  All
video-decoding calls are mocked so that no real video files are needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from internal_datasets.raw_video.frame_index import (
    EpisodeBoundary,
    FrameIndex,
    _nearest_neighbor_resample,
)
from internal_datasets.raw_video.manifest import (
    CameraConfig,
    DatasetManifest,
    EpisodeEntry,
    load_manifest,
)
from internal_datasets.raw_video.stats import (
    DatasetStats,
    EpisodeAccumulatorState,
    WelfordAccumulator,
    _load_episode_stats,
    _save_episode_stats,
    compute_episode_stats_background,
    compute_stats,
    load_or_compute_stats,
)
from internal_datasets.raw_video.video_decode import VideoInfo

# ============================================================================
# Fixtures
# ============================================================================


def _make_manifest_dict(
    *,
    num_episodes: int = 1,
    cameras: list[str] | None = None,
    state_dim: int = 6,
    action_dim: int = 6,
    fps: int = 30,
) -> dict:
    """Build a raw manifest dict suitable for writing to manifest.json."""
    if cameras is None:
        cameras = ["top"]
    cam_configs = [{"name": c} for c in cameras]
    episodes = []
    for i in range(num_episodes):
        ep_dir = f"episode_{i:03d}"
        video_files = {c: f"cam_{c}.mp4" for c in cameras}
        episodes.append({"episode_dir": ep_dir, "data_file": "data.jsonl", "video_files": video_files})
    return {
        "name": "test_dataset",
        "fps": fps,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "state_names": [f"state_{i}" for i in range(state_dim)],
        "action_names": [f"action_{i}" for i in range(action_dim)],
        "cameras": cam_configs,
        "episodes": episodes,
    }


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
            "state": [float(i)] * state_dim,
            "action": [float(i) * 0.1] * action_dim,
        }
        for i in range(n)
    ]


@pytest.fixture()
def tmp_dataset(tmp_path: Path) -> Path:
    """Create a minimal on-disk dataset with 2 episodes and 1 camera.

    Layout::
        tmp_path/
        ├── manifest.json
        ├── episode_000/
        │   ├── data.jsonl   (10 rows)
        │   └── cam_top.mp4  (empty file)
        └── episode_001/
            ├── data.jsonl   (5 rows)
            └── cam_top.mp4  (empty file)
    """
    manifest_dict = _make_manifest_dict(num_episodes=2, cameras=["top"])
    (tmp_path / "manifest.json").write_text(json.dumps(manifest_dict), encoding="utf-8")

    for i, n_rows in enumerate([10, 5]):
        ep_dir = tmp_path / f"episode_{i:03d}"
        ep_dir.mkdir()
        _write_jsonl(ep_dir / "data.jsonl", _make_data_rows(n_rows))
        (ep_dir / "cam_top.mp4").touch()

    return tmp_path


# ============================================================================
# manifest.py tests
# ============================================================================


class TestCameraConfig:
    def test_camera_config_valid_name(self):
        for name in ("top", "gripper_cam", "_hidden", "cam123"):
            cfg = CameraConfig(name=name)
            assert cfg.name == name

    def test_camera_config_invalid_name(self):
        for bad_name in ("1cam", "cam-bad", "cam.bad", "cam bad", ""):
            with pytest.raises(ValidationError):
                CameraConfig(name=bad_name)


class TestEpisodeEntry:
    def test_episode_entry_defaults(self):
        entry = EpisodeEntry(
            episode_dir="ep0",
            video_files={"top": "cam_top.mp4"},
        )
        assert entry.data_file == "data.jsonl"


class TestDatasetManifest:
    def test_dataset_manifest_camera_mismatch(self):
        """Episode declares cameras that don't match the manifest-level cameras."""
        with pytest.raises(ValidationError, match="camera mismatch"):
            DatasetManifest(
                name="test",
                fps=30,
                state_dim=6,
                action_dim=6,
                state_names=[f"state_{i}" for i in range(6)],
                action_names=[f"action_{i}" for i in range(6)],
                cameras=[CameraConfig(name="top")],
                episodes=[
                    EpisodeEntry(
                        episode_dir="ep0",
                        video_files={"wrong_cam": "vid.mp4"},
                    )
                ],
            )


class TestLoadManifest:
    def test_load_manifest_happy_path(self, tmp_dataset: Path):
        manifest = load_manifest(tmp_dataset)
        assert manifest.name == "test_dataset"
        assert manifest.fps == 30
        assert len(manifest.episodes) == 2
        assert len(manifest.cameras) == 1
        assert manifest.cameras[0].name == "top"

    def test_load_manifest_missing_file(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="Manifest file not found"):
            load_manifest(tmp_path)

    def test_load_manifest_missing_episode_dir(self, tmp_path: Path):
        manifest_dict = _make_manifest_dict(num_episodes=1, cameras=["top"])
        (tmp_path / "manifest.json").write_text(json.dumps(manifest_dict), encoding="utf-8")
        # Episode dir intentionally not created.
        with pytest.raises(FileNotFoundError, match="Episode directory"):
            load_manifest(tmp_path)

    def test_load_manifest_missing_video_file(self, tmp_path: Path):
        manifest_dict = _make_manifest_dict(num_episodes=1, cameras=["top"])
        (tmp_path / "manifest.json").write_text(json.dumps(manifest_dict), encoding="utf-8")
        ep_dir = tmp_path / "episode_000"
        ep_dir.mkdir()
        _write_jsonl(ep_dir / "data.jsonl", _make_data_rows(5))
        # Video file intentionally not created.
        with pytest.raises(FileNotFoundError, match="Video file"):
            load_manifest(tmp_path)


# ============================================================================
# frame_index.py tests — _nearest_neighbor_resample
# ============================================================================


class TestNearestNeighborResample:
    def test_resample_identity(self):
        result = _nearest_neighbor_resample(5, 5)
        np.testing.assert_array_equal(result, np.arange(5))

    def test_resample_single_frame(self):
        result = _nearest_neighbor_resample(1, 10)
        np.testing.assert_array_equal(result, np.array([0]))

    def test_resample_empty(self):
        result = _nearest_neighbor_resample(0, 5)
        assert len(result) == 0

    def test_resample_downsample(self):
        # 10 video frames, 5 data rows -> each video frame maps to the nearest data row.
        # scale = (5-1)/(10-1) = 4/9 ≈ 0.4444
        # frame_i -> round(i * 4/9)
        result = _nearest_neighbor_resample(10, 5)
        assert len(result) == 10
        # First video frame -> data row 0, last video frame -> data row 4.
        assert result[0] == 0
        assert result[-1] == 4
        # All values must be in [0, 4]
        assert np.all(result >= 0)
        assert np.all(result <= 4)
        # Expected: round([0, 0.44, 0.89, 1.33, 1.78, 2.22, 2.67, 3.11, 3.56, 4.0])
        #         = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
        expected = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        np.testing.assert_array_equal(result, expected)


# ============================================================================
# frame_index.py tests — FrameIndex
# ============================================================================


def _make_video_info(num_frames: int = 10, fps: float = 30.0) -> VideoInfo:
    return VideoInfo(num_frames=num_frames, fps=fps, width=640, height=480, duration_s=num_frames / fps, codec="h264")


class TestFrameIndex:
    @patch("internal_datasets.raw_video.frame_index._count_lines", return_value=10)
    @patch("internal_datasets.raw_video.frame_index.get_video_info")
    def test_frame_index_single_episode(self, mock_info, mock_lines, tmp_dataset):
        mock_info.return_value = _make_video_info(num_frames=10)
        manifest = load_manifest(tmp_dataset)
        # Use only the first episode.
        manifest.episodes = manifest.episodes[:1]

        fi = FrameIndex(manifest, tmp_dataset)

        assert fi.total_frames == 10
        assert fi.num_episodes == 1

        # Identity mapping when video frames == data rows.
        for i in range(10):
            ep, vid, row = fi.lookup(i)
            assert ep == 0
            assert vid == i
            assert row == i

    @patch("internal_datasets.raw_video.frame_index._count_lines")
    @patch("internal_datasets.raw_video.frame_index.get_video_info")
    def test_frame_index_multi_episode(self, mock_info, mock_lines, tmp_dataset):
        mock_info.return_value = _make_video_info(num_frames=10)
        mock_lines.return_value = 10
        manifest = load_manifest(tmp_dataset)

        fi = FrameIndex(manifest, tmp_dataset)

        assert fi.total_frames == 20  # 10 + 10
        assert fi.num_episodes == 2

        # First episode occupies [0, 10), second [10, 20).
        ep0, vid0, row0 = fi.lookup(0)
        assert ep0 == 0 and vid0 == 0 and row0 == 0

        ep9, vid9, row9 = fi.lookup(9)
        assert ep9 == 0 and vid9 == 9 and row9 == 9

        ep10, vid10, row10 = fi.lookup(10)
        assert ep10 == 1 and vid10 == 0 and row10 == 0

        ep19, vid19, row19 = fi.lookup(19)
        assert ep19 == 1 and vid19 == 9 and row19 == 9

        # Boundaries
        boundaries = fi.boundaries
        assert len(boundaries) == 2
        assert boundaries[0] == EpisodeBoundary(
            episode_idx=0, start=0, end=10, num_video_frames=10, num_data_rows=10, video_fps=30.0
        )
        assert boundaries[1] == EpisodeBoundary(
            episode_idx=1, start=10, end=20, num_video_frames=10, num_data_rows=10, video_fps=30.0
        )

    @patch("internal_datasets.raw_video.frame_index._count_lines", return_value=5)
    @patch("internal_datasets.raw_video.frame_index.get_video_info")
    def test_frame_index_fps_resampling(self, mock_info, mock_lines, tmp_dataset):
        """Video has 10 frames but only 5 data rows -> nearest-neighbour resampling."""
        mock_info.return_value = _make_video_info(num_frames=10)
        manifest = load_manifest(tmp_dataset)
        manifest.episodes = manifest.episodes[:1]

        fi = FrameIndex(manifest, tmp_dataset)

        assert fi.total_frames == 10
        # Video frame 0 -> data row 0, frame 9 -> data row 4.
        _, _, row_first = fi.lookup(0)
        _, _, row_last = fi.lookup(9)
        assert row_first == 0
        assert row_last == 4

    @patch("internal_datasets.raw_video.frame_index._count_lines", return_value=10)
    @patch("internal_datasets.raw_video.frame_index.get_video_info")
    def test_frame_index_lookup_out_of_range(self, mock_info, mock_lines, tmp_dataset):
        mock_info.return_value = _make_video_info(num_frames=10)
        manifest = load_manifest(tmp_dataset)
        manifest.episodes = manifest.episodes[:1]

        fi = FrameIndex(manifest, tmp_dataset)

        with pytest.raises(IndexError):
            fi.lookup(-1)
        with pytest.raises(IndexError):
            fi.lookup(10)

    @patch("internal_datasets.raw_video.frame_index._count_lines", return_value=10)
    @patch("internal_datasets.raw_video.frame_index.get_video_info")
    def test_frame_index_camera_mismatch(self, mock_info, mock_lines, tmp_path):
        """Two cameras in the same episode report different frame counts -> ValueError."""
        # Build a manifest with two cameras.
        manifest_dict = _make_manifest_dict(num_episodes=1, cameras=["top", "gripper"])
        (tmp_path / "manifest.json").write_text(json.dumps(manifest_dict), encoding="utf-8")
        ep_dir = tmp_path / "episode_000"
        ep_dir.mkdir()
        _write_jsonl(ep_dir / "data.jsonl", _make_data_rows(10))
        (ep_dir / "cam_top.mp4").touch()
        (ep_dir / "cam_gripper.mp4").touch()

        manifest = load_manifest(tmp_path)

        # Return different frame counts for each camera.
        def side_effect(video_path):
            if "cam_top" in str(video_path):
                return _make_video_info(num_frames=10)
            return _make_video_info(num_frames=8)

        mock_info.side_effect = side_effect

        with pytest.raises(ValueError, match="different frame counts"):
            FrameIndex(manifest, tmp_path)


# ============================================================================
# stats.py tests
# ============================================================================


class TestWelfordAccumulator:
    def test_welford_single_dim(self):
        acc = WelfordAccumulator(1)
        values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        for v in values:
            acc.update(np.array([v]))

        assert acc.count == 8
        np.testing.assert_allclose(acc.mean, [5.0])
        # Population std of the above values: sqrt(var) where var = mean((x-mean)^2)
        expected_std = np.std(values)
        np.testing.assert_allclose(acc.std, [expected_std], atol=1e-10)
        np.testing.assert_allclose(acc.min, [2.0])
        np.testing.assert_allclose(acc.max, [9.0])

    def test_welford_batch(self):
        values = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        # Individual updates.
        acc_individual = WelfordAccumulator(2)
        for row in values:
            acc_individual.update(row)

        # Batch update.
        acc_batch = WelfordAccumulator(2)
        acc_batch.update_batch(values)

        assert acc_individual.count == acc_batch.count
        np.testing.assert_allclose(acc_individual.mean, acc_batch.mean)
        np.testing.assert_allclose(acc_individual.std, acc_batch.std)
        np.testing.assert_allclose(acc_individual.min, acc_batch.min)
        np.testing.assert_allclose(acc_individual.max, acc_batch.max)

    def test_welford_empty(self):
        acc = WelfordAccumulator(3)
        assert acc.count == 0
        np.testing.assert_array_equal(acc.std, np.zeros(3))
        np.testing.assert_array_equal(acc.mean, np.zeros(3))


class TestDatasetStats:
    def test_dataset_stats_json_roundtrip(self):
        stats = DatasetStats(
            state={"mean": [1.0, 2.0], "std": [0.5, 0.3], "min": [0.0, 0.0], "max": [3.0, 4.0]},
            action={"mean": [0.1], "std": [0.01], "min": [-1.0], "max": [1.0]},
            images={
                "top": {
                    "mean": [0.5, 0.5, 0.5],
                    "std": [0.2, 0.2, 0.2],
                    "min": [0.0, 0.0, 0.0],
                    "max": [1.0, 1.0, 1.0],
                }
            },
        )
        json_str = stats.to_json()
        restored = DatasetStats.from_json(json_str)

        assert restored.state == stats.state
        assert restored.action == stats.action
        assert restored.images == stats.images


# ============================================================================
# adapter.py tests
# ============================================================================


class TestRawVideoDatasetAdapter:
    """Tests for ``RawVideoDatasetAdapter``.

    All heavy dependencies (manifest loading, frame index, stats, video
    decoding, JSONL loading) are mocked so that we exercise only the adapter
    logic itself.
    """

    _PATCH_PREFIX = "internal_datasets.raw_video.adapter"

    def _build_adapter(
        self,
        *,
        num_episodes: int = 1,
        frames_per_episode: int = 10,
        state_dim: int = 6,
        action_dim: int = 6,
        cameras: list[str] | None = None,
        fps: int = 30,
        image_transforms=None,
    ):
        """Construct a ``RawVideoDatasetAdapter`` with all externals mocked."""
        if cameras is None:
            cameras = ["top"]

        from internal_datasets.raw_video.manifest import CameraConfig as CC
        from internal_datasets.raw_video.manifest import DatasetManifest as DM
        from internal_datasets.raw_video.manifest import EpisodeEntry as EE

        cam_objs = [CC(name=c) for c in cameras]
        episodes = [
            EE(
                episode_dir=f"episode_{i:03d}",
                data_file="data.jsonl",
                video_files={c: f"cam_{c}.mp4" for c in cameras},
            )
            for i in range(num_episodes)
        ]
        manifest = DM(
            name="test",
            fps=fps,
            state_dim=state_dim,
            action_dim=action_dim,
            state_names=[f"joint_{i}.pos" for i in range(state_dim)],
            action_names=[f"joint_{i}.pos" for i in range(action_dim)],
            cameras=cam_objs,
            episodes=episodes,
        )

        # Build mock frame index.
        total = num_episodes * frames_per_episode
        mock_fi = MagicMock(spec=FrameIndex)
        mock_fi.total_frames = total
        mock_fi.num_episodes = num_episodes

        boundaries = []
        for ep in range(num_episodes):
            start = ep * frames_per_episode
            end = start + frames_per_episode
            boundaries.append(
                EpisodeBoundary(
                    episode_idx=ep,
                    start=start,
                    end=end,
                    num_video_frames=frames_per_episode,
                    num_data_rows=frames_per_episode,
                    video_fps=float(fps),
                )
            )
        mock_fi.boundaries = boundaries

        def lookup_fn(idx):
            ep = idx // frames_per_episode
            local = idx % frames_per_episode
            return (ep, local, local)

        mock_fi.lookup.side_effect = lookup_fn

        def episode_range_fn(ep_idx):
            return (ep_idx * frames_per_episode, (ep_idx + 1) * frames_per_episode)

        mock_fi.get_episode_range.side_effect = episode_range_fn

        # Build mock stats.
        mock_stats = DatasetStats(
            state={
                "mean": [0.0] * state_dim,
                "std": [1.0] * state_dim,
                "min": [-1.0] * state_dim,
                "max": [1.0] * state_dim,
            },
            action={
                "mean": [0.0] * action_dim,
                "std": [1.0] * action_dim,
                "min": [-1.0] * action_dim,
                "max": [1.0] * action_dim,
            },
            images={
                c: {"mean": [0.5, 0.5, 0.5], "std": [0.2, 0.2, 0.2], "min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0]}
                for c in cameras
            },
        )

        # Build pre-loaded data arrays (as _load_jsonl_data would).
        state_data = [np.random.randn(frames_per_episode, state_dim).astype(np.float32) for _ in range(num_episodes)]
        action_data = [np.random.randn(frames_per_episode, action_dim).astype(np.float32) for _ in range(num_episodes)]
        timestamps = [np.linspace(0, 1, frames_per_episode) for _ in range(num_episodes)]

        # Camera shapes.
        camera_shapes = {c: (3, 480, 640) for c in cameras}

        # VideoInfo mock for get_video_info.
        mock_video_info = _make_video_info(num_frames=frames_per_episode, fps=float(fps))

        # Patch everything and instantiate the adapter.
        with (
            patch(f"{self._PATCH_PREFIX}.load_manifest", return_value=manifest),
            patch(f"{self._PATCH_PREFIX}.FrameIndex", return_value=mock_fi),
            patch(f"{self._PATCH_PREFIX}.load_or_compute_stats", return_value=mock_stats),
            patch(f"{self._PATCH_PREFIX}._load_jsonl_data", return_value=(state_data, action_data, timestamps)),
            patch(f"{self._PATCH_PREFIX}.get_video_info", return_value=mock_video_info),
            patch(f"{self._PATCH_PREFIX}.decode_frame", return_value=torch.rand(3, 480, 640)),
            patch(f"{self._PATCH_PREFIX}.decode_frames", return_value=torch.rand(4, 3, 480, 640)),
        ):
            from internal_datasets.raw_video.adapter import RawVideoDatasetAdapter

            adapter = RawVideoDatasetAdapter(Path("/fake/dataset"), image_transforms=image_transforms)

        # After construction, also patch decode_frame/decode_frames for __getitem__.
        adapter._decode_frame_patch = patch(f"{self._PATCH_PREFIX}.decode_frame", return_value=torch.rand(3, 480, 640))
        adapter._decode_frames_patch = patch(
            f"{self._PATCH_PREFIX}.decode_frames", return_value=torch.rand(4, 3, 480, 640)
        )

        return adapter, manifest, mock_fi, mock_stats, state_data, action_data

    def test_adapter_len(self):
        adapter, _, mock_fi, *_ = self._build_adapter(num_episodes=2, frames_per_episode=15)
        assert len(adapter) == mock_fi.total_frames
        assert len(adapter) == 30

    def test_adapter_getitem_simple(self):
        adapter, manifest, _, _, state_data, action_data = self._build_adapter()

        with (
            patch(f"{self._PATCH_PREFIX}.decode_frame", return_value=torch.rand(3, 480, 640)),
        ):
            obs = adapter[0]

        assert obs.action is not None
        assert obs.state is not None
        assert obs.images is not None
        assert "top" in obs.images
        assert obs.episode_index.item() == 0
        assert obs.frame_index.item() == 0
        assert obs.index.item() == 0
        assert obs.task_index.item() == 0
        # State/action should match the pre-loaded arrays.
        np.testing.assert_allclose(obs.state.numpy(), state_data[0][0], atol=1e-6)
        np.testing.assert_allclose(obs.action.numpy(), action_data[0][0], atol=1e-6)

    def test_adapter_getitem_temporal(self):
        adapter, manifest, _, _, state_data, action_data = self._build_adapter(num_episodes=1, frames_per_episode=10)
        adapter.delta_indices = {"action": [0, 1, 2, 3]}

        with (
            patch(f"{self._PATCH_PREFIX}.decode_frame", return_value=torch.rand(3, 480, 640)),
            patch(f"{self._PATCH_PREFIX}.decode_frames", return_value=torch.rand(4, 3, 480, 640)),
        ):
            obs = adapter[0]

        # Action should be stacked along temporal axis: (4, action_dim).
        assert obs.action is not None
        assert obs.action.shape == (4, 6)
        # Pad mask should exist.
        assert obs.extra is not None
        assert "action_is_pad" in obs.extra
        assert obs.extra["action_is_pad"].shape == (4,)
        # At index 0, deltas [0,1,2,3] → targets [0,1,2,3] all within [0,10) → no padding.
        assert not obs.extra["action_is_pad"].any()

    def test_adapter_getitem_temporal_with_padding(self):
        """Test that clamping at the end of an episode produces pad flags."""
        adapter, _, _, _, _, _ = self._build_adapter(num_episodes=1, frames_per_episode=10)
        adapter.delta_indices = {"action": [0, 1, 2, 3]}

        with patch(f"{self._PATCH_PREFIX}.decode_frame", return_value=torch.rand(3, 480, 640)):
            obs = adapter[9]  # last frame, deltas [0,1,2,3] → targets [9,10,11,12]

        assert obs.extra is not None
        pad = obs.extra["action_is_pad"]
        # target 9 is valid, 10/11/12 are clamped → 3 pads.
        assert pad.tolist() == [False, True, True, True]

    def test_adapter_fps(self):
        adapter, manifest, *_ = self._build_adapter(fps=60)
        assert adapter.fps == 60

    def test_adapter_observation_features(self):
        adapter, *_ = self._build_adapter(state_dim=6, cameras=["top", "gripper"])

        from physicalai.data.observation import FeatureType

        obs_features = adapter.observation_features
        assert "state" in obs_features
        assert obs_features["state"].ftype == FeatureType.STATE
        assert obs_features["state"].shape == (6,)
        assert obs_features["state"].normalization_data is not None
        assert obs_features["state"].normalization_data.mean == [0.0] * 6

        assert "top" in obs_features
        assert obs_features["top"].ftype == FeatureType.VISUAL
        assert obs_features["top"].shape == (3, 480, 640)

        assert "gripper" in obs_features
        assert obs_features["gripper"].ftype == FeatureType.VISUAL

    def test_adapter_action_features(self):
        adapter, *_ = self._build_adapter(action_dim=4)

        from physicalai.data.observation import FeatureType

        action_features = adapter.action_features
        assert "action" in action_features
        assert action_features["action"].ftype == FeatureType.ACTION
        assert action_features["action"].shape == (4,)
        assert action_features["action"].normalization_data is not None
        assert action_features["action"].normalization_data.std == [1.0] * 4

    def test_adapter_delta_indices_getter_setter(self):
        adapter, *_ = self._build_adapter()
        # Initially empty.
        assert adapter.delta_indices == {}

        new_deltas = {"action": [0, 1, 2], "observation.state": [-1, 0, 1]}
        adapter.delta_indices = new_deltas
        assert adapter.delta_indices == new_deltas

    def test_adapter_is_dataset_subclass(self):
        adapter, *_ = self._build_adapter()

        from physicalai.data.dataset import Dataset
        from torch.utils.data import Dataset as TorchDataset

        assert isinstance(adapter, Dataset)
        assert isinstance(adapter, TorchDataset)

    def test_adapter_image_transforms_applied(self):
        """Verify that image_transforms is called for each camera in simple mode."""
        transform_fn = MagicMock(side_effect=lambda x: x * 0.5)
        adapter, *_ = self._build_adapter(cameras=["top", "gripper"], image_transforms=transform_fn)

        with patch(f"{self._PATCH_PREFIX}.decode_frame", return_value=torch.rand(3, 480, 640)):
            obs = adapter[0]

        # The transform should have been called once per camera.
        assert transform_fn.call_count == 2
        assert "top" in obs.images
        assert "gripper" in obs.images

    def test_adapter_image_transforms_temporal(self):
        """Verify that image_transforms is called per frame in temporal mode."""
        call_count = 0

        def counting_transform(x):
            nonlocal call_count
            call_count += 1
            return x

        adapter, *_ = self._build_adapter(cameras=["top"], image_transforms=counting_transform)
        adapter.delta_indices = {"action": [0, 1, 2, 3]}

        with (
            patch(f"{self._PATCH_PREFIX}.decode_frame", return_value=torch.rand(3, 480, 640)),
            patch(f"{self._PATCH_PREFIX}.decode_frames", return_value=torch.rand(4, 3, 480, 640)),
        ):
            obs = adapter[0]

        # Single camera, single frame (not in delta_indices for images) → 1 call.
        assert call_count == 1

    def test_adapter_no_transforms_when_none(self):
        """Verify no error and identity behaviour when image_transforms is None."""
        adapter, *_ = self._build_adapter(image_transforms=None)

        with patch(f"{self._PATCH_PREFIX}.decode_frame", return_value=torch.rand(3, 480, 640)) as mock_decode:
            obs = adapter[0]

        assert obs.images is not None
        assert "top" in obs.images
        # The decoded tensor should be passed through unchanged.
        assert obs.images["top"].shape == (3, 480, 640)


# ============================================================================
# stats.py tests — Group 1: WelfordAccumulator merge, serialization, batch
# ============================================================================


class TestWelfordAccumulatorExtended:
    """Additional tests for WelfordAccumulator: merge, serialization, incremental."""

    def test_merge_two_accumulators(self):
        """Merging two accumulators gives the same result as one accumulator over all data."""
        data_a = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        data_b = np.array([[7.0, 8.0], [9.0, 10.0]])

        acc_a = WelfordAccumulator(2)
        acc_a.update_batch(data_a)
        acc_b = WelfordAccumulator(2)
        acc_b.update_batch(data_b)

        acc_a.merge(acc_b)

        # Compare with single accumulator over all data.
        all_data = np.vstack([data_a, data_b])
        acc_all = WelfordAccumulator(2)
        acc_all.update_batch(all_data)

        assert acc_a.count == acc_all.count
        np.testing.assert_allclose(acc_a.mean, acc_all.mean, atol=1e-12)
        np.testing.assert_allclose(acc_a.std, acc_all.std, atol=1e-12)
        np.testing.assert_allclose(acc_a.min, acc_all.min)
        np.testing.assert_allclose(acc_a.max, acc_all.max)

    def test_merge_into_empty(self):
        """Merging a populated accumulator into an empty one copies its state."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        acc_src = WelfordAccumulator(2)
        acc_src.update_batch(data)

        acc_dst = WelfordAccumulator(2)
        assert acc_dst.count == 0
        acc_dst.merge(acc_src)

        assert acc_dst.count == acc_src.count
        np.testing.assert_allclose(acc_dst.mean, acc_src.mean)
        np.testing.assert_allclose(acc_dst.std, acc_src.std)
        np.testing.assert_allclose(acc_dst.min, acc_src.min)
        np.testing.assert_allclose(acc_dst.max, acc_src.max)

    def test_merge_empty_into_populated_is_noop(self):
        """Merging an empty accumulator into a populated one changes nothing."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        acc = WelfordAccumulator(2)
        acc.update_batch(data)

        mean_before = acc.mean.copy()
        std_before = acc.std.copy()

        acc.merge(WelfordAccumulator(2))

        np.testing.assert_array_equal(acc.mean, mean_before)
        np.testing.assert_array_equal(acc.std, std_before)

    def test_serialization_roundtrip(self):
        """to_dict / from_dict preserves accumulator state."""
        acc = WelfordAccumulator(3)
        data = np.random.randn(50, 3)
        acc.update_batch(data)

        d = acc.to_dict()
        restored = WelfordAccumulator.from_dict(d)

        assert restored.count == acc.count
        np.testing.assert_allclose(restored.mean, acc.mean)
        np.testing.assert_allclose(restored.std, acc.std)
        np.testing.assert_allclose(restored.min, acc.min)
        np.testing.assert_allclose(restored.max, acc.max)

    def test_serialization_empty_accumulator(self):
        """Serializing and restoring an empty accumulator works."""
        acc = WelfordAccumulator(4)
        d = acc.to_dict()
        restored = WelfordAccumulator.from_dict(d)
        assert restored.count == 0
        np.testing.assert_array_equal(restored.mean, np.zeros(4))

    def test_incremental_batch_matches_single_updates(self):
        """Multiple small batches give the same result as single-sample updates."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal((100, 5))

        acc_single = WelfordAccumulator(5)
        for row in data:
            acc_single.update(row)

        acc_batch = WelfordAccumulator(5)
        for start in range(0, 100, 10):
            acc_batch.update_batch(data[start : start + 10])

        assert acc_single.count == acc_batch.count
        np.testing.assert_allclose(acc_single.mean, acc_batch.mean, atol=1e-12)
        np.testing.assert_allclose(acc_single.std, acc_batch.std, atol=1e-10)
        np.testing.assert_allclose(acc_single.min, acc_batch.min)
        np.testing.assert_allclose(acc_single.max, acc_batch.max)

    def test_batch_update_empty_is_noop(self):
        """update_batch with zero rows does not change state."""
        acc = WelfordAccumulator(3)
        acc.update(np.array([1.0, 2.0, 3.0]))
        count_before = acc.count
        mean_before = acc.mean.copy()

        acc.update_batch(np.zeros((0, 3)))
        assert acc.count == count_before
        np.testing.assert_array_equal(acc.mean, mean_before)


# ============================================================================
# stats.py tests — Group 2: EpisodeAccumulatorState roundtrip
# ============================================================================


class TestEpisodeAccumulatorState:
    def test_json_roundtrip(self):
        """EpisodeAccumulatorState survives JSON serialization."""
        acc = WelfordAccumulator(2)
        acc.update_batch(np.array([[1.0, 2.0], [3.0, 4.0]]))
        img_acc = WelfordAccumulator(3)
        img_acc.update_batch(np.random.rand(10, 3))

        state = EpisodeAccumulatorState(
            state=acc.to_dict(),
            action=acc.to_dict(),
            images={"top": img_acc.to_dict()},
            num_frames=20,
            image_samples=10,
        )

        json_str = state.to_json()
        restored = EpisodeAccumulatorState.from_json(json_str)

        assert restored.num_frames == 20
        assert restored.image_samples == 10
        assert restored.state == state.state
        assert restored.action == state.action
        assert "top" in restored.images
        assert restored.images["top"]["n"] == img_acc.count

    def test_roundtrip_no_images(self):
        """Roundtrip works with empty image dict."""
        acc = WelfordAccumulator(1)
        acc.update(np.array([5.0]))
        state = EpisodeAccumulatorState(
            state=acc.to_dict(),
            action=acc.to_dict(),
            images={},
            num_frames=1,
            image_samples=0,
        )
        restored = EpisodeAccumulatorState.from_json(state.to_json())
        assert restored.images == {}
        assert restored.num_frames == 1


# ============================================================================
# stats.py tests — Group 3: Per-episode stats I/O
# ============================================================================


class TestPerEpisodeStatsIO:
    def test_save_and_load(self, tmp_path: Path):
        """_save_episode_stats / _load_episode_stats roundtrip."""
        acc = WelfordAccumulator(2)
        acc.update_batch(np.array([[1.0, 2.0], [3.0, 4.0]]))
        ep_state = EpisodeAccumulatorState(
            state=acc.to_dict(),
            action=acc.to_dict(),
            images={},
            num_frames=2,
            image_samples=0,
        )
        ep_dir = tmp_path / "ep_000"
        ep_dir.mkdir()
        _save_episode_stats(ep_dir, ep_state)

        loaded = _load_episode_stats(ep_dir)
        assert loaded is not None
        assert loaded.num_frames == 2
        assert loaded.state == ep_state.state

    def test_load_missing_returns_none(self, tmp_path: Path):
        """_load_episode_stats returns None when stats.json doesn't exist."""
        ep_dir = tmp_path / "ep_000"
        ep_dir.mkdir()
        assert _load_episode_stats(ep_dir) is None

    def test_load_corrupt_returns_none(self, tmp_path: Path):
        """_load_episode_stats returns None for corrupt JSON."""
        ep_dir = tmp_path / "ep_000"
        ep_dir.mkdir()
        (ep_dir / "stats.json").write_text("NOT VALID JSON {{{", encoding="utf-8")
        assert _load_episode_stats(ep_dir) is None


# ============================================================================
# stats.py tests — Group 4: compute_stats with source write-back
# ============================================================================


class TestComputeStatsWriteBack:
    """Tests for compute_stats() source_dataset_root write-back behavior."""

    def _make_dataset(self, root: Path, num_episodes: int = 2, frames: int = 5) -> DatasetManifest:
        """Create a minimal dataset and return its manifest."""
        from internal_datasets.raw_video.manifest import CameraConfig, DatasetManifest, EpisodeEntry, save_manifest

        episodes = []
        for i in range(num_episodes):
            ep_dir_name = f"episode_{i:03d}"
            ep_dir = root / ep_dir_name
            ep_dir.mkdir(parents=True, exist_ok=True)
            rows = [
                {"timestamp": t / 30.0, "state": [float(t)] * 2, "action": [float(t) * 0.1] * 2} for t in range(frames)
            ]
            _write_jsonl(ep_dir / "data.jsonl", rows)
            (ep_dir / "cam_top.mp4").touch()
            episodes.append(
                EpisodeEntry(episode_dir=ep_dir_name, data_file="data.jsonl", video_files={"top": "cam_top.mp4"})
            )
        manifest = DatasetManifest(
            name="test",
            fps=30,
            state_dim=2,
            action_dim=2,
            state_names=["s0", "s1"],
            action_names=["a0", "a1"],
            cameras=[CameraConfig(name="top")],
            episodes=episodes,
        )
        save_manifest(root, manifest)
        return manifest

    @patch("internal_datasets.raw_video.stats.decode_frames")
    @patch("internal_datasets.raw_video.stats.get_video_info")
    def test_compute_stats_writes_per_episode_cache(self, mock_info, mock_decode, tmp_path):
        """compute_stats writes stats.json to each episode dir."""
        mock_info.return_value = _make_video_info(num_frames=5)
        mock_decode.return_value = torch.rand(5, 3, 4, 4)

        manifest = self._make_dataset(tmp_path)
        compute_stats(manifest, tmp_path, image_sample_count=5)

        for ep in manifest.episodes:
            assert (tmp_path / ep.episode_dir / "stats.json").is_file()

    @patch("internal_datasets.raw_video.stats.decode_frames")
    @patch("internal_datasets.raw_video.stats.get_video_info")
    def test_compute_stats_writes_back_to_source(self, mock_info, mock_decode, tmp_path):
        """compute_stats with source_dataset_root writes stats to source too."""
        mock_info.return_value = _make_video_info(num_frames=5)
        mock_decode.return_value = torch.rand(5, 3, 4, 4)

        source = tmp_path / "source"
        snapshot = tmp_path / "snapshot"
        manifest = self._make_dataset(source)
        # Copy to snapshot (simulating snapshot creation).
        import shutil

        shutil.copytree(source, snapshot)

        compute_stats(manifest, snapshot, source_dataset_root=source, image_sample_count=5)

        # Both snapshot and source should have per-episode stats.
        for ep in manifest.episodes:
            assert (snapshot / ep.episode_dir / "stats.json").is_file()
            assert (source / ep.episode_dir / "stats.json").is_file()

    @patch("internal_datasets.raw_video.stats.decode_frames")
    @patch("internal_datasets.raw_video.stats.get_video_info")
    def test_compute_stats_reads_from_source_cache(self, mock_info, mock_decode, tmp_path):
        """compute_stats reads cached stats from source when local is missing."""
        mock_info.return_value = _make_video_info(num_frames=5)
        mock_decode.return_value = torch.rand(5, 3, 4, 4)

        source = tmp_path / "source"
        manifest = self._make_dataset(source)

        # Pre-compute stats for source so episodes have caches.
        compute_stats(manifest, source, image_sample_count=5)

        # Create a snapshot without per-episode stats.
        snapshot = tmp_path / "snapshot"
        import shutil

        shutil.copytree(source, snapshot)
        # Remove stats from snapshot only.
        for ep in manifest.episodes:
            stats_file = snapshot / ep.episode_dir / "stats.json"
            if stats_file.is_file():
                stats_file.unlink()

        # decode_frames should NOT be called because source has cached stats.
        mock_decode.reset_mock()
        compute_stats(manifest, snapshot, source_dataset_root=source, image_sample_count=5)

        # decode_frames should not have been called (all from cache).
        mock_decode.assert_not_called()

        # Stats should now also be copied into snapshot.
        for ep in manifest.episodes:
            assert (snapshot / ep.episode_dir / "stats.json").is_file()

    @patch("internal_datasets.raw_video.stats.decode_frames")
    @patch("internal_datasets.raw_video.stats.get_video_info")
    def test_compute_stats_no_writeback_when_source_none(self, mock_info, mock_decode, tmp_path):
        """Without source_dataset_root, no write-back occurs."""
        mock_info.return_value = _make_video_info(num_frames=5)
        mock_decode.return_value = torch.rand(5, 3, 4, 4)

        manifest = self._make_dataset(tmp_path)
        compute_stats(manifest, tmp_path, image_sample_count=5)

        # Stats should only be in local, not in some other location.
        for ep in manifest.episodes:
            assert (tmp_path / ep.episode_dir / "stats.json").is_file()

    @patch("internal_datasets.raw_video.stats.decode_frames")
    @patch("internal_datasets.raw_video.stats.get_video_info")
    def test_compute_stats_uses_local_cache_first(self, mock_info, mock_decode, tmp_path):
        """When local cache exists, source is not checked and decode is skipped."""
        mock_info.return_value = _make_video_info(num_frames=5)
        mock_decode.return_value = torch.rand(5, 3, 4, 4)

        source = tmp_path / "source"
        manifest = self._make_dataset(source)

        # Pre-compute local cache.
        compute_stats(manifest, source, image_sample_count=5)

        mock_decode.reset_mock()
        # Run again — should use local cache, no decoding.
        compute_stats(manifest, source, image_sample_count=5)
        mock_decode.assert_not_called()


# ============================================================================
# stats.py tests — Group 5: load_or_compute_stats cache behavior
# ============================================================================


class TestLoadOrComputeStats:
    """Tests for load_or_compute_stats cache freshness and passthrough."""

    def _make_dataset(self, root: Path, frames: int = 5) -> DatasetManifest:
        """Create a minimal 1-episode dataset."""
        from internal_datasets.raw_video.manifest import CameraConfig, DatasetManifest, EpisodeEntry, save_manifest

        ep_dir = root / "episode_000"
        ep_dir.mkdir(parents=True, exist_ok=True)
        rows = [{"timestamp": t / 30.0, "state": [float(t)] * 2, "action": [float(t) * 0.1] * 2} for t in range(frames)]
        _write_jsonl(ep_dir / "data.jsonl", rows)
        (ep_dir / "cam_top.mp4").touch()
        manifest = DatasetManifest(
            name="test",
            fps=30,
            state_dim=2,
            action_dim=2,
            state_names=["s0", "s1"],
            action_names=["a0", "a1"],
            cameras=[CameraConfig(name="top")],
            episodes=[
                EpisodeEntry(episode_dir="episode_000", data_file="data.jsonl", video_files={"top": "cam_top.mp4"})
            ],
        )
        save_manifest(root, manifest)
        return manifest

    @patch("internal_datasets.raw_video.stats.decode_frames")
    @patch("internal_datasets.raw_video.stats.get_video_info")
    def test_creates_merged_cache(self, mock_info, mock_decode, tmp_path):
        """load_or_compute_stats creates .cache/stats.json."""
        mock_info.return_value = _make_video_info(num_frames=5)
        mock_decode.return_value = torch.rand(5, 3, 4, 4)

        manifest = self._make_dataset(tmp_path)
        load_or_compute_stats(manifest, tmp_path)

        assert (tmp_path / ".cache" / "stats.json").is_file()

    @patch("internal_datasets.raw_video.stats.decode_frames")
    @patch("internal_datasets.raw_video.stats.get_video_info")
    def test_returns_cached_on_second_call(self, mock_info, mock_decode, tmp_path):
        """Second call returns from merged cache without recomputation."""
        mock_info.return_value = _make_video_info(num_frames=5)
        mock_decode.return_value = torch.rand(5, 3, 4, 4)

        manifest = self._make_dataset(tmp_path)
        stats1 = load_or_compute_stats(manifest, tmp_path)
        mock_decode.reset_mock()
        stats2 = load_or_compute_stats(manifest, tmp_path)

        # No video decoding on second call.
        mock_decode.assert_not_called()
        assert stats1.state == stats2.state

    @patch("internal_datasets.raw_video.stats.decode_frames")
    @patch("internal_datasets.raw_video.stats.get_video_info")
    def test_passes_source_dataset_root_through(self, mock_info, mock_decode, tmp_path):
        """load_or_compute_stats passes source_dataset_root to compute_stats."""
        mock_info.return_value = _make_video_info(num_frames=5)
        mock_decode.return_value = torch.rand(5, 3, 4, 4)

        source = tmp_path / "source"
        snapshot = tmp_path / "snapshot"
        manifest = self._make_dataset(source)
        import shutil

        shutil.copytree(source, snapshot)

        load_or_compute_stats(manifest, snapshot, source_dataset_root=source)

        # Source should have per-episode stats written back.
        assert (source / "episode_000" / "stats.json").is_file()


# ============================================================================
# stats.py tests — Group 6: Background computation
# ============================================================================


class TestComputeEpisodeStatsBackground:
    """Tests for compute_episode_stats_background."""

    def _make_dataset(self, root: Path, frames: int = 5) -> tuple[DatasetManifest, EpisodeEntry]:
        """Create a 1-episode dataset, return (manifest, episode)."""
        from internal_datasets.raw_video.manifest import CameraConfig, DatasetManifest, EpisodeEntry, save_manifest

        ep_dir = root / "episode_000"
        ep_dir.mkdir(parents=True, exist_ok=True)
        rows = [{"timestamp": t / 30.0, "state": [float(t)] * 2, "action": [float(t) * 0.1] * 2} for t in range(frames)]
        _write_jsonl(ep_dir / "data.jsonl", rows)
        (ep_dir / "cam_top.mp4").touch()
        episode = EpisodeEntry(episode_dir="episode_000", data_file="data.jsonl", video_files={"top": "cam_top.mp4"})
        manifest = DatasetManifest(
            name="test",
            fps=30,
            state_dim=2,
            action_dim=2,
            state_names=["s0", "s1"],
            action_names=["a0", "a1"],
            cameras=[CameraConfig(name="top")],
            episodes=[episode],
        )
        save_manifest(root, manifest)
        return manifest, episode

    @patch("internal_datasets.raw_video.stats.decode_frames")
    @patch("internal_datasets.raw_video.stats.get_video_info")
    def test_background_writes_to_both_paths(self, mock_info, mock_decode, tmp_path):
        """Background computation writes stats to both cache and source."""
        mock_info.return_value = _make_video_info(num_frames=5)
        mock_decode.return_value = torch.rand(5, 3, 4, 4)

        cache_root = tmp_path / "cache"
        source_root = tmp_path / "source"
        manifest, episode = self._make_dataset(cache_root)
        # Also create the episode dir in source.
        source_ep_dir = source_root / "episode_000"
        source_ep_dir.mkdir(parents=True, exist_ok=True)

        compute_episode_stats_background(manifest, episode, cache_root, source_dataset_root=source_root)

        assert (cache_root / "episode_000" / "stats.json").is_file()
        assert (source_ep_dir / "stats.json").is_file()

    @patch("internal_datasets.raw_video.stats.decode_frames")
    @patch("internal_datasets.raw_video.stats.get_video_info")
    def test_background_noop_when_cached(self, mock_info, mock_decode, tmp_path):
        """Background is a no-op when stats are already cached locally."""
        mock_info.return_value = _make_video_info(num_frames=5)
        mock_decode.return_value = torch.rand(5, 3, 4, 4)

        manifest, episode = self._make_dataset(tmp_path)

        # Pre-compute stats.
        compute_episode_stats_background(manifest, episode, tmp_path)
        assert (tmp_path / "episode_000" / "stats.json").is_file()

        # Second call should not decode anything.
        mock_decode.reset_mock()
        compute_episode_stats_background(manifest, episode, tmp_path)
        mock_decode.assert_not_called()
