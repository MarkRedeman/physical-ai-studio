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
from internal_datasets.raw_video.stats import DatasetStats, WelfordAccumulator
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

        from getiaction.data.observation import FeatureType

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

        from getiaction.data.observation import FeatureType

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

        from getiaction.data.dataset import Dataset
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
