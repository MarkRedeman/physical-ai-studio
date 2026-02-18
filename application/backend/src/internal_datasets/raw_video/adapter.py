# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Raw-video dataset adapter for the GetiAction training pipeline.

This module provides :class:`RawVideoDatasetAdapter`, an implementation of the
abstract :class:`getiaction.data.Dataset` base class that reads training data
directly from raw video files and JSONL telemetry logs -- without requiring a
prior conversion step to the LeRobot HuggingFace format.

The adapter pre-loads all scalar data (state, action, timestamps) into memory
as contiguous numpy arrays for fast random access, while video frames are
decoded lazily on demand.  When ``delta_indices`` are set by the training
pipeline, the adapter produces temporally-windowed tensors with proper
clamping and padding masks at episode boundaries.

Typical usage::

    from pathlib import Path
    from internal_datasets.raw_video.adapter import RawVideoDatasetAdapter

    dataset = RawVideoDatasetAdapter(Path("data/raw_dataset"))
    print(len(dataset))        # total frames across all episodes
    obs = dataset[0]           # single-frame Observation
    dataset.delta_indices = {"action": [0, 1, 2, 3]}
    obs = dataset[0]           # obs.action has shape (4, action_dim)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from getiaction.data.dataset import Dataset
from getiaction.data.observation import Feature, FeatureType, NormalizationParameters, Observation

from .frame_index import FrameIndex
from .manifest import load_manifest
from .stats import DatasetStats, load_or_compute_stats
from .video_decode import decode_frame, decode_frames

if TYPE_CHECKING:
    from .manifest import DatasetManifest

logger = logging.getLogger(__name__)


def _load_jsonl_data(
    manifest: DatasetManifest,
    dataset_root: Path,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Pre-load all JSONL episode data into memory as contiguous numpy arrays.

    Each JSONL file contains one JSON object per line with the fields
    ``"timestamp"``, ``"state"``, and ``"action"``.

    Args:
        manifest: Validated dataset manifest.
        dataset_root: Root directory of the dataset on disk.

    Returns:
        A 3-tuple of:
        - ``state_data``:  list of arrays, each ``(num_rows, state_dim)``
        - ``action_data``: list of arrays, each ``(num_rows, action_dim)``
        - ``timestamps``:  list of arrays, each ``(num_rows,)``

    Raises:
        ValueError: If a JSONL row is missing required fields or contains
            vectors of unexpected dimensionality.
    """
    state_data: list[np.ndarray] = []
    action_data: list[np.ndarray] = []
    timestamps: list[np.ndarray] = []

    for ep_idx, episode in enumerate(manifest.episodes):
        data_path = dataset_root / episode.episode_dir / episode.data_file

        rows_state: list[list[float]] = []
        rows_action: list[list[float]] = []
        rows_ts: list[float] = []

        with data_path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {line_no} of {data_path}: {exc}") from exc

                for field in ("timestamp", "state", "action"):
                    if field not in record:
                        raise ValueError(f"Missing field {field!r} on line {line_no} of {data_path}")

                state_vec = record["state"]
                action_vec = record["action"]

                if len(state_vec) != manifest.state_dim:
                    raise ValueError(
                        f"Episode {ep_idx}, line {line_no}: expected state_dim="
                        f"{manifest.state_dim}, got {len(state_vec)}"
                    )
                if len(action_vec) != manifest.action_dim:
                    raise ValueError(
                        f"Episode {ep_idx}, line {line_no}: expected action_dim="
                        f"{manifest.action_dim}, got {len(action_vec)}"
                    )

                rows_state.append(state_vec)
                rows_action.append(action_vec)
                rows_ts.append(float(record["timestamp"]))

        state_data.append(np.array(rows_state, dtype=np.float32))
        action_data.append(np.array(rows_action, dtype=np.float32))
        timestamps.append(np.array(rows_ts, dtype=np.float64))

        logger.debug(
            "Loaded episode %d (%s): %d rows",
            ep_idx,
            episode.episode_dir,
            len(rows_ts),
        )

    return state_data, action_data, timestamps


class RawVideoDatasetAdapter(Dataset):
    """Dataset adapter that reads raw video files and JSONL telemetry directly.

    This adapter implements the :class:`getiaction.data.Dataset` abstract base
    class so that raw-video datasets can be consumed by the training pipeline
    without an intermediate format conversion step.

    All scalar data (state vectors, action vectors, timestamps) are eagerly
    loaded into memory during construction for fast random access.  Video
    frames are decoded lazily via :func:`.video_decode.decode_frame` /
    :func:`.video_decode.decode_frames`.

    When ``delta_indices`` are configured by the training framework, the
    adapter returns temporally-windowed observations with proper clamping at
    episode boundaries and boolean pad masks in the ``extra`` dict.

    Args:
        dataset_root: Path to the top-level dataset directory that contains
            ``manifest.json``.
        tolerance_s: Tolerance in seconds used to keep delta timestamps in
            sync with the target FPS.  Passed through to the training
            pipeline.  Defaults to ``1e-4``.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, dataset_root: Path, *, tolerance_s: float = 1e-4) -> None:
        super().__init__()

        self._dataset_root = Path(dataset_root)
        self._tolerance_s = tolerance_s

        # Load and validate the manifest.
        self._manifest: DatasetManifest = load_manifest(self._dataset_root)

        # Build the frame index that maps global indices to (episode, frame).
        self._frame_index = FrameIndex(self._manifest, self._dataset_root)

        # Pre-load all JSONL data into contiguous numpy arrays.
        self._state_data: list[np.ndarray]
        self._action_data: list[np.ndarray]
        self._timestamps: list[np.ndarray]
        self._state_data, self._action_data, self._timestamps = _load_jsonl_data(self._manifest, self._dataset_root)

        # Load (or lazily compute and cache) normalization statistics.
        self._stats: DatasetStats = load_or_compute_stats(self._manifest, self._frame_index, self._dataset_root)

        # Delta indices are set later by the training pipeline.
        self._delta_indices: dict[str, list[int]] = {}

        logger.info(
            "RawVideoDatasetAdapter initialised: %d episodes, %d total frames, state_dim=%d, action_dim=%d, fps=%d",
            len(self._manifest.episodes),
            len(self),
            self._manifest.state_dim,
            self._manifest.action_dim,
            self._manifest.fps,
        )

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the total number of indexable frames across all episodes."""
        return self._frame_index.total_frames

    def __getitem__(self, idx: int) -> Observation:
        """Load and return a single training observation.

        The observation contents depend on whether ``delta_indices`` have been
        set:

        * **Simple mode** (no delta indices): returns single-frame tensors for
          state, action, and each camera image.
        * **Temporal windowing mode** (delta indices set): returns stacked
          tensors along a temporal axis for each key listed in
          ``delta_indices``, with boolean pad masks in the ``extra`` dict.

        Args:
            idx: Global frame index in ``[0, len(self))``.

        Returns:
            An :class:`~getiaction.data.observation.Observation` populated with
            the requested data.

        Raises:
            IndexError: If *idx* is out of range.
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} is out of range for dataset with {len(self)} frames")

        if self._delta_indices:
            return self._getitem_temporal(idx)
        return self._getitem_simple(idx)

    # ------------------------------------------------------------------
    # Simple (single-frame) access
    # ------------------------------------------------------------------

    def _getitem_simple(self, idx: int) -> Observation:
        """Retrieve a single-frame observation without temporal windowing."""
        episode_idx, video_frame_idx, data_row_idx = self._frame_index.lookup(idx)
        episode = self._manifest.episodes[episode_idx]

        # Decode one video frame per camera.
        images: dict[str, torch.Tensor] = {}
        for cam in self._manifest.cameras:
            video_path = self._dataset_root / episode.episode_dir / episode.video_files[cam.name]
            images[cam.name] = decode_frame(video_path, video_frame_idx)

        # Scalar data from pre-loaded numpy arrays.
        state_tensor = torch.from_numpy(self._state_data[episode_idx][data_row_idx].copy())
        action_tensor = torch.from_numpy(self._action_data[episode_idx][data_row_idx].copy())

        timestamp = video_frame_idx / self._manifest.fps

        return Observation(
            action=action_tensor,
            state=state_tensor,
            images=images,
            episode_index=torch.tensor(episode_idx),
            frame_index=torch.tensor(video_frame_idx),
            index=torch.tensor(idx),
            task_index=torch.tensor(0),
            timestamp=torch.tensor(timestamp),
        )

    # ------------------------------------------------------------------
    # Temporal windowing access
    # ------------------------------------------------------------------

    def _getitem_temporal(self, idx: int) -> Observation:
        """Retrieve a temporally-windowed observation with pad masks.

        For every key in ``delta_indices`` the adapter resolves the target
        global index ``idx + delta``, clamps it to the episode that *idx*
        belongs to, and marks clamped positions as padded.
        """
        episode_idx, video_frame_idx, data_row_idx = self._frame_index.lookup(idx)
        episode = self._manifest.episodes[episode_idx]
        ep_start, ep_end = self._frame_index.episode_range(episode_idx)

        # Containers for the final observation fields.
        action_tensor: torch.Tensor | None = None
        state_tensor: torch.Tensor | None = None
        images: dict[str, torch.Tensor] = {}
        extra: dict[str, torch.Tensor] = {}

        # Track which cameras have already been handled via delta keys so we
        # can fill the remaining ones with single-frame decode.
        cameras_handled: set[str] = set()

        for key, deltas in self._delta_indices.items():
            # Compute clamped indices and pad mask for this key.
            clamped_global: list[int] = []
            is_pad: list[bool] = []
            for delta in deltas:
                target = idx + delta
                clamped = max(ep_start, min(target, ep_end - 1))
                clamped_global.append(clamped)
                is_pad.append(target != clamped)

            pad_mask = torch.tensor(is_pad, dtype=torch.bool)

            if key == "action":
                frames = []
                for g_idx in clamped_global:
                    ep_of_frame, _, row = self._frame_index.lookup(g_idx)
                    frames.append(torch.from_numpy(self._action_data[ep_of_frame][row].copy()))
                action_tensor = torch.stack(frames)  # (T, action_dim)
                extra["action_is_pad"] = pad_mask

            elif key == "observation.state":
                frames = []
                for g_idx in clamped_global:
                    ep_of_frame, _, row = self._frame_index.lookup(g_idx)
                    frames.append(torch.from_numpy(self._state_data[ep_of_frame][row].copy()))
                state_tensor = torch.stack(frames)  # (T, state_dim)
                extra["observation.state_is_pad"] = pad_mask

            elif key.startswith("observation.images."):
                cam_name = key[len("observation.images.") :]
                cameras_handled.add(cam_name)

                # Collect all frame indices for batch decode.
                frame_indices_per_idx: list[tuple[int, int]] = []  # (ep, vid_frame)
                for g_idx in clamped_global:
                    ep_of_frame, vid_frame, _ = self._frame_index.lookup(g_idx)
                    frame_indices_per_idx.append((ep_of_frame, vid_frame))

                # When all frames come from the same episode we can batch-decode.
                unique_episodes = {ep for ep, _ in frame_indices_per_idx}
                if len(unique_episodes) == 1:
                    vid_frames = [vf for _, vf in frame_indices_per_idx]
                    video_path = self._dataset_root / episode.episode_dir / episode.video_files[cam_name]
                    decoded = decode_frames(video_path, vid_frames)  # (T, C, H, W)
                else:
                    # Rare edge case: should not happen due to clamping, but
                    # handle gracefully by decoding one-by-one.
                    decoded_list = []
                    for ep_of_frame, vid_frame in frame_indices_per_idx:
                        ep_entry = self._manifest.episodes[ep_of_frame]
                        video_path = self._dataset_root / ep_entry.episode_dir / ep_entry.video_files[cam_name]
                        decoded_list.append(decode_frame(video_path, vid_frame))
                    decoded = torch.stack(decoded_list)

                images[cam_name] = decoded  # (T, C, H, W)
                extra[f"observation.images.{cam_name}_is_pad"] = pad_mask

        # Fill in state/action if they were not covered by delta_indices.
        if action_tensor is None:
            action_tensor = torch.from_numpy(self._action_data[episode_idx][data_row_idx].copy())
        if state_tensor is None:
            state_tensor = torch.from_numpy(self._state_data[episode_idx][data_row_idx].copy())

        # Decode single frames for cameras not in delta_indices.
        for cam in self._manifest.cameras:
            if cam.name not in cameras_handled and cam.name not in images:
                video_path = self._dataset_root / episode.episode_dir / episode.video_files[cam.name]
                images[cam.name] = decode_frame(video_path, video_frame_idx)

        timestamp = video_frame_idx / self._manifest.fps

        return Observation(
            action=action_tensor,
            state=state_tensor,
            images=images,
            episode_index=torch.tensor(episode_idx),
            frame_index=torch.tensor(video_frame_idx),
            index=torch.tensor(idx),
            task_index=torch.tensor(0),
            timestamp=torch.tensor(timestamp),
            extra=extra if extra else None,
        )

    # ------------------------------------------------------------------
    # Feature descriptors
    # ------------------------------------------------------------------

    @property
    def raw_features(self) -> dict:
        """Return a dict describing the raw feature schema of this dataset.

        Keys follow the dotted convention used elsewhere in the platform::

            {
                "observation.state": {"dtype": "float32", "shape": (state_dim,)},
                "observation.images.<cam>": {"dtype": "video", "shape": (C, H, W)},
                "action": {"dtype": "float32", "shape": (action_dim,)},
            }
        """
        features: dict[str, dict] = {
            "observation.state": {
                "dtype": "float32",
                "shape": (self._manifest.state_dim,),
            },
            "action": {
                "dtype": "float32",
                "shape": (self._manifest.action_dim,),
            },
        }

        for cam in self._manifest.cameras:
            shape = self._frame_index.camera_shape(cam.name)  # (C, H, W)
            features[f"observation.images.{cam.name}"] = {
                "dtype": "video",
                "shape": shape,
            }

        return features

    @property
    def observation_features(self) -> dict[str, Feature]:
        """Return observation features with normalization parameters.

        Keys are short names (without the ``"observation."`` prefix):

        - ``"state"`` -- :attr:`FeatureType.STATE`
        - ``"<cam_name>"`` -- :attr:`FeatureType.VISUAL` for each camera
        """
        features: dict[str, Feature] = {}

        # State feature.
        state_stats = self._stats.state
        features["state"] = Feature(
            ftype=FeatureType.STATE,
            shape=(self._manifest.state_dim,),
            name="state",
            normalization_data=NormalizationParameters(
                mean=state_stats["mean"],
                std=state_stats["std"],
                min=state_stats["min"],
                max=state_stats["max"],
            ),
        )

        # Camera features.
        for cam in self._manifest.cameras:
            cam_stats = self._stats.cameras[cam.name]
            shape = self._frame_index.camera_shape(cam.name)  # (C, H, W)
            features[cam.name] = Feature(
                ftype=FeatureType.VISUAL,
                shape=shape,
                name=cam.name,
                normalization_data=NormalizationParameters(
                    mean=cam_stats["mean"],
                    std=cam_stats["std"],
                    min=cam_stats["min"],
                    max=cam_stats["max"],
                ),
            )

        return features

    @property
    def action_features(self) -> dict[str, Feature]:
        """Return action features with normalization parameters.

        Returns a single entry keyed ``"action"`` with
        :attr:`FeatureType.ACTION`.
        """
        action_stats = self._stats.action
        return {
            "action": Feature(
                ftype=FeatureType.ACTION,
                shape=(self._manifest.action_dim,),
                name="action",
                normalization_data=NormalizationParameters(
                    mean=action_stats["mean"],
                    std=action_stats["std"],
                    min=action_stats["min"],
                    max=action_stats["max"],
                ),
            ),
        }

    # ------------------------------------------------------------------
    # Metadata properties
    # ------------------------------------------------------------------

    @property
    def fps(self) -> int:
        """Target frames per second declared in the dataset manifest."""
        return self._manifest.fps

    @property
    def tolerance_s(self) -> float:
        """Tolerance in seconds for delta-timestamp synchronisation."""
        return self._tolerance_s

    @property
    def delta_indices(self) -> dict[str, list[int]]:
        """Current temporal windowing configuration.

        An empty dict means single-frame mode.  When set by the training
        pipeline, keys are dotted feature paths (e.g. ``"action"``,
        ``"observation.state"``, ``"observation.images.top"``) and values
        are lists of integer offsets relative to the current index.
        """
        return self._delta_indices

    @delta_indices.setter
    def delta_indices(self, indices: dict[str, list[int]]) -> None:
        """Set the temporal windowing configuration.

        Args:
            indices: Mapping from dotted feature keys to lists of integer
                frame offsets.
        """
        self._delta_indices = indices


__all__ = ["RawVideoDatasetAdapter"]
