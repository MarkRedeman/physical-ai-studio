# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Welford's online statistics computation and JSON caching for dataset normalization.

Computes element-wise mean, std, min, and max for state/action features (1-D float
vectors) in a single pass using Welford's numerically-stable online algorithm.
For image features, per-channel (C,) statistics are computed by decoding a uniform
sample of video frames.

Computed statistics are cached as ``.cache/stats.json`` inside the dataset root so
that subsequent loads are near-instantaneous.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .frame_index import FrameIndex
from .manifest import DatasetManifest
from .video_decode import decode_frames

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Welford accumulator
# ---------------------------------------------------------------------------


class WelfordAccumulator:
    """Numerically-stable online statistics using Welford's algorithm.

    Tracks element-wise mean, variance, min, and max over a stream of
    identically-shaped samples.

    Args:
        dim: Dimensionality of each sample vector.
    """

    def __init__(self, dim: int) -> None:
        self._dim = dim
        self._n: int = 0
        self._mean = np.zeros(dim, dtype=np.float64)
        self._m2 = np.zeros(dim, dtype=np.float64)
        self._min = np.full(dim, np.inf, dtype=np.float64)
        self._max = np.full(dim, -np.inf, dtype=np.float64)

    # -- single-sample update -----------------------------------------------

    def update(self, x: np.ndarray) -> None:
        """Update running statistics with a single sample of shape ``(dim,)``.

        Uses Welford's online algorithm for numerically-stable variance
        computation.

        Args:
            x: A 1-D array of length ``dim``.
        """
        x = np.asarray(x, dtype=np.float64)
        self._n += 1
        delta = x - self._mean
        self._mean += delta / self._n
        delta2 = x - self._mean
        self._m2 += delta * delta2
        np.minimum(self._min, x, out=self._min)
        np.maximum(self._max, x, out=self._max)

    # -- batch update -------------------------------------------------------

    def update_batch(self, x: np.ndarray) -> None:
        """Update running statistics with a batch of shape ``(N, dim)``.

        Iterates over the batch dimension and applies Welford's update for
        each sample.  This preserves numerical stability while remaining
        straightforward.

        Args:
            x: A 2-D array of shape ``(N, dim)``.
        """
        x = np.asarray(x, dtype=np.float64)
        for row in x:
            self.update(row)

    # -- read-out properties ------------------------------------------------

    @property
    def count(self) -> int:
        """Number of samples accumulated so far."""
        return self._n

    @property
    def mean(self) -> np.ndarray:
        """Element-wise mean. Shape ``(dim,)``."""
        return self._mean.copy()

    @property
    def std(self) -> np.ndarray:
        """Element-wise (population) standard deviation. Shape ``(dim,)``.

        Returns zeros if fewer than two samples have been observed.
        """
        if self._n < 2:
            return np.zeros(self._dim, dtype=np.float64)
        return np.sqrt(self._m2 / self._n)

    @property
    def min(self) -> np.ndarray:
        """Element-wise minimum. Shape ``(dim,)``."""
        return self._min.copy()

    @property
    def max(self) -> np.ndarray:
        """Element-wise maximum. Shape ``(dim,)``."""
        return self._max.copy()


# ---------------------------------------------------------------------------
# DatasetStats dataclass
# ---------------------------------------------------------------------------


@dataclass
class DatasetStats:
    """Aggregated normalization statistics for a raw-video dataset.

    Attributes:
        state: Element-wise stats for the state vector.  Keys ``"mean"``,
            ``"std"``, ``"min"``, ``"max"``; each value is a list of floats
            with length ``state_dim``.
        action: Element-wise stats for the action vector (same structure as
            *state*).
        images: Per-camera, per-channel stats.  Outer key is the camera name;
            inner keys are ``"mean"``, ``"std"``, ``"min"``, ``"max"``, each
            a list of 3 floats (one per RGB channel).
    """

    state: dict[str, list[float]] = field(default_factory=dict)
    action: dict[str, list[float]] = field(default_factory=dict)
    images: dict[str, dict[str, list[float]]] = field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize to a JSON string."""
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, s: str) -> DatasetStats:
        """Deserialize from a JSON string.

        Args:
            s: JSON string previously produced by :meth:`to_json`.

        Returns:
            Reconstructed :class:`DatasetStats` instance.
        """
        data = json.loads(s)
        return cls(**data)


# ---------------------------------------------------------------------------
# Helper: accumulator → dict
# ---------------------------------------------------------------------------


def _accumulator_to_dict(acc: WelfordAccumulator) -> dict[str, list[float]]:
    """Convert a :class:`WelfordAccumulator` into the dict schema expected by
    :class:`DatasetStats`."""
    return {
        "mean": acc.mean.tolist(),
        "std": acc.std.tolist(),
        "min": acc.min.tolist(),
        "max": acc.max.tolist(),
    }


# ---------------------------------------------------------------------------
# Compute stats
# ---------------------------------------------------------------------------


def compute_stats(
    manifest: DatasetManifest,
    frame_index: FrameIndex,
    dataset_root: Path,
    *,
    image_sample_count: int = 500,
) -> DatasetStats:
    """Compute normalization statistics over the entire dataset.

    State and action statistics are computed in a single sequential pass over
    every JSONL row using :class:`WelfordAccumulator`.  Image statistics are
    computed by uniformly sampling *image_sample_count* frames across the
    dataset and computing per-channel (C,) mean / std / min / max.

    Args:
        manifest: Parsed dataset manifest.
        frame_index: Pre-built frame index for fast video seeking.
        dataset_root: Root directory of the dataset on disk.
        image_sample_count: Number of video frames to decode for image
            statistics.  Defaults to ``500``.

    Returns:
        Fully populated :class:`DatasetStats`.
    """
    state_acc = WelfordAccumulator(manifest.state_dim)
    action_acc = WelfordAccumulator(manifest.action_dim)

    # ------------------------------------------------------------------
    # 1. Single pass over JSONL data for state / action stats
    # ------------------------------------------------------------------
    total_episodes = len(manifest.episodes)
    log_interval = max(1, total_episodes // 10)

    total_frames = 0
    for ep_idx, episode in enumerate(manifest.episodes):
        data_path = dataset_root / episode.episode_dir / episode.data_file

        with open(data_path, encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                state_acc.update(np.asarray(row["state"], dtype=np.float64))
                action_acc.update(np.asarray(row["action"], dtype=np.float64))
                total_frames += 1

        if (ep_idx + 1) % log_interval == 0 or ep_idx == total_episodes - 1:
            pct = 100.0 * (ep_idx + 1) / total_episodes
            logger.info(
                "State/action stats: processed episode %d/%d (%.0f%%), %d total frames so far",
                ep_idx + 1,
                total_episodes,
                pct,
                total_frames,
            )

    logger.info(
        "State/action stats complete: %d episodes, %d frames",
        total_episodes,
        total_frames,
    )

    # ------------------------------------------------------------------
    # 2. Sample video frames for per-channel image stats
    # ------------------------------------------------------------------
    camera_names = [cam.name for cam in manifest.cameras]
    image_stats: dict[str, dict[str, list[float]]] = {}

    total_dataset_frames = frame_index.total_frames

    if total_dataset_frames > 0 and camera_names:
        sample_count = min(image_sample_count, total_dataset_frames)
        # Uniformly spaced global frame indices
        sample_indices = np.linspace(0, total_dataset_frames - 1, sample_count, dtype=int)
        # Remove duplicates (can happen when total_dataset_frames < sample_count)
        sample_indices = np.unique(sample_indices)
        sample_count = len(sample_indices)

        logger.info(
            "Sampling %d frames for image stats across %d camera(s)",
            sample_count,
            len(camera_names),
        )

        for cam_name in camera_names:
            acc = WelfordAccumulator(3)  # RGB channels

            # Group sampled global indices by episode so we can batch-decode
            # frames from each video file efficiently.
            episode_frame_map: dict[int, list[int]] = defaultdict(list)
            for g_idx in sample_indices:
                ep_idx_s, vid_frame, _ = frame_index.lookup(int(g_idx))
                episode_frame_map[ep_idx_s].append(vid_frame)

            for ep_idx_s, vid_frames in sorted(episode_frame_map.items()):
                episode = manifest.episodes[ep_idx_s]
                video_path = dataset_root / episode.episode_dir / episode.video_files[cam_name]
                frames_tensor = decode_frames(video_path, vid_frames)  # (N, C, H, W)

                for frame in frames_tensor:
                    # frame: float32, shape (C, H, W), values in [0, 1].
                    # Flatten spatial dims so each pixel becomes a sample for
                    # per-channel stats.  shape -> (C, H*W) -> (H*W, C)
                    pixels = frame.numpy().reshape(frame.shape[0], -1).T  # (H*W, C)
                    acc.update_batch(pixels)

            image_stats[cam_name] = _accumulator_to_dict(acc)
            logger.info(
                "Image stats for camera %r: mean=%s, std=%s",
                cam_name,
                image_stats[cam_name]["mean"],
                image_stats[cam_name]["std"],
            )

    return DatasetStats(
        state=_accumulator_to_dict(state_acc),
        action=_accumulator_to_dict(action_acc),
        images=image_stats,
    )


# ---------------------------------------------------------------------------
# Cache-aware loader
# ---------------------------------------------------------------------------

_CACHE_DIR = ".cache"
_CACHE_FILENAME = "stats.json"


def load_or_compute_stats(
    manifest: DatasetManifest,
    frame_index: FrameIndex,
    dataset_root: Path,
) -> DatasetStats:
    """Load cached dataset stats or compute and cache them.

    The cache file is stored at ``<dataset_root>/.cache/stats.json``.  It is
    considered valid when its modification time is newer than that of
    ``manifest.json``.

    Args:
        manifest: Parsed dataset manifest.
        frame_index: Pre-built frame index for fast video seeking.
        dataset_root: Root directory of the dataset on disk.

    Returns:
        :class:`DatasetStats` loaded from cache or freshly computed.
    """
    cache_path = dataset_root / _CACHE_DIR / _CACHE_FILENAME
    manifest_path = dataset_root / "manifest.json"

    if cache_path.is_file() and manifest_path.is_file():
        manifest_mtime = manifest_path.stat().st_mtime
        cache_mtime = cache_path.stat().st_mtime
        if manifest_mtime < cache_mtime:
            logger.info("Loading cached stats from %s", cache_path)
            return DatasetStats.from_json(cache_path.read_text(encoding="utf-8"))
        logger.info("Cache is stale (manifest modified after cache); recomputing stats")
    else:
        logger.info("No cached stats found; computing stats from scratch")

    stats = compute_stats(manifest, frame_index, dataset_root)

    # Persist to cache
    cache_dir = dataset_root / _CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(stats.to_json(), encoding="utf-8")
    logger.info("Stats cached to %s", cache_path)

    return stats


__all__ = [
    "DatasetStats",
    "WelfordAccumulator",
    "compute_stats",
    "load_or_compute_stats",
]
