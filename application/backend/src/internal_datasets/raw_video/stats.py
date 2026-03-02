# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Welford's online statistics computation and JSON caching for dataset normalization.

Computes element-wise mean, std, min, and max for state/action features (1-D float
vectors) using Welford's numerically-stable online algorithm.  For image features,
per-channel (C,) statistics are computed by decoding a uniform sample of video frames.

Statistics are cached at two levels:

1. **Per-episode**: ``<episode_dir>/stats.json`` stores the raw Welford accumulator
   state for that episode's state, action, and image data.  These files travel with
   the episode directory during copy/delete operations.
2. **Merged dataset-level**: ``<dataset_root>/.cache/stats.json`` stores the final
   merged statistics for the entire dataset.

When loading stats, episodes with existing per-episode caches are skipped (no video
decoding needed).  Only episodes missing a ``stats.json`` are computed fresh.  The
per-episode accumulators are then merged in O(num_episodes) to produce the final
dataset-level statistics.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .manifest import DatasetManifest, EpisodeEntry
from .video_decode import decode_frames, get_video_info

from loguru import logger


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

        Uses the parallel/batch variant of Welford's algorithm to merge
        the batch statistics with the running accumulator in O(dim) time,
        regardless of batch size.

        Args:
            x: A 2-D array of shape ``(N, dim)``.
        """
        x = np.asarray(x, dtype=np.float64)
        batch_n = x.shape[0]
        if batch_n == 0:
            return

        batch_mean = x.mean(axis=0)
        batch_m2 = ((x - batch_mean) ** 2).sum(axis=0)
        batch_min = x.min(axis=0)
        batch_max = x.max(axis=0)

        if self._n == 0:
            # First batch — just adopt its statistics directly.
            self._n = batch_n
            self._mean = batch_mean
            self._m2 = batch_m2
            np.copyto(self._min, batch_min)
            np.copyto(self._max, batch_max)
        else:
            # Parallel Welford merge.
            combined_n = self._n + batch_n
            delta = batch_mean - self._mean
            self._mean = (self._n * self._mean + batch_n * batch_mean) / combined_n
            self._m2 += batch_m2 + delta**2 * self._n * batch_n / combined_n
            np.minimum(self._min, batch_min, out=self._min)
            np.maximum(self._max, batch_max, out=self._max)
            self._n = combined_n

    # -- merge two accumulators ---------------------------------------------

    def merge(self, other: WelfordAccumulator) -> None:
        """Merge another accumulator into this one.

        Uses the parallel Welford formula to combine two sets of running
        statistics in O(dim) time.

        Args:
            other: Another accumulator with the same dimensionality.
        """
        if other._n == 0:
            return

        if self._n == 0:
            self._n = other._n
            self._mean = other._mean.copy()
            self._m2 = other._m2.copy()
            self._min = other._min.copy()
            self._max = other._max.copy()
            return

        combined_n = self._n + other._n
        delta = other._mean - self._mean
        self._mean = (self._n * self._mean + other._n * other._mean) / combined_n
        self._m2 += other._m2 + delta**2 * self._n * other._n / combined_n
        np.minimum(self._min, other._min, out=self._min)
        np.maximum(self._max, other._max, out=self._max)
        self._n = combined_n

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

    # -- serialization ------------------------------------------------------

    def to_dict(self) -> dict:
        """Export the raw accumulator state as a JSON-safe dict."""
        return {
            "n": self._n,
            "mean": self._mean.tolist(),
            "m2": self._m2.tolist(),
            "min": self._min.tolist(),
            "max": self._max.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> WelfordAccumulator:
        """Restore a :class:`WelfordAccumulator` from a dict produced by :meth:`to_dict`."""
        dim = len(d["mean"])
        acc = cls(dim)
        acc._n = d["n"]
        acc._mean = np.array(d["mean"], dtype=np.float64)
        acc._m2 = np.array(d["m2"], dtype=np.float64)
        acc._min = np.array(d["min"], dtype=np.float64)
        acc._max = np.array(d["max"], dtype=np.float64)
        return acc


# ---------------------------------------------------------------------------
# DatasetStats dataclass (final merged output)
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
# Per-episode accumulator state
# ---------------------------------------------------------------------------

_EPISODE_STATS_FILENAME = "stats.json"


@dataclass
class EpisodeAccumulatorState:
    """Raw Welford accumulator states for a single episode.

    Stores the intermediate accumulator state (n, mean, m2, min, max) rather
    than final statistics, so that multiple episodes can be merged accurately
    using the parallel Welford formula.

    Attributes:
        state: Accumulator state dict for the state vector.
        action: Accumulator state dict for the action vector.
        images: Per-camera accumulator state dicts for image channels.
        num_frames: Total number of JSONL data rows in the episode.
        image_samples: Number of video frames sampled for image stats.
    """

    state: dict
    action: dict
    images: dict[str, dict]
    num_frames: int = 0
    image_samples: int = 0

    def to_json(self) -> str:
        """Serialize to a JSON string."""
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, s: str) -> EpisodeAccumulatorState:
        """Deserialize from a JSON string."""
        data = json.loads(s)
        return cls(**data)


# ---------------------------------------------------------------------------
# Helpers
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


def _save_episode_stats(ep_dir: Path, ep_stats: EpisodeAccumulatorState) -> None:
    """Write per-episode stats to ``<ep_dir>/stats.json``."""
    ep_dir.mkdir(parents=True, exist_ok=True)
    stats_path = ep_dir / _EPISODE_STATS_FILENAME
    stats_path.write_text(ep_stats.to_json(), encoding="utf-8")


def _load_episode_stats(ep_dir: Path) -> EpisodeAccumulatorState | None:
    """Load per-episode stats from ``<ep_dir>/stats.json``, or return None."""
    stats_path = ep_dir / _EPISODE_STATS_FILENAME
    if stats_path.is_file():
        try:
            return EpisodeAccumulatorState.from_json(stats_path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Corrupt episode stats at {}; will recompute", stats_path)
            return None
    return None


# ---------------------------------------------------------------------------
# Per-episode stats computation
# ---------------------------------------------------------------------------


def _compute_single_episode_stats(
    manifest: DatasetManifest,
    episode: EpisodeEntry,
    dataset_root: Path,
    *,
    image_sample_count: int,
) -> EpisodeAccumulatorState:
    """Compute stats for a single episode.

    Reads every JSONL row for state/action stats. Uniformly samples up to
    *image_sample_count* video frames for per-channel image stats.

    Args:
        manifest: Parent dataset manifest (for dimensions and camera names).
        episode: The episode entry to process.
        dataset_root: Root directory of the dataset on disk.
        image_sample_count: Maximum number of video frames to sample from
            this episode for image statistics.

    Returns:
        :class:`EpisodeAccumulatorState` containing raw accumulator states.
    """
    ep_dir = dataset_root / episode.episode_dir
    state_acc = WelfordAccumulator(manifest.state_dim)
    action_acc = WelfordAccumulator(manifest.action_dim)

    # ------------------------------------------------------------------
    # 1. State / action stats from JSONL
    # ------------------------------------------------------------------
    data_path = ep_dir / episode.data_file
    num_frames = 0

    with open(data_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            state_acc.update(np.asarray(row["state"], dtype=np.float64))
            action_acc.update(np.asarray(row["action"], dtype=np.float64))
            num_frames += 1

    # ------------------------------------------------------------------
    # 2. Image stats from sampled video frames
    # ------------------------------------------------------------------
    camera_names = [cam.name for cam in manifest.cameras]
    image_accumulators: dict[str, dict] = {}
    actual_image_samples = 0

    if camera_names and num_frames > 0:
        # Determine video frame count from first camera
        first_cam = camera_names[0]
        first_video = ep_dir / episode.video_files[first_cam]
        video_info = get_video_info(first_video)
        num_video_frames = video_info.num_frames

        if num_video_frames > 0:
            sample_count = min(image_sample_count, num_video_frames)
            sample_indices = np.linspace(0, num_video_frames - 1, sample_count, dtype=int)
            sample_indices = np.unique(sample_indices).tolist()
            actual_image_samples = len(sample_indices)

            for cam_name in camera_names:
                acc = WelfordAccumulator(3)  # RGB channels
                video_path = ep_dir / episode.video_files[cam_name]
                frames_tensor = decode_frames(video_path, sample_indices)  # (N, C, H, W)

                # Batch-process all frames at once:
                # (N, C, H, W) -> (N, C, H*W) -> (N, H*W, C) -> (N*H*W, C)
                frames_np = frames_tensor.numpy()  # float32, [0, 1]
                n, c, h, w = frames_np.shape
                # Reshape to (N*H*W, C) in one shot
                pixels = frames_np.transpose(0, 2, 3, 1).reshape(-1, c)  # (N*H*W, C)
                acc.update_batch(pixels)

                image_accumulators[cam_name] = acc.to_dict()

    return EpisodeAccumulatorState(
        state=state_acc.to_dict(),
        action=action_acc.to_dict(),
        images=image_accumulators,
        num_frames=num_frames,
        image_samples=actual_image_samples,
    )


# ---------------------------------------------------------------------------
# Merge per-episode accumulators into final DatasetStats
# ---------------------------------------------------------------------------


def _merge_episode_states(
    episode_states: list[EpisodeAccumulatorState],
    camera_names: list[str],
    state_dim: int,
    action_dim: int,
) -> DatasetStats:
    """Merge a list of per-episode accumulator states into final :class:`DatasetStats`.

    Uses the parallel Welford formula to combine accumulators across episodes.

    Args:
        episode_states: Per-episode accumulator states.
        camera_names: Camera names from the manifest.
        state_dim: Dimensionality of the state vector.
        action_dim: Dimensionality of the action vector.

    Returns:
        Merged :class:`DatasetStats`.
    """
    state_acc = WelfordAccumulator(state_dim)
    action_acc = WelfordAccumulator(action_dim)
    image_accs: dict[str, WelfordAccumulator] = {name: WelfordAccumulator(3) for name in camera_names}

    for ep_state in episode_states:
        state_acc.merge(WelfordAccumulator.from_dict(ep_state.state))
        action_acc.merge(WelfordAccumulator.from_dict(ep_state.action))
        for cam_name in camera_names:
            if cam_name in ep_state.images:
                image_accs[cam_name].merge(WelfordAccumulator.from_dict(ep_state.images[cam_name]))

    image_stats: dict[str, dict[str, list[float]]] = {}
    for cam_name in camera_names:
        if image_accs[cam_name].count > 0:
            image_stats[cam_name] = _accumulator_to_dict(image_accs[cam_name])

    return DatasetStats(
        state=_accumulator_to_dict(state_acc),
        action=_accumulator_to_dict(action_acc),
        images=image_stats,
    )


# ---------------------------------------------------------------------------
# Compute stats with per-episode caching
# ---------------------------------------------------------------------------


def compute_stats(
    manifest: DatasetManifest,
    dataset_root: Path,
    *,
    source_dataset_root: Path | None = None,
    image_sample_count: int = 500,
) -> DatasetStats:
    """Compute normalization statistics over the entire dataset.

    Uses per-episode caching: each episode's stats are cached in
    ``<episode_dir>/stats.json``.  Only episodes missing a cache file are
    computed from scratch.  All per-episode accumulators are then merged
    to produce the final dataset-level statistics.

    When *source_dataset_root* is provided (e.g. when running against a
    snapshot copy), newly computed per-episode stats are also written back
    to the corresponding episode directory in the source dataset so that
    future snapshots benefit from the cache.

    Args:
        manifest: Parsed dataset manifest.
        dataset_root: Root directory of the dataset on disk.
        source_dataset_root: If not ``None``, also write per-episode stats
            to ``<source_dataset_root>/<episode_dir>/stats.json`` after
            computing them.  This enables cache persistence across snapshots.
        image_sample_count: Total number of video frames to sample across
            the entire dataset for image statistics.  Distributed
            proportionally across episodes.  Defaults to ``500``.

    Returns:
        Fully populated :class:`DatasetStats`.
    """
    total_episodes = len(manifest.episodes)
    if total_episodes == 0:
        return DatasetStats(
            state=_accumulator_to_dict(WelfordAccumulator(manifest.state_dim)),
            action=_accumulator_to_dict(WelfordAccumulator(manifest.action_dim)),
            images={},
        )

    # Distribute image samples proportionally across episodes.
    # First, count total JSONL rows per episode (cheap line count).
    episode_frame_counts: list[int] = []
    for episode in manifest.episodes:
        data_path = dataset_root / episode.episode_dir / episode.data_file
        count = _count_lines(data_path)
        episode_frame_counts.append(count)

    total_frames = sum(episode_frame_counts)

    # Allocate image samples proportionally to episode size.
    episode_image_samples: list[int] = []
    if total_frames > 0:
        for count in episode_frame_counts:
            share = count / total_frames
            episode_image_samples.append(max(1, int(round(share * image_sample_count))))
    else:
        episode_image_samples = [0] * total_episodes

    # Process each episode — use cache if available.
    episode_states: list[EpisodeAccumulatorState] = []
    computed_count = 0
    cached_count = 0

    log_interval = max(1, total_episodes // 10)

    for ep_idx, episode in enumerate(manifest.episodes):
        ep_dir = dataset_root / episode.episode_dir

        # Try loading cached stats — check local first, then source.
        cached = _load_episode_stats(ep_dir)
        if cached is None and source_dataset_root is not None:
            source_ep_dir = source_dataset_root / episode.episode_dir
            cached = _load_episode_stats(source_ep_dir)
            if cached is not None:
                # Copy from source into local so future reads are fast.
                _save_episode_stats(ep_dir, cached)

        if cached is not None:
            episode_states.append(cached)
            cached_count += 1
        else:
            ep_stats = _compute_single_episode_stats(
                manifest,
                episode,
                dataset_root,
                image_sample_count=episode_image_samples[ep_idx],
            )
            _save_episode_stats(ep_dir, ep_stats)
            # Write back to source so future snapshots benefit.
            if source_dataset_root is not None:
                source_ep_dir = source_dataset_root / episode.episode_dir
                if source_ep_dir.is_dir():
                    _save_episode_stats(source_ep_dir, ep_stats)
            episode_states.append(ep_stats)
            computed_count += 1

        if (ep_idx + 1) % log_interval == 0 or ep_idx == total_episodes - 1:
            pct = 100.0 * (ep_idx + 1) / total_episodes
            logger.info(
                "Stats progress: episode %d/%d (%.0f%%) — %d cached, %d computed",
                ep_idx + 1,
                total_episodes,
                pct,
                cached_count,
                computed_count,
            )

    logger.info(
        "Per-episode stats: %d episodes total (%d from cache, %d computed)",
        total_episodes,
        cached_count,
        computed_count,
    )

    # Merge all per-episode accumulators into final stats.
    camera_names = [cam.name for cam in manifest.cameras]
    result = _merge_episode_states(episode_states, camera_names, manifest.state_dim, manifest.action_dim)

    for cam_name in camera_names:
        if cam_name in result.images:
            logger.info(
                "Image stats for camera %r: mean=%s, std=%s",
                cam_name,
                result.images[cam_name]["mean"],
                result.images[cam_name]["std"],
            )

    return result


# ---------------------------------------------------------------------------
# Cache-aware loader
# ---------------------------------------------------------------------------

_CACHE_DIR = ".cache"
_CACHE_FILENAME = "stats.json"


def load_or_compute_stats(
    manifest: DatasetManifest,
    dataset_root: Path,
    *,
    source_dataset_root: Path | None = None,
) -> DatasetStats:
    """Load cached dataset stats or compute and cache them.

    The merged cache file is stored at ``<dataset_root>/.cache/stats.json``.
    It is considered valid when its modification time is newer than that of
    ``manifest.json``.  When stale or missing, per-episode caches in each
    ``<episode_dir>/stats.json`` are consulted — only episodes without a
    cache require video decoding.

    Args:
        manifest: Parsed dataset manifest.
        dataset_root: Root directory of the dataset on disk.
        source_dataset_root: If not ``None``, passed through to
            :func:`compute_stats` so that newly computed per-episode stats
            are also written back to the source dataset.

    Returns:
        :class:`DatasetStats` loaded from cache or freshly computed.
    """
    cache_path = dataset_root / _CACHE_DIR / _CACHE_FILENAME
    manifest_path = dataset_root / "manifest.json"

    # Fast path: merged dataset-level cache
    if cache_path.is_file() and manifest_path.is_file():
        manifest_mtime = manifest_path.stat().st_mtime
        cache_mtime = cache_path.stat().st_mtime
        if manifest_mtime < cache_mtime:
            logger.info("Loading cached stats from %s", cache_path)
            return DatasetStats.from_json(cache_path.read_text(encoding="utf-8"))
        logger.info("Merged cache is stale (manifest modified after cache); recomputing")
    else:
        logger.info("No merged cache found; computing stats (with per-episode caching)")

    stats = compute_stats(manifest, dataset_root, source_dataset_root=source_dataset_root)

    # Persist merged result
    cache_dir = dataset_root / _CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(stats.to_json(), encoding="utf-8")
    logger.info("Merged stats cached to %s", cache_path)

    return stats


# ---------------------------------------------------------------------------
# Background computation helper
# ---------------------------------------------------------------------------


def compute_episode_stats_background(
    manifest: DatasetManifest,
    episode: EpisodeEntry,
    dataset_root: Path,
    *,
    source_dataset_root: Path | None = None,
    image_sample_count: int = 100,
) -> None:
    """Compute and cache stats for a single episode (designed for background use).

    This function is safe to call from a background thread.  If the episode
    already has cached stats, this is a no-op.

    When *source_dataset_root* is provided, the computed stats are written to
    **both** the local ``dataset_root`` and the ``source_dataset_root``.  This
    ensures that even if the local directory (e.g. a recording cache) is
    deleted before the thread finishes, the source dataset still receives the
    cached stats.

    Args:
        manifest: Parent dataset manifest.
        episode: The episode entry to process.
        dataset_root: Root directory of the dataset.
        source_dataset_root: If not ``None``, also write per-episode stats
            to the corresponding episode directory in this path.
        image_sample_count: Number of video frames to sample for image stats
            within this episode.  Defaults to ``100``.
    """
    ep_dir = dataset_root / episode.episode_dir
    if _load_episode_stats(ep_dir) is not None:
        logger.debug("Episode stats already cached for {}", episode.episode_dir)
        # Still write to source if it doesn't have the stats yet.
        if source_dataset_root is not None:
            source_ep_dir = source_dataset_root / episode.episode_dir
            if source_ep_dir.is_dir() and _load_episode_stats(source_ep_dir) is None:
                cached = _load_episode_stats(ep_dir)
                if cached is not None:
                    _save_episode_stats(source_ep_dir, cached)
        return

    try:
        ep_stats = _compute_single_episode_stats(manifest, episode, dataset_root, image_sample_count=image_sample_count)
        _save_episode_stats(ep_dir, ep_stats)
        # Also write to source dataset so the cache persists across teardown.
        if source_dataset_root is not None:
            source_ep_dir = source_dataset_root / episode.episode_dir
            if source_ep_dir.is_dir():
                _save_episode_stats(source_ep_dir, ep_stats)
        logger.info("Background stats computed for episode {}", episode.episode_dir)
    except Exception:
        logger.warning(
            "Background stats computation failed for episode {}; will retry at training time",
            episode.episode_dir,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _count_lines(path: Path) -> int:
    """Count the number of lines in a file efficiently."""
    count = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            count += chunk.count(b"\n")
    return count


__all__ = [
    "DatasetStats",
    "EpisodeAccumulatorState",
    "WelfordAccumulator",
    "compute_episode_stats_background",
    "compute_stats",
    "load_or_compute_stats",
]
