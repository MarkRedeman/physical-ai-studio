# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Frame alignment table and episode boundary tracking.

Maps a flat global frame index space ``[0, total_frames)`` to per-episode
video frame indices and JSONL data-row indices.  When the video frame count
differs from the number of data rows, nearest-neighbour resampling is used
to align data rows to video frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .manifest import DatasetManifest
from .video_decode import get_video_info


@dataclass(frozen=True)
class EpisodeBoundary:
    """Immutable descriptor for one episode's position in the global index space.

    Attributes:
        episode_idx: Zero-based episode index.
        start: Global start index (inclusive).
        end: Global end index (exclusive).
        num_video_frames: Number of frames in the video file.
        num_data_rows: Number of rows in the JSONL data file.
        video_fps: Actual FPS of the video file.
    """

    episode_idx: int
    start: int
    end: int
    num_video_frames: int
    num_data_rows: int
    video_fps: float


class FrameIndex:
    """O(1) global-to-episode frame lookup table.

    Builds flat numpy arrays so that ``lookup(idx)`` is a constant-time
    indexing operation rather than a binary search.

    Args:
        manifest: Parsed and validated dataset manifest.
        dataset_root: Path to the top-level dataset directory containing
            episode sub-directories.

    Raises:
        ValueError: If cameras within an episode have different frame counts.
    """

    _episode_ids: np.ndarray
    """Shape ``(total_frames,)`` — maps global idx to episode index."""

    _video_frame_ids: np.ndarray
    """Shape ``(total_frames,)`` — maps global idx to video frame index."""

    _data_row_ids: np.ndarray
    """Shape ``(total_frames,)`` — maps global idx to JSONL data row index."""

    _boundaries: list[EpisodeBoundary]
    _total_frames: int

    def __init__(self, manifest: DatasetManifest, dataset_root: Path) -> None:
        episode_id_chunks: list[np.ndarray] = []
        video_frame_id_chunks: list[np.ndarray] = []
        data_row_id_chunks: list[np.ndarray] = []
        boundaries: list[EpisodeBoundary] = []

        global_offset = 0

        for ep_idx, episode in enumerate(manifest.episodes):
            ep_dir = dataset_root / episode.episode_dir

            # ----------------------------------------------------------
            # Determine video frame count (must agree across cameras)
            # ----------------------------------------------------------
            frame_counts: dict[str, int] = {}
            video_fps: float | None = None

            for camera_name, video_filename in episode.video_files.items():
                video_path = ep_dir / video_filename
                info = get_video_info(video_path)
                frame_counts[camera_name] = info.num_frames

                if video_fps is None:
                    video_fps = info.fps
                # Use first camera's FPS as representative; they should match.

            unique_counts = set(frame_counts.values())
            if len(unique_counts) != 1:
                details = ", ".join(f"{cam}={count}" for cam, count in sorted(frame_counts.items()))
                raise ValueError(
                    f"Episode {episode.episode_dir!r} has cameras with different "
                    f"frame counts: {details}. All cameras in an episode must "
                    f"have the same number of frames."
                )

            num_video_frames = unique_counts.pop()
            assert video_fps is not None  # guaranteed by manifest validation (≥1 camera)

            # ----------------------------------------------------------
            # Count JSONL data rows (cheap line count, no parsing)
            # ----------------------------------------------------------
            data_path = ep_dir / episode.data_file
            num_data_rows = _count_lines(data_path)

            # ----------------------------------------------------------
            # Build per-frame arrays for this episode
            # ----------------------------------------------------------
            video_frame_ids = np.arange(num_video_frames, dtype=np.int64)
            data_row_ids = _nearest_neighbor_resample(num_video_frames, num_data_rows)

            episode_id_chunks.append(np.full(num_video_frames, ep_idx, dtype=np.int64))
            video_frame_id_chunks.append(video_frame_ids)
            data_row_id_chunks.append(data_row_ids)

            boundaries.append(
                EpisodeBoundary(
                    episode_idx=ep_idx,
                    start=global_offset,
                    end=global_offset + num_video_frames,
                    num_video_frames=num_video_frames,
                    num_data_rows=num_data_rows,
                    video_fps=video_fps,
                )
            )

            global_offset += num_video_frames

        # ----------------------------------------------------------
        # Concatenate into flat arrays
        # ----------------------------------------------------------
        if episode_id_chunks:
            self._episode_ids = np.concatenate(episode_id_chunks)
            self._video_frame_ids = np.concatenate(video_frame_id_chunks)
            self._data_row_ids = np.concatenate(data_row_id_chunks)
        else:
            self._episode_ids = np.empty(0, dtype=np.int64)
            self._video_frame_ids = np.empty(0, dtype=np.int64)
            self._data_row_ids = np.empty(0, dtype=np.int64)

        self._boundaries = boundaries
        self._total_frames = global_offset

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def total_frames(self) -> int:
        """Total number of frames across all episodes."""
        return self._total_frames

    @property
    def num_episodes(self) -> int:
        """Number of episodes in the dataset."""
        return len(self._boundaries)

    @property
    def boundaries(self) -> list[EpisodeBoundary]:
        """Episode boundary descriptors (read-only copy)."""
        return list(self._boundaries)

    # ------------------------------------------------------------------
    # Lookup methods
    # ------------------------------------------------------------------

    def lookup(self, global_idx: int) -> tuple[int, int, int]:
        """Map a global frame index to episode-local indices.

        This is an O(1) operation backed by pre-computed numpy arrays.

        Args:
            global_idx: Index in ``[0, total_frames)``.

        Returns:
            A tuple of ``(episode_idx, video_frame_idx, data_row_idx)``.

        Raises:
            IndexError: If *global_idx* is out of range.
        """
        if global_idx < 0 or global_idx >= self._total_frames:
            raise IndexError(f"Global index {global_idx} out of range [0, {self._total_frames})")

        return (
            int(self._episode_ids[global_idx]),
            int(self._video_frame_ids[global_idx]),
            int(self._data_row_ids[global_idx]),
        )

    def get_episode_range(self, episode_idx: int) -> tuple[int, int]:
        """Return the ``(start, end)`` global indices for an episode.

        Args:
            episode_idx: Zero-based episode index.

        Returns:
            ``(start, end)`` where *start* is inclusive and *end* is exclusive.

        Raises:
            IndexError: If *episode_idx* is out of range.
        """
        if episode_idx < 0 or episode_idx >= len(self._boundaries):
            raise IndexError(f"Episode index {episode_idx} out of range [0, {len(self._boundaries)})")

        boundary = self._boundaries[episode_idx]
        return boundary.start, boundary.end

    def get_episode_for_global_idx(self, global_idx: int) -> int:
        """Return the episode index that contains the given global frame index.

        This is an O(1) operation.

        Args:
            global_idx: Index in ``[0, total_frames)``.

        Returns:
            Zero-based episode index.

        Raises:
            IndexError: If *global_idx* is out of range.
        """
        if global_idx < 0 or global_idx >= self._total_frames:
            raise IndexError(f"Global index {global_idx} out of range [0, {self._total_frames})")

        return int(self._episode_ids[global_idx])


# ======================================================================
# Internal helpers
# ======================================================================


def _nearest_neighbor_resample(num_video_frames: int, num_data_rows: int) -> np.ndarray:
    """Compute data-row indices for each video frame using nearest-neighbour resampling.

    Args:
        num_video_frames: Number of frames in the video (N).
        num_data_rows: Number of rows in the JSONL data file (M).

    Returns:
        Array of shape ``(num_video_frames,)`` with dtype ``int64``,
        where element *i* is the JSONL row index for video frame *i*.
    """
    if num_video_frames == 0:
        return np.empty(0, dtype=np.int64)

    if num_video_frames == 1:
        return np.zeros(1, dtype=np.int64)

    if num_data_rows == num_video_frames:
        return np.arange(num_video_frames, dtype=np.int64)

    # For video frame i: data_row = round(i * (M - 1) / (N - 1))
    frame_indices = np.arange(num_video_frames, dtype=np.float64)
    scale = (num_data_rows - 1) / (num_video_frames - 1)
    return np.rint(frame_indices * scale).astype(np.int64)


def _count_lines(path: Path) -> int:
    """Count the number of lines in a file efficiently.

    Args:
        path: Path to the file.

    Returns:
        Number of lines (i.e. number of newline-terminated records).
    """
    count = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            count += chunk.count(b"\n")
    return count


__all__ = [
    "EpisodeBoundary",
    "FrameIndex",
]
