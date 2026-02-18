# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Video frame decoding with torchcodec (primary) and pyav (fallback) backends."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

_HAS_TORCHCODEC = False
_HAS_PYAV = False

try:
    from torchcodec.decoders import VideoDecoder  # noqa: F401

    _HAS_TORCHCODEC = True
except ImportError:
    pass

if not _HAS_TORCHCODEC:
    try:
        import av  # noqa: F401

        _HAS_PYAV = True
    except ImportError:
        pass


def _check_backend() -> None:
    """Raise ``ImportError`` if no video decoding backend is available."""
    if not _HAS_TORCHCODEC and not _HAS_PYAV:
        raise ImportError(
            "No video decoding backend found. Install torchcodec (`pip install torchcodec`) or pyav (`pip install av`)."
        )


@dataclass(frozen=True)
class VideoInfo:
    """Immutable container for video stream metadata.

    Attributes:
        num_frames: Total number of frames in the video.
        fps: Average frames per second.
        width: Frame width in pixels.
        height: Frame height in pixels.
        duration_s: Duration of the video in seconds.
        codec: Codec name (e.g. ``"h264"``).
    """

    num_frames: int
    fps: float
    width: int
    height: int
    duration_s: float
    codec: str


# ---------------------------------------------------------------------------
# torchcodec helpers
# ---------------------------------------------------------------------------


def _get_video_info_torchcodec(video_path: Path) -> VideoInfo:
    """Probe video metadata via torchcodec."""
    from torchcodec.decoders import VideoDecoder

    decoder = VideoDecoder(str(video_path))
    meta = decoder.metadata
    return VideoInfo(
        num_frames=meta.num_frames,
        fps=float(meta.average_fps),
        width=meta.width,
        height=meta.height,
        duration_s=float(meta.duration_seconds),
        codec=meta.codec,
    )


def _decode_frames_torchcodec(video_path: Path, frame_indices: list[int]) -> torch.Tensor:
    """Decode specific frames via torchcodec and return float32 NCHW tensor in [0, 1]."""
    from torchcodec.decoders import VideoDecoder

    decoder = VideoDecoder(str(video_path))
    num_frames = decoder.metadata.num_frames

    for idx in frame_indices:
        if idx < 0 or idx >= num_frames:
            raise IndexError(f"Frame index {idx} is out of bounds for video with {num_frames} frames.")

    frame_batch = decoder.get_frames_at(indices=frame_indices)
    # frame_batch.data is uint8 with shape (N, C, H, W)
    return frame_batch.data.to(torch.float32) / 255.0


# ---------------------------------------------------------------------------
# pyav helpers
# ---------------------------------------------------------------------------


def _get_video_info_pyav(video_path: Path) -> VideoInfo:
    """Probe video metadata via pyav."""
    import av

    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        # average_rate is a Fraction; convert to float
        fps = float(stream.average_rate) if stream.average_rate else 0.0
        num_frames = stream.frames
        width = stream.width
        height = stream.height
        codec = stream.codec_context.name

        # Compute duration: prefer stream duration, fall back to container duration.
        if stream.duration is not None and stream.time_base is not None:
            duration_s = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            # container.duration is in microseconds (AV_TIME_BASE)
            duration_s = container.duration / 1_000_000.0
        else:
            duration_s = num_frames / fps if fps > 0 else 0.0

        # If the container doesn't report frame count, estimate it.
        if num_frames == 0 and fps > 0:
            num_frames = int(round(duration_s * fps))

        return VideoInfo(
            num_frames=num_frames,
            fps=fps,
            width=width,
            height=height,
            duration_s=duration_s,
            codec=codec,
        )
    finally:
        container.close()


def _decode_frames_pyav(video_path: Path, frame_indices: list[int]) -> torch.Tensor:
    """Decode specific frames via pyav and return float32 NCHW tensor in [0, 1].

    For efficiency the function sorts the requested indices and performs
    forward seeks, avoiding full sequential scans when possible.
    """
    import av

    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"

        # Determine total frame count for bounds checking.
        total_frames = stream.frames
        if total_frames == 0:
            fps = float(stream.average_rate) if stream.average_rate else 0.0
            if stream.duration is not None and stream.time_base is not None:
                dur = float(stream.duration * stream.time_base)
            elif container.duration is not None:
                dur = container.duration / 1_000_000.0
            else:
                dur = 0.0
            total_frames = int(round(dur * fps)) if fps > 0 else 0

        for idx in frame_indices:
            if idx < 0 or idx >= total_frames:
                raise IndexError(f"Frame index {idx} is out of bounds for video with {total_frames} frames.")

        # Build a sorted order so we can seek forward through the file.
        sorted_order = sorted(range(len(frame_indices)), key=lambda i: frame_indices[i])
        target_set = set(frame_indices)

        # We will decode by walking through the stream.  For large gaps we
        # seek to the nearest keyframe before the target.
        results: dict[int, torch.Tensor] = {}
        frame_counter = 0
        remaining = set(frame_indices)

        # Seek-and-decode strategy: iterate through the stream once,
        # collecting all requested frames.
        min_target = min(frame_indices)
        if min_target > 0:
            # Seek to a position before the first target frame.
            time_base = stream.time_base
            fps = float(stream.average_rate) if stream.average_rate else 30.0
            target_pts = int(min_target / fps / time_base)
            container.seek(target_pts, stream=stream)

        for frame in container.decode(video=0):
            if frame_counter in target_set and frame_counter in remaining:
                rgb = frame.to_rgb().to_ndarray()  # (H, W, 3) uint8
                tensor = torch.from_numpy(rgb).permute(2, 0, 1).to(torch.float32) / 255.0
                results[frame_counter] = tensor
                remaining.discard(frame_counter)

            frame_counter += 1

            if not remaining:
                break

        # If any frames were missed (e.g. due to seek overshoot), do a
        # sequential fallback for the remaining ones.
        if remaining:
            container.seek(0, stream=stream)
            frame_counter = 0
            for frame in container.decode(video=0):
                if frame_counter in remaining:
                    rgb = frame.to_rgb().to_ndarray()
                    tensor = torch.from_numpy(rgb).permute(2, 0, 1).to(torch.float32) / 255.0
                    results[frame_counter] = tensor
                    remaining.discard(frame_counter)

                frame_counter += 1
                if not remaining:
                    break

        # Assemble output in the original request order.
        frames = [results[frame_indices[i]] for i in range(len(frame_indices))]
        return torch.stack(frames)
    finally:
        container.close()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_video_info(video_path: Path) -> VideoInfo:
    """Return metadata for a video file without decoding frames.

    Uses torchcodec when available, otherwise falls back to pyav.

    Args:
        video_path: Path to the video file.

    Returns:
        A :class:`VideoInfo` dataclass with stream metadata.

    Raises:
        FileNotFoundError: If *video_path* does not exist.
        RuntimeError: If the video cannot be read or parsed.
    """
    _check_backend()

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    try:
        if _HAS_TORCHCODEC:
            return _get_video_info_torchcodec(video_path)
        return _get_video_info_pyav(video_path)
    except FileNotFoundError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Failed to read video metadata from {video_path}: {exc}") from exc


def decode_frames(video_path: Path, frame_indices: list[int] | np.ndarray) -> torch.Tensor:
    """Decode specific frames from a video file by index.

    Uses torchcodec when available, otherwise falls back to pyav.

    Args:
        video_path: Path to the video file.
        frame_indices: Indices of the frames to decode.  May be
            non-sequential.

    Returns:
        A ``torch.Tensor`` of shape ``(N, C, H, W)`` in ``float32`` with
        values in ``[0, 1]`` and RGB channel order.

    Raises:
        FileNotFoundError: If *video_path* does not exist.
        RuntimeError: If the video cannot be read or decoded.
        IndexError: If any frame index is out of bounds.
    """
    _check_backend()

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if isinstance(frame_indices, np.ndarray):
        frame_indices = frame_indices.tolist()

    if len(frame_indices) == 0:
        return torch.empty(0, 3, 0, 0, dtype=torch.float32)

    try:
        if _HAS_TORCHCODEC:
            return _decode_frames_torchcodec(video_path, frame_indices)
        return _decode_frames_pyav(video_path, frame_indices)
    except (FileNotFoundError, IndexError):
        raise
    except Exception as exc:
        raise RuntimeError(f"Failed to decode frames from {video_path}: {exc}") from exc


def decode_frame(video_path: Path, frame_index: int) -> torch.Tensor:
    """Decode a single frame from a video file.

    Convenience wrapper around :func:`decode_frames`.

    Args:
        video_path: Path to the video file.
        frame_index: Zero-based index of the frame to decode.

    Returns:
        A ``torch.Tensor`` of shape ``(C, H, W)`` in ``float32`` with
        values in ``[0, 1]`` and RGB channel order.

    Raises:
        FileNotFoundError: If *video_path* does not exist.
        RuntimeError: If the video cannot be read or decoded.
        IndexError: If the frame index is out of bounds.
    """
    return decode_frames(video_path, [frame_index])[0]


__all__ = [
    "VideoInfo",
    "get_video_info",
    "decode_frames",
    "decode_frame",
]
