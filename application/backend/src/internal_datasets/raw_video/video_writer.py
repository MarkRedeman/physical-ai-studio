# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Video writer that pipes raw RGB frames to an ffmpeg subprocess.

Each :class:`VideoWriter` instance manages a single ffmpeg process that
encodes frames into an MP4 file (H.264 / libx264).  Frames are written to
the process stdin as raw bytes, which avoids any intermediate disk I/O for
individual frames.

Typical usage::

    writer = VideoWriter(Path("episode_000/cam_top.mp4"), fps=30, width=640, height=480)
    writer.start()
    for frame in frames:
        writer.write_frame(frame)   # numpy HWC RGB uint8
    writer.close()                  # blocks until ffmpeg finishes
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Final

import numpy as np
from loguru import logger

# ffmpeg pixel format that corresponds to numpy HWC RGB uint8 arrays.
_INPUT_PIX_FMT: Final[str] = "rgb24"


class VideoWriter:
    """Encode a stream of RGB frames to an MP4 file via ffmpeg stdin pipe.

    Args:
        output_path: Destination ``.mp4`` file path.  Parent directory must
            exist before :meth:`start` is called.
        fps: Target frames per second for the output video.
        width: Frame width in pixels.
        height: Frame height in pixels.
        codec: Video codec to use (default ``"libx264"``).
        preset: Encoding speed/quality preset (default ``"fast"``).
        crf: Constant Rate Factor for quality control (default ``23``).
            Lower values mean higher quality / larger files.
    """

    def __init__(
        self,
        output_path: Path,
        *,
        fps: int,
        width: int,
        height: int,
        codec: str = "libx264",
        preset: str = "fast",
        crf: int = 23,
    ) -> None:
        self._output_path = output_path
        self._fps = fps
        self._width = width
        self._height = height
        self._codec = codec
        self._preset = preset
        self._crf = crf
        self._process: subprocess.Popen | None = None
        self._frame_count: int = 0

    @property
    def output_path(self) -> Path:
        """Destination MP4 file path."""
        return self._output_path

    @property
    def frame_count(self) -> int:
        """Number of frames written so far."""
        return self._frame_count

    def start(self) -> None:
        """Spawn the ffmpeg subprocess.

        Raises:
            RuntimeError: If the writer is already started.
            FileNotFoundError: If the output directory does not exist.
        """
        if self._process is not None:
            raise RuntimeError("VideoWriter already started")

        if not self._output_path.parent.is_dir():
            raise FileNotFoundError(f"Output directory does not exist: {self._output_path.parent}")

        cmd = [
            "ffmpeg",
            "-y",  # overwrite output
            "-loglevel",
            "error",
            # Input format: raw video from stdin
            "-f",
            "rawvideo",
            "-pix_fmt",
            _INPUT_PIX_FMT,
            "-s",
            f"{self._width}x{self._height}",
            "-r",
            str(self._fps),
            "-i",
            "pipe:0",
            # Output encoding
            "-c:v",
            self._codec,
            "-preset",
            self._preset,
            "-crf",
            str(self._crf),
            "-pix_fmt",
            "yuv420p",  # broad compatibility
            "-movflags",
            "+faststart",  # enable streaming
            str(self._output_path),
        ]

        logger.debug("Starting ffmpeg: {}", " ".join(cmd))
        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._frame_count = 0

    def write_frame(self, frame: np.ndarray) -> None:
        """Write a single frame to the video.

        Args:
            frame: A numpy array of shape ``(H, W, 3)`` with dtype ``uint8``
                in RGB channel order.

        Raises:
            RuntimeError: If the writer is not started or ffmpeg has exited.
            ValueError: If the frame shape does not match the expected
                dimensions.
        """
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("VideoWriter is not started — call start() first")

        if self._process.poll() is not None:
            stderr = self._process.stderr.read().decode() if self._process.stderr else ""
            raise RuntimeError(f"ffmpeg process exited unexpectedly (code {self._process.returncode}): {stderr}")

        expected_shape = (self._height, self._width, 3)
        if frame.shape != expected_shape:
            raise ValueError(f"Frame shape {frame.shape} does not match expected {expected_shape}")

        if frame.dtype != np.uint8:
            raise ValueError(f"Frame dtype must be uint8, got {frame.dtype}")

        try:
            self._process.stdin.write(frame.tobytes())
        except BrokenPipeError as exc:
            stderr = self._process.stderr.read().decode() if self._process.stderr else ""
            raise RuntimeError(f"ffmpeg stdin pipe broken: {stderr}") from exc

        self._frame_count += 1

    def close(self, timeout: float = 30.0) -> None:
        """Close stdin and wait for ffmpeg to finish encoding.

        Args:
            timeout: Maximum seconds to wait for ffmpeg to exit.

        Raises:
            RuntimeError: If ffmpeg exits with a non-zero return code or
                times out.
        """
        if self._process is None:
            return

        try:
            if self._process.stdin is not None:
                self._process.stdin.close()

            self._process.wait(timeout=timeout)

            if self._process.returncode != 0:
                stderr = self._process.stderr.read().decode() if self._process.stderr else ""
                raise RuntimeError(
                    f"ffmpeg exited with code {self._process.returncode} for {self._output_path}: {stderr}"
                )

            logger.debug(
                "VideoWriter finished: {} ({} frames)",
                self._output_path,
                self._frame_count,
            )
        except subprocess.TimeoutExpired:
            self._process.kill()
            raise RuntimeError(f"ffmpeg timed out after {timeout}s for {self._output_path}")
        finally:
            # Ensure subprocess resources are cleaned up
            if self._process.stdout:
                self._process.stdout.close()
            if self._process.stderr:
                self._process.stderr.close()
            self._process = None

    def abort(self) -> None:
        """Kill the ffmpeg process immediately without waiting.

        Safe to call even if the writer was never started or is already
        closed.
        """
        if self._process is None:
            return

        try:
            self._process.kill()
            self._process.wait(timeout=5.0)
        except Exception:
            pass
        finally:
            for stream in (self._process.stdin, self._process.stdout, self._process.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except Exception:
                        pass
            self._process = None

    @property
    def is_running(self) -> bool:
        """Whether the ffmpeg process is currently alive."""
        return self._process is not None and self._process.poll() is None

    def __del__(self) -> None:
        """Safety net: abort if the writer was not properly closed."""
        if self._process is not None:
            self.abort()


__all__ = ["VideoWriter"]
