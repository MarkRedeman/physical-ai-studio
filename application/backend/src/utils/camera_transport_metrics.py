"""Low-overhead aggregate metrics for camera transport diagnostics."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class CameraTransportMetrics:
    """Accumulate measurements and periodically return a summary."""

    interval_s: float = 5.0
    started_at: float = field(default_factory=time.perf_counter)
    frames: int = 0
    jpeg_bytes: int = 0
    encode_s: float = 0.0
    queue_s: float = 0.0
    send_s: float = 0.0

    def add_encode(self, elapsed_s: float, jpeg_size: int) -> None:
        self.frames += 1
        self.jpeg_bytes += jpeg_size
        self.encode_s += elapsed_s

    def add_frame(self) -> None:
        self.frames += 1

    def add_queue_delay(self, elapsed_s: float) -> None:
        self.queue_s += elapsed_s

    def add_send(self, elapsed_s: float) -> None:
        self.send_s += elapsed_s

    def summary(self) -> dict[str, float] | None:
        """Return interval measurements, resetting the accumulator when due."""
        now = time.perf_counter()
        elapsed_s = now - self.started_at
        if elapsed_s < self.interval_s:
            return None

        frames = self.frames
        summary = {
            "fps": frames / elapsed_s,
            "jpeg_kib_per_frame": self.jpeg_bytes / frames / 1024 if frames else 0.0,
            "encode_ms_per_frame": self.encode_s / frames * 1000 if frames else 0.0,
            "queue_ms_per_frame": self.queue_s / frames * 1000 if frames else 0.0,
            "send_ms_per_frame": self.send_s / frames * 1000 if frames else 0.0,
            "mib_per_s": self.jpeg_bytes / elapsed_s / (1024 * 1024),
        }
        self.started_at = now
        self.frames = 0
        self.jpeg_bytes = 0
        self.encode_s = 0.0
        self.queue_s = 0.0
        self.send_s = 0.0
        return summary
