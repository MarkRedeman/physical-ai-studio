"""Bounded timing statistics published by the model worker, not the control thread."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from loguru import logger

from schemas.inference_metrics import InferenceMetricsMessage

if TYPE_CHECKING:
    from collections.abc import Callable

METRICS = "evt/model/metrics"
_PUBLISH_INTERVAL_S = 0.25
_WINDOW = 20


@dataclass
class InferenceMetrics:
    """Average recent successful predictions and emit at most four updates per second."""

    model_id: UUID
    backend: str
    device: str
    publish: Callable[[InferenceMetricsMessage], None]
    _run_id: UUID = field(default_factory=uuid4)
    _samples: deque[float] = field(default_factory=lambda: deque(maxlen=_WINDOW))
    _prediction_count: int = 0
    _published_at: float | None = None

    def reset(self) -> None:
        """Start a new timing series when PolicySource resets the model for a run."""
        self._run_id = uuid4()
        self._samples.clear()
        self._prediction_count = 0
        self._published_at = None

    def record(self, latency_ms: float) -> None:
        """Record a completed prediction; the first queue-seeding call is warm-up."""
        self._prediction_count += 1
        is_warmup = self._prediction_count == 1
        if not is_warmup:
            self._samples.append(latency_ms)
        now = time.monotonic()
        if self._published_at is not None and now - self._published_at < _PUBLISH_INTERVAL_S:
            return
        self._published_at = now
        message = InferenceMetricsMessage(
            model_id=self.model_id,
            backend=self.backend,
            device=self.device,
            run_id=self._run_id,
            timestamp=time.time(),
            latency_ms=latency_ms,
            average_latency_ms=sum(self._samples) / len(self._samples) if self._samples else None,
            sample_count=len(self._samples),
            prediction_count=self._prediction_count,
            is_warmup=is_warmup,
        )
        try:
            self.publish(message)
        except Exception:
            # Timing is observational: publishing it must not fail an otherwise valid prediction.
            logger.opt(exception=True).warning("Could not publish inference timing")
