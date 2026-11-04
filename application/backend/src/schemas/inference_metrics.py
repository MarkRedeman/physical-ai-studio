"""Live model timing sent over a runtime session's existing websocket."""

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field


class InferenceMetricsMessage(BaseModel):
    """One completed action-chunk prediction, excluding worker communication."""

    type: Literal["inference_metrics"] = "inference_metrics"
    model_id: UUID
    backend: str
    device: str
    run_id: UUID
    """Changes when the model resets, so timing from previous runs cannot be reused."""
    timestamp: float = Field(ge=0, allow_inf_nan=False)
    """Prediction completion time as UTC Unix seconds."""
    latency_ms: float = Field(ge=0, allow_inf_nan=False)
    average_latency_ms: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    sample_count: int = Field(ge=0, le=20)
    """Number of recent steady-state predictions in the average; warm-up is excluded."""
    prediction_count: int = Field(ge=1)
    is_warmup: bool
