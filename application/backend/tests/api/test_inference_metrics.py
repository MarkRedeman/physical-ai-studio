from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from api.runtime_ws import _forward_events
from schemas.inference_metrics import InferenceMetricsMessage


async def test_inference_metrics_are_forwarded_over_the_existing_socket() -> None:
    message = InferenceMetricsMessage(
        model_id=uuid4(),
        backend="onnx",
        device="cpu",
        run_id=uuid4(),
        timestamp=1000.0,
        latency_ms=42.0,
        average_latency_ms=38.0,
        sample_count=20,
        prediction_count=21,
        is_warmup=False,
    )
    events = SimpleNamespace(next=AsyncMock(side_effect=[message, RuntimeError("End of test stream")]))
    socket = SimpleNamespace(send_json=AsyncMock())
    with pytest.raises(RuntimeError, match="End of test stream"):
        await _forward_events(socket, events)
    socket.send_json.assert_awaited_once_with(message.model_dump(mode="json"))
