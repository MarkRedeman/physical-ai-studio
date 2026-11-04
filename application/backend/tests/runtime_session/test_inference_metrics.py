import asyncio
from unittest.mock import patch

import pytest

from runtime_session.inference_metrics import METRICS
from schemas.inference_metrics import InferenceMetricsMessage
from schemas.runtime_session import ExecutionConfig, ModelControlConfig
from tests.workers.fake_model import FakeModelWorker

from .fixtures import make_environment
from .test_model import _model


@pytest.fixture
def fake_model_worker():
    with patch("runtime_session.model.ModelWorker", FakeModelWorker):
        yield


@pytest.mark.parametrize("mode", ["sync", "async"])
async def test_model_worker_emits_metrics_in_both_execution_modes(
    new_session, fake_cameras, fake_model_worker, tmp_path, mode
) -> None:
    session = await new_session()
    await session.load_environment(make_environment())
    model = _model(tmp_path)
    await session.load_model(model, "torch", "cpu")
    with session.bus.subscribe(METRICS, InferenceMetricsMessage) as metrics:
        await session.set_control(ModelControlConfig(hz=50, execution=ExecutionConfig(mode=mode), grace_s=0))
        first = await asyncio.wait_for(metrics.next(), timeout=5)
        assert first.model_id == model.id
        assert first.backend == "torch"
        assert first.is_warmup
        assert first.average_latency_ms is None
        steady = await asyncio.wait_for(metrics.next(), timeout=5)
        assert not steady.is_warmup
        assert steady.average_latency_ms is not None
        assert steady.latency_ms >= 0
        assert steady.run_id == first.run_id
        await session.set_control(None)
