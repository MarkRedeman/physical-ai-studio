from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import pytest
from physicalai.inference import ExecutionCapabilities
from physicalai.inference.callbacks import RTCLatencyTracker
from physicalai.runtime import AsyncExecution, ChunkedActionQueue, RTCActionQueue, RTCExecution, SyncExecution

from runtime_session.bus import Bus
from runtime_session.controls.model import ModelControl
from runtime_session.model import ModelInputs
from runtime_session.store import FeatureSpec, Store
from schemas.runtime_session import ExecutionConfig, ModelControlConfig
from workers.model_worker import Empty, Inference, ModelWorker, PrepareExecutionRequest


def test_worker_delegates_preparation_and_capabilities_to_runtime():
    model = Mock(execution_capabilities=ExecutionCapabilities(("sync", "async", "rtc"), 4, 2))
    metrics = Mock()
    inference = Inference(model, Mock(), {}, metrics=metrics)
    inference.prepare_execution(PrepareExecutionRequest(mode="rtc"))
    model.configure_execution.assert_called_once_with("rtc")
    model.reset.assert_not_called()  # Runtime owns model reset during configuration.
    metrics.reset.assert_called_once()
    ready = inference.ready(Empty())
    assert ready.supported_execution_modes == ["sync", "async", "rtc"]
    assert (ready.chunk_size, ready.action_dim) == (4, 2)
    inference.prepare_execution(PrepareExecutionRequest(mode="async", generation=2))
    with pytest.raises(ValueError, match="Stale"):
        inference.prepare_execution(PrepareExecutionRequest(mode="rtc", generation=1))
    assert model.configure_execution.call_count == 2
    model.configure_execution.assert_called_with("async")


def test_failed_preparation_does_not_consume_generation_or_reset_metrics():
    model = Mock(configure_execution=Mock(side_effect=ValueError("Unsupported")))
    metrics = Mock()
    inference = Inference(model, Mock(), {}, metrics=metrics)
    request = PrepareExecutionRequest(mode="rtc")
    with pytest.raises(ValueError, match="Unsupported"):
        inference.prepare_execution(request)
    metrics.reset.assert_not_called()
    model.configure_execution.side_effect = None
    inference.prepare_execution(request)
    metrics.reset.assert_called_once()


async def test_worker_checks_runtime_action_dimensions_against_the_follower(tmp_path, monkeypatch):
    store = Store.create(FeatureSpec(("a", "b")), uuid4().hex)
    model = Mock(execution_capabilities=ExecutionCapabilities(("sync", "async", "rtc"), 4, 3))
    monkeypatch.setattr(ModelWorker, "open_model", lambda _self: model)
    worker = ModelWorker(
        tmp_path,
        backend="openvino",
        device="CPU",
        session_id=uuid4().hex,
        store_name=store.name,
        spec=store.spec,
        images={},
    )
    try:
        with Bus.host(worker.session_id), pytest.raises(ValueError, match="follower's joints"):
            async with worker.lifecycle():
                pytest.fail("An incompatible model must not become ready")
    finally:
        store.unlink()


@pytest.mark.parametrize("mode", ["sync", "async", "rtc"])
def test_control_selects_matching_executor_queue_and_prepares_before_connect(mode):
    store = Store.create(FeatureSpec(("a", "b")), uuid4().hex)
    calls = []
    model = SimpleNamespace(
        inputs=ModelInputs(2, {}),
        supported_execution_modes=["sync", "async", "rtc"],
        chunk_size=4,
        latency_tracker=RTCLatencyTracker(),
        prepare_execution=lambda selected: calls.append(selected),
    )
    try:
        control = ModelControl(store, model, ModelControlConfig(execution=ExecutionConfig(mode=mode)))
        expected = {"sync": SyncExecution, "async": AsyncExecution, "rtc": RTCExecution}
        assert isinstance(control.source._execution, expected[mode])
        assert isinstance(control.source.action_queue, RTCActionQueue if mode == "rtc" else ChunkedActionQueue)
        if mode == "rtc":
            assert control.config.execution.execution_horizon == 2
            assert control.source._execution._latency_tracker is model.latency_tracker
        control.source.connect = lambda **kwargs: calls.append("connect")
        control.on_start()
        assert calls == [mode, "connect"]
        with pytest.raises(ValueError, match="half"):
            ModelControl(store, model, ModelControlConfig(execution=ExecutionConfig(mode="rtc", execution_horizon=3)))
    finally:
        store.unlink()
