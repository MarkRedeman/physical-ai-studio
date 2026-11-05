from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import numpy as np
import pytest

from runtime_session import inference_metrics as metrics_module
from runtime_session.inference_metrics import InferenceMetrics
from schemas.inference_metrics import InferenceMetricsMessage
from workers import model_worker
from workers.model_worker import Empty, Inference, InferRequest


@pytest.fixture
def timing(monkeypatch):
    clock = Mock(return_value=0.0)
    monkeypatch.setattr(metrics_module, "time", SimpleNamespace(monotonic=clock, time=lambda: 1000.0))
    messages: list[InferenceMetricsMessage] = []
    metrics = InferenceMetrics(uuid4(), "onnx", "cpu", messages.append)
    return metrics, messages, clock


def test_warmup_is_reported_but_excluded_from_the_average(timing) -> None:
    metrics, messages, clock = timing
    metrics.record(800.0)
    assert messages[-1].is_warmup
    assert messages[-1].average_latency_ms is None
    assert messages[-1].sample_count == 0

    clock.return_value = 0.1
    metrics.record(40.0)
    assert len(messages) == 1  # High-rate samples contribute without flooding the websocket.
    clock.return_value = 0.3
    metrics.record(60.0)
    assert messages[-1].latency_ms == 60.0
    assert messages[-1].average_latency_ms == 50.0
    assert messages[-1].sample_count == 2
    assert messages[-1].prediction_count == 3
    assert not messages[-1].is_warmup


def test_average_window_is_bounded(timing) -> None:
    metrics, messages, clock = timing
    metrics.record(1000.0)
    for value in range(1, 31):
        clock.return_value = value * 0.3
        metrics.record(float(value))
    assert messages[-1].sample_count == 20
    assert messages[-1].average_latency_ms == 20.5  # Last 20 samples: 11 through 30.


def test_reset_starts_a_new_series_and_publishes_warmup_immediately(timing) -> None:
    metrics, messages, _clock = timing
    metrics.record(10.0)
    previous_run = messages[-1].run_id
    metrics.reset()
    metrics.record(300.0)
    assert messages[-1].run_id != previous_run
    assert messages[-1].is_warmup
    assert messages[-1].prediction_count == 1
    assert messages[-1].average_latency_ms is None


def test_a_telemetry_publisher_failure_does_not_fail_prediction(timing) -> None:
    metrics, _messages, _clock = timing
    metrics.publish = Mock(side_effect=RuntimeError("Telemetry unavailable"))
    metrics.record(42.0)
    metrics.publish.assert_called_once()


def test_worker_measures_the_model_call_and_preserves_actions(timing, monkeypatch) -> None:
    metrics, messages, _clock = timing
    monkeypatch.setattr(model_worker, "time", SimpleNamespace(perf_counter=Mock(side_effect=[10.0, 10.042])))
    predictor = Mock(predict_action_chunk=Mock(return_value=np.array([[1.0, 2.0]])))
    store = Mock()
    inference = Inference(predictor, store, {}, metrics=metrics)
    result = inference.infer(InferRequest(state=[0.0, 0.0], task="Pick up the cube"))
    assert result.actions == [[1.0, 2.0]]
    assert messages[-1].latency_ms == pytest.approx(42.0)
    store.read_observation.assert_not_called()
    inference.reset(Empty())
    predictor.reset.assert_called_once()


def test_failed_predictions_do_not_emit_latency(timing, monkeypatch) -> None:
    metrics, messages, _clock = timing
    monkeypatch.setattr(model_worker, "time", SimpleNamespace(perf_counter=lambda: 10.0))
    inference = Inference(
        Mock(predict_action_chunk=Mock(side_effect=RuntimeError("Prediction failed"))), Mock(), {}, metrics
    )
    with pytest.raises(RuntimeError, match="Prediction failed"):
        inference.infer(InferRequest(state=[0.0]))
    assert messages == []


def test_rtc_inputs_are_preserved_while_reporting_inference_latency(timing, monkeypatch) -> None:
    metrics, messages, _clock = timing
    monkeypatch.setattr(model_worker, "time", SimpleNamespace(perf_counter=Mock(side_effect=[2.0, 2.025])))
    predictor = Mock(predict_action_chunk=Mock(return_value=np.array([[1.0, 2.0]])))
    inference = Inference(predictor, Mock(), {}, metrics=metrics)
    result = inference.infer(
        InferRequest(
            state=[0.0, 0.0],
            prev_chunk_left_over=[[[1.0, 2.0], [3.0, 4.0]]],
            inference_delay=1,
            execution_horizon=2,
            max_guidance_weight=5.0,
        )
    )
    observation = predictor.predict_action_chunk.call_args.args[0]
    assert observation["prev_chunk_left_over"].shape == (1, 2, 2)
    assert observation["prev_chunk_left_over"].dtype == np.float32
    assert observation["inference_delay"].dtype == np.int64
    assert observation["execution_horizon"].dtype == np.int64
    assert observation["max_guidance_weight"].dtype == np.float32
    assert result.actions == [[1.0, 2.0]]
    assert messages[-1].latency_ms == pytest.approx(25.0)
