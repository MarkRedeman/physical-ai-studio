# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import multiprocessing as mp
import queue
from unittest.mock import MagicMock

import pytest

from workers.model_integration_worker import ModelIntegration


@pytest.fixture
def worker(test_model):
    return ModelIntegration(
        model=test_model,
        backend="torch",
        data_manifest=MagicMock(),
        mp_terminate_event=mp.Event(),
        event_queue=mp.Queue(),
    )


class TestTaskControl:
    def test_get_task_returns_empty_string_by_default(self, worker):
        assert worker.get_task() == ""

    def test_start_task_writes_task_name(self, worker):
        worker.start_task("pick_and_place")
        assert worker.get_task() == "pick_and_place"

    def test_start_task_sets_start_event(self, worker):
        worker.start_task("foo")
        assert worker._start_task_event.is_set()

    def test_stop_task_sets_stop_event(self, worker):
        worker.stop_task()
        assert worker._stop_task_event.is_set()

    def test_task_name_truncated_at_255_bytes(self, worker):
        long_task = "a" * 300
        worker.start_task(long_task)
        assert len(worker.get_task()) <= 255


class TestHandleStartTask:
    @pytest.mark.anyio
    async def test_sets_is_running(self, worker):
        worker._start_task_event.set()
        await worker._handle_start_task()
        assert worker.is_running is True

    @pytest.mark.anyio
    async def test_emits_start_task_event(self, worker):
        event_queue = queue.Queue()
        worker.event_queue = event_queue
        worker._start_task_event.set()
        await worker._handle_start_task()
        msg = event_queue.get_nowait()
        assert msg["event"] == "start_task"
        assert msg["state"]["is_running"] is True

    @pytest.mark.anyio
    async def test_clears_start_event_after_handling(self, worker):
        worker._start_task_event.set()
        await worker._handle_start_task()
        assert not worker._start_task_event.is_set()

    @pytest.mark.anyio
    async def test_noop_when_event_not_set(self, worker):
        await worker._handle_start_task()
        assert worker.is_running is False


class TestHandleStopTask:
    @pytest.mark.anyio
    async def test_clears_is_running(self, worker):
        worker.is_running = True
        worker._stop_task_event.set()
        await worker._handle_stop_task()
        assert worker.is_running is False

    @pytest.mark.anyio
    async def test_emits_stop_task_event(self, worker):
        event_queue = queue.Queue()
        worker.event_queue = event_queue
        worker.is_running = True
        worker._stop_task_event.set()
        await worker._handle_stop_task()
        msg = event_queue.get_nowait()
        assert msg["event"] == "stop_task"
        assert msg["state"]["is_running"] is False

    @pytest.mark.anyio
    async def test_noop_when_event_not_set(self, worker):
        worker.is_running = True
        await worker._handle_stop_task()
        assert worker.is_running is True


class TestTeardown:
    @pytest.mark.anyio
    async def test_stops_child_workers(self, worker):
        child = MagicMock()
        worker._child_workers = [child]
        await worker.teardown()
        child.stop.assert_called_once()

    @pytest.mark.anyio
    async def test_calls_model_integration_teardown(self, worker):
        mock_integration = MagicMock()
        worker.model_integration = mock_integration
        await worker.teardown()
        mock_integration.teardown.assert_called_once()

    @pytest.mark.anyio
    async def test_no_error_when_model_integration_is_none(self, worker):
        worker.model_integration = None
        await worker.teardown()  # should not raise
