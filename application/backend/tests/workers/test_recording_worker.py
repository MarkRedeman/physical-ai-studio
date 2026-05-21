# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import multiprocessing as mp
import queue
from unittest.mock import MagicMock

import pytest

from workers.recording_worker import RecordingWorker


@pytest.fixture
def worker(test_dataset):
    return RecordingWorker(
        dataset=test_dataset,
        data_manifest=MagicMock(),
        mp_terminate_event=mp.Event(),
        event_queue=mp.Queue(),
    )


class TestEpisodeControl:
    def test_get_task_returns_empty_string_by_default(self, worker):
        assert worker.get_task() == ""

    def test_start_episode_writes_task_name(self, worker):
        worker.start_episode("collect_blocks")
        assert worker.get_task() == "collect_blocks"

    def test_start_episode_sets_start_event(self, worker):
        worker.start_episode("task")
        assert worker._start_event.is_set()

    def test_save_episode_sets_save_event(self, worker):
        worker.save_episode()
        assert worker._save_event.is_set()

    def test_discard_episode_sets_discard_event(self, worker):
        worker.discard_episode()
        assert worker._discard_event.is_set()

    def test_task_name_truncated_at_255_bytes(self, worker):
        long_task = "x" * 300
        worker.start_episode(long_task)
        assert len(worker.get_task()) <= 255


class TestHandleStartRecording:
    @pytest.mark.anyio
    async def test_sets_is_recording(self, worker):
        worker._start_event.set()
        await worker._handle_start_recording()
        assert worker._is_recording is True

    @pytest.mark.anyio
    async def test_emits_start_recording_event(self, worker):
        event_queue = queue.Queue()
        worker.event_queue = event_queue
        worker._start_event.set()
        await worker._handle_start_recording()
        msg = event_queue.get_nowait()
        assert msg["event"] == "start_recording"
        assert msg["state"]["is_recording"] is True

    @pytest.mark.anyio
    async def test_clears_start_event_after_handling(self, worker):
        worker._start_event.set()
        await worker._handle_start_recording()
        assert not worker._start_event.is_set()

    @pytest.mark.anyio
    async def test_noop_when_event_not_set(self, worker):
        await worker._handle_start_recording()
        assert worker._is_recording is False


class TestHandleSaveEpisode:
    @pytest.mark.anyio
    async def test_saves_episode_and_stops_recording(self, worker):
        mutation = MagicMock()
        worker.recording_mutation = mutation
        worker._is_recording = True
        worker._save_event.set()
        await worker._handle_save_episode()
        mutation.save_episode.assert_called_once()
        assert worker._is_recording is False

    @pytest.mark.anyio
    async def test_increments_episode_counter(self, worker):
        worker.recording_mutation = MagicMock()
        worker._is_recording = True
        worker._save_event.set()
        await worker._handle_save_episode()
        assert worker._episodes_recorded == 1

    @pytest.mark.anyio
    async def test_emits_save_episode_event(self, worker):
        event_queue = queue.Queue()
        worker.event_queue = event_queue
        worker.recording_mutation = MagicMock()
        worker._is_recording = True
        worker._save_event.set()
        await worker._handle_save_episode()
        msg = event_queue.get_nowait()
        assert msg["event"] == "save_episode"
        assert msg["state"]["is_recording"] is False
        assert msg["state"]["episodes_recorded"] == 1

    @pytest.mark.anyio
    async def test_noop_when_no_mutation(self, worker):
        worker.recording_mutation = None
        worker._save_event.set()
        await worker._handle_save_episode()
        assert worker._episodes_recorded == 0


class TestHandleDiscardEpisode:
    @pytest.mark.anyio
    async def test_discards_buffer_and_stops_recording(self, worker):
        mutation = MagicMock()
        worker.recording_mutation = mutation
        worker._is_recording = True
        worker._discard_event.set()
        await worker._handle_discard_episode()
        mutation.discard_buffer.assert_called_once()
        assert worker._is_recording is False

    @pytest.mark.anyio
    async def test_emits_discard_episode_event(self, worker):
        event_queue = queue.Queue()
        worker.event_queue = event_queue
        worker.recording_mutation = MagicMock()
        worker._discard_event.set()
        await worker._handle_discard_episode()
        msg = event_queue.get_nowait()
        assert msg["event"] == "discard_episode"
        assert msg["state"]["is_recording"] is False


class TestTeardown:
    @pytest.mark.anyio
    async def test_calls_mutation_teardown(self, worker):
        mutation = MagicMock()
        worker.recording_mutation = mutation
        await worker.teardown()
        mutation.teardown.assert_called_once()

    @pytest.mark.anyio
    async def test_no_error_when_mutation_is_none(self, worker):
        worker.recording_mutation = None
        await worker.teardown()  # should not raise
