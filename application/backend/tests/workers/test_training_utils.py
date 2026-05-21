# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import multiprocessing as mp
import queue
from unittest.mock import MagicMock
from uuid import uuid4

from workers.training.utils import TrainingLogCallback, TrainingTrackingCallback, TrainingTrackingDispatcher


class TestTrainingLogCallbackAutoInterval:
    def test_zero_steps_returns_one(self):
        assert TrainingLogCallback._auto_every_n_steps(0) == 1

    def test_negative_steps_returns_one(self):
        assert TrainingLogCallback._auto_every_n_steps(-10) == 1

    def test_small_run_returns_one(self):
        # 100 steps: 100 // 1000 = 0 → max(1, 0) = 1 → min(100, 1) = 1
        assert TrainingLogCallback._auto_every_n_steps(100) == 1

    def test_medium_run_scales_with_steps(self):
        # 10_000 steps: 10_000 // 1000 = 10 → min(100, 10) = 10
        assert TrainingLogCallback._auto_every_n_steps(10_000) == 10

    def test_large_run_capped_at_100(self):
        assert TrainingLogCallback._auto_every_n_steps(500_000) == 100

    def test_never_exceeds_100(self):
        assert TrainingLogCallback._auto_every_n_steps(10_000_000) <= 100

    def test_never_below_one(self):
        for n in [0, 1, 50, 999]:
            assert TrainingLogCallback._auto_every_n_steps(n) >= 1


class TestTrainingTrackingCallback:
    def _make(self):
        stop_event = mp.Event()
        interrupt_event = mp.Event()
        dispatcher = MagicMock()
        cb = TrainingTrackingCallback(
            shutdown_event=stop_event,
            interrupt_event=interrupt_event,
            dispatcher=dispatcher,
        )
        return cb, stop_event, interrupt_event, dispatcher

    def _make_trainer(self, global_step=50, max_steps=100):
        trainer = MagicMock()
        trainer.global_step = global_step
        trainer.max_steps = max_steps
        trainer.should_stop = False
        return trainer

    def test_on_train_batch_end_calls_dispatcher(self):
        cb, _, _, dispatcher = self._make()
        trainer = self._make_trainer(global_step=50, max_steps=100)
        cb.on_train_batch_end(trainer, MagicMock(), {}, None, None)
        dispatcher.update_progress.assert_called_once_with(50, extra_info={"train/loss_step": None})

    def test_progress_computed_as_percentage(self):
        cb, _, _, dispatcher = self._make()
        trainer = self._make_trainer(global_step=25, max_steps=100)
        cb.on_train_batch_end(trainer, MagicMock(), {}, None, None)
        progress = dispatcher.update_progress.call_args[0][0]
        assert progress == 25

    def test_stops_trainer_on_shutdown_event(self):
        cb, stop_event, _, _ = self._make()
        stop_event.set()
        trainer = self._make_trainer()
        cb.on_train_batch_end(trainer, MagicMock(), {}, None, None)
        assert trainer.should_stop is True

    def test_stops_trainer_on_interrupt_event(self):
        cb, _, interrupt_event, _ = self._make()
        interrupt_event.set()
        trainer = self._make_trainer()
        cb.on_train_batch_end(trainer, MagicMock(), {}, None, None)
        assert trainer.should_stop is True

    def test_does_not_stop_trainer_when_no_events_set(self):
        cb, _, _, _ = self._make()
        trainer = self._make_trainer()
        cb.on_train_batch_end(trainer, MagicMock(), {}, None, None)
        assert trainer.should_stop is False


class TestTrainingTrackingDispatcher:
    def _make(self):
        interrupt_event = mp.Event()
        dispatcher = TrainingTrackingDispatcher(
            job_id=uuid4(),
            event_queue=MagicMock(),
            interrupt_event=interrupt_event,
        )
        dispatcher.queue = queue.Queue()  # replace mp.Queue with synchronous queue for tests
        return dispatcher

    def test_update_progress_puts_on_internal_queue(self):
        dispatcher = self._make()
        dispatcher.update_progress(42, {"train/loss_step": 0.5})
        item = dispatcher.queue.get_nowait()
        assert item == (42, {"train/loss_step": 0.5})

    def test_update_progress_preserves_extra_info(self):
        dispatcher = self._make()
        extra = {"train/loss_step": 1.23, "lr": 0.001}
        dispatcher.update_progress(10, extra)
        _, received_extra = dispatcher.queue.get_nowait()
        assert received_extra == extra
