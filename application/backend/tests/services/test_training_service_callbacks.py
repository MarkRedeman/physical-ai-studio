from unittest.mock import MagicMock

from services.training_service import (
    TrainingLogCallback,
    TrainingTrackingCallback,
    _get_total_training_steps,
    _should_emit_step_update,
)


def test_get_total_training_steps_uses_max_steps_when_available():
    trainer = MagicMock(max_steps=100, estimated_stepping_batches=250)

    assert _get_total_training_steps(trainer) == 100


def test_get_total_training_steps_falls_back_to_estimated_batches():
    trainer = MagicMock(max_steps=-1, estimated_stepping_batches=250)

    assert _get_total_training_steps(trainer) == 250


def test_tracking_callback_uses_safe_total_steps():
    trainer = MagicMock(max_steps=-1, estimated_stepping_batches=200, global_step=50, log_every_n_steps=1)
    dispatcher = MagicMock()
    callback = TrainingTrackingCallback(
        shutdown_event=MagicMock(is_set=MagicMock(return_value=False)),
        interrupt_event=MagicMock(is_set=MagicMock(return_value=False)),
        dispatcher=dispatcher,
    )

    callback.on_train_batch_end(trainer, MagicMock(), {"loss": None}, None, 0)

    dispatcher.update_progress.assert_called_once_with(25, extra_info={"train/loss_step": None})


def test_tracking_callback_logs_every_step_during_dense_warmup():
    trainer = MagicMock(max_steps=100, estimated_stepping_batches=100, global_step=2, log_every_n_steps=5)
    dispatcher = MagicMock()
    callback = TrainingTrackingCallback(
        shutdown_event=MagicMock(is_set=MagicMock(return_value=False)),
        interrupt_event=MagicMock(is_set=MagicMock(return_value=False)),
        dispatcher=dispatcher,
    )

    callback.on_train_batch_end(trainer, MagicMock(), {"loss": None}, None, 0)

    dispatcher.update_progress.assert_called_once()


def test_tracking_callback_uses_trainer_log_every_n_steps_after_dense_warmup():
    trainer = MagicMock(max_steps=2000, estimated_stepping_batches=2000, global_step=1001, log_every_n_steps=5)
    dispatcher = MagicMock()
    callback = TrainingTrackingCallback(
        shutdown_event=MagicMock(is_set=MagicMock(return_value=False)),
        interrupt_event=MagicMock(is_set=MagicMock(return_value=False)),
        dispatcher=dispatcher,
    )

    callback.on_train_batch_end(trainer, MagicMock(), {"loss": None}, None, 0)

    dispatcher.update_progress.assert_not_called()


def test_tracking_callback_still_checks_interrupt_when_update_is_throttled():
    trainer = MagicMock(max_steps=2000, estimated_stepping_batches=2000, global_step=1001, log_every_n_steps=5)
    callback = TrainingTrackingCallback(
        shutdown_event=MagicMock(is_set=MagicMock(return_value=False)),
        interrupt_event=MagicMock(is_set=MagicMock(return_value=True)),
        dispatcher=MagicMock(),
    )

    callback.on_train_batch_end(trainer, MagicMock(), {"loss": None}, None, 0)

    assert trainer.should_stop is True


def test_should_emit_step_update_uses_trainer_log_every_n_steps():
    assert _should_emit_step_update(MagicMock(global_step=1000, log_every_n_steps=100)) is True
    assert _should_emit_step_update(MagicMock(global_step=1001, log_every_n_steps=5)) is False
    assert _should_emit_step_update(MagicMock(global_step=1005, log_every_n_steps=5)) is True


def test_log_callback_uses_safe_total_steps_for_cadence():
    trainer = MagicMock(max_steps=-1, estimated_stepping_batches=2000)
    callback = TrainingLogCallback()

    callback.on_fit_start(trainer, MagicMock())

    assert callback.every_n_steps == 2
