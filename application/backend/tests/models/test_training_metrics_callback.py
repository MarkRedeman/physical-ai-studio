from unittest.mock import MagicMock, patch

import pytest
import torch

from models.training_metrics_callback import Average, TrainingMetricsCallback


class TinyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([2.0]))


def _make_trainer() -> MagicMock:
    trainer = MagicMock()
    trainer.current_epoch = 2
    trainer.global_step = 15
    trainer.num_training_batches = 10
    trainer.sanity_checking = False
    trainer.logger = MagicMock()
    trainer.optimizers = [MagicMock(param_groups=[{"lr": 0.001}])]
    trainer.log_every_n_steps = 1
    return trainer


class TestAverage:
    def test_get_returns_none_until_values_are_added(self):
        average = Average()

        assert average.get() is None

    def test_add_get_and_reset(self):
        average = Average()

        average.add(1.0)
        average.add(3.0)

        assert average.get() == 2.0
        average.reset()
        assert average.total == 0.0
        assert average.count == 0
        assert average.get() is None


class TestTrainingMetricsCallback:
    def test_logs_tensor_loss_step_metrics(self):
        callback = TrainingMetricsCallback(log_grad_norm=False)
        trainer = _make_trainer()
        module = TinyModule()

        callback.on_train_batch_start(trainer, module, None, 0)
        callback.on_train_batch_end(trainer, module, torch.tensor(3.5), None, 0)

        trainer.logger.log_metrics.assert_called_once()
        metrics = trainer.logger.log_metrics.call_args.args[0]
        assert metrics["train/loss_step"] == 3.5
        assert metrics["train/fractional_epoch"] == 1.5
        assert metrics["train/lr"] == 0.001
        assert metrics["system/accelerator_memory_mb"] == 0.0
        assert metrics["system/accelerator_memory_total_mb"] >= 0.0
        assert metrics["system/accelerator_memory_percent"] >= 0.0
        assert metrics["system/step_time_s"] >= 0.0
        assert "train/grad_norm" not in metrics

    def test_logs_dict_loss_action_error_and_grad_norm(self):
        callback = TrainingMetricsCallback()
        trainer = _make_trainer()
        module = TinyModule()
        module.weight.grad = torch.tensor([4.0])

        outputs = {"loss": torch.tensor(2.0), "action_error": torch.tensor(0.25)}
        callback.on_train_batch_start(trainer, module, None, 0)
        callback.on_train_batch_end(trainer, module, outputs, None, 0)

        metrics = trainer.logger.log_metrics.call_args.args[0]
        assert metrics["train/loss_step"] == 2.0
        assert metrics["train/action_error_step"] == 0.25
        assert metrics["train/grad_norm"] == 4.0

    def test_logs_every_step_during_dense_warmup(self):
        callback = TrainingMetricsCallback(log_grad_norm=False)
        trainer = _make_trainer()
        trainer.global_step = 2
        trainer.log_every_n_steps = 5
        module = TinyModule()

        callback.on_train_batch_start(trainer, module, None, 0)
        callback.on_train_batch_end(trainer, module, torch.tensor(3.5), None, 0)

        trainer.logger.log_metrics.assert_called_once()
        assert callback._train_loss.get() == 3.5

    def test_uses_trainer_log_every_n_steps_after_dense_warmup(self):
        callback = TrainingMetricsCallback(log_grad_norm=False)
        trainer = _make_trainer()
        trainer.global_step = 1001
        trainer.log_every_n_steps = 5
        module = TinyModule()

        callback.on_train_batch_start(trainer, module, None, 0)
        callback.on_train_batch_end(trainer, module, torch.tensor(3.5), None, 0)

        trainer.logger.log_metrics.assert_not_called()

        trainer.global_step = 1005
        callback.on_train_batch_start(trainer, module, None, 1)
        callback.on_train_batch_end(trainer, module, torch.tensor(2.5), None, 1)

        trainer.logger.log_metrics.assert_called_once()
        metrics = trainer.logger.log_metrics.call_args.args[0]
        assert metrics["train/loss_step"] == 2.5

    def test_always_logs_first_step_metrics(self):
        callback = TrainingMetricsCallback(log_grad_norm=False)
        trainer = _make_trainer()
        trainer.global_step = 1
        trainer.log_every_n_steps = 100
        module = TinyModule()

        callback.on_train_batch_start(trainer, module, None, 0)
        callback.on_train_batch_end(trainer, module, torch.tensor(3.5), None, 0)

        trainer.logger.log_metrics.assert_called_once()

    def test_logs_epoch_averages_and_clears_accumulators(self):
        callback = TrainingMetricsCallback()
        trainer = _make_trainer()
        module = TinyModule()

        callback._train_loss.add(1.0)
        callback._train_loss.add(3.0)
        callback._train_action_error.add(0.2)
        callback._train_action_error.add(0.4)
        callback.on_train_epoch_end(trainer, module)

        metrics = trainer.logger.log_metrics.call_args.args[0]
        assert metrics == {
            "train/epoch": 2.0,
            "train/loss_epoch": 2.0,
            "train/action_error_epoch": pytest.approx(0.3),
        }
        assert callback._train_loss.total == 0.0
        assert callback._train_loss.count == 0
        assert callback._train_action_error.total == 0.0
        assert callback._train_action_error.count == 0

    def test_validation_sanity_check_is_ignored(self):
        callback = TrainingMetricsCallback()
        trainer = _make_trainer()
        trainer.sanity_checking = True
        module = TinyModule()

        callback.on_validation_batch_end(trainer, module, {"loss": torch.tensor(1.0)}, None, 0)
        callback.on_validation_epoch_end(trainer, module)

        assert callback._val_loss.total == 0.0
        assert callback._val_loss.count == 0
        trainer.logger.log_metrics.assert_not_called()

    def test_logs_validation_epoch_metrics(self):
        callback = TrainingMetricsCallback()
        trainer = _make_trainer()
        module = TinyModule()

        callback.on_validation_batch_end(
            trainer,
            module,
            {"loss": torch.tensor(1.0), "action_error": torch.tensor(0.5)},
            None,
            0,
        )
        callback.on_validation_batch_end(
            trainer,
            module,
            {"loss": torch.tensor(3.0), "action_error": torch.tensor(0.7)},
            None,
            1,
        )
        callback.on_validation_epoch_end(trainer, module)

        metrics = trainer.logger.log_metrics.call_args.args[0]
        assert metrics == {
            "val/epoch": 2.0,
            "val/loss": 2.0,
            "val/action_error": pytest.approx(0.6),
        }

    def test_memory_fallback_returns_zero_on_cpu(self):
        with (
            patch("models.training_metrics_callback.torch.cuda.is_available", return_value=False),
            patch("models.training_metrics_callback.torch.xpu.is_available", return_value=False, create=True),
            patch("models.training_metrics_callback.torch.backends.mps.is_available", return_value=False),
        ):
            assert TrainingMetricsCallback._get_accelerator_memory_mb() == 0.0
            assert TrainingMetricsCallback._get_accelerator_memory_total_mb() == 0.0
            assert TrainingMetricsCallback._get_accelerator_memory_percent() == 0.0

    def test_cuda_memory_percent_uses_current_device(self):
        props = MagicMock(total_memory=20_000_000)
        with (
            patch("models.training_metrics_callback.torch.cuda.is_available", return_value=True),
            patch("models.training_metrics_callback.torch.cuda.current_device", return_value=1),
            patch("models.training_metrics_callback.torch.cuda.memory_allocated", return_value=5_000_000),
            patch("models.training_metrics_callback.torch.cuda.get_device_properties", return_value=props),
        ):
            assert TrainingMetricsCallback._get_accelerator_memory_mb() == 5.0
            assert TrainingMetricsCallback._get_accelerator_memory_total_mb() == 20.0
            assert TrainingMetricsCallback._get_accelerator_memory_percent() == 25.0

    def test_logs_nvml_utilization_and_power_when_available(self):
        callback = TrainingMetricsCallback(log_grad_norm=False)
        trainer = _make_trainer()
        module = TinyModule()

        with (
            patch.object(TrainingMetricsCallback, "_get_accelerator_utilization_percent", return_value=87.0),
            patch.object(TrainingMetricsCallback, "_get_accelerator_power_w", return_value=123.4),
        ):
            callback.on_train_batch_start(trainer, module, None, 0)
            callback.on_train_batch_end(trainer, module, torch.tensor(3.5), None, 0)

        metrics = trainer.logger.log_metrics.call_args.args[0]
        assert metrics["system/accelerator_utilization_percent"] == 87.0
        assert metrics["system/accelerator_power_w"] == 123.4

    def test_omits_nvml_metrics_when_unavailable(self):
        callback = TrainingMetricsCallback(log_grad_norm=False)
        trainer = _make_trainer()
        module = TinyModule()

        with (
            patch.object(TrainingMetricsCallback, "_get_accelerator_utilization_percent", return_value=None),
            patch.object(TrainingMetricsCallback, "_get_accelerator_power_w", return_value=None),
        ):
            callback.on_train_batch_start(trainer, module, None, 0)
            callback.on_train_batch_end(trainer, module, torch.tensor(3.5), None, 0)

        metrics = trainer.logger.log_metrics.call_args.args[0]
        assert "system/accelerator_utilization_percent" not in metrics
        assert "system/accelerator_power_w" not in metrics
