import time
from dataclasses import dataclass
from functools import cache
from typing import Any

import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only

DENSE_METRIC_LOGGING_STEPS = 1000


@dataclass
class Average:
    total: float = 0.0
    count: int = 0

    def add(self, value: float) -> None:
        self.total += value
        self.count += 1

    def get(self) -> float | None:
        if self.count == 0:
            return None
        return self.total / self.count

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0


class TrainingMetricsCallback(Callback):
    """Log VLA training dashboard metrics to the configured Lightning logger."""

    def __init__(self, log_grad_norm: bool = True):
        super().__init__()
        self.log_grad_norm = log_grad_norm
        self._train_loss = Average()
        self._train_action_error = Average()
        self._val_loss = Average()
        self._val_action_error = Average()
        self._step_start_time: float | None = None

    @staticmethod
    def _get_accelerator_memory_mb() -> float:
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated(torch.cuda.current_device()) / 1e6
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.xpu.memory_allocated(torch.xpu.current_device()) / 1e6
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.mps.driver_allocated_memory() / 1e6
        return 0.0

    @staticmethod
    def _get_accelerator_memory_total_mb() -> float:
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / 1e6
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.xpu.get_device_properties(torch.xpu.current_device()).total_memory / 1e6
        return 0.0

    @classmethod
    def _get_accelerator_memory_percent(cls) -> float:
        total_memory_mb = cls._get_accelerator_memory_total_mb()
        if total_memory_mb == 0.0:
            return 0.0
        return cls._get_accelerator_memory_mb() / total_memory_mb * 100.0

    @staticmethod
    @cache
    def _get_nvml() -> Any | None:
        try:
            import pynvml

            pynvml.nvmlInit()
            return pynvml
        except Exception:
            return None

    @staticmethod
    def _get_cuda_nvml_handle() -> Any | None:
        nvml = TrainingMetricsCallback._get_nvml()
        if nvml is None or not torch.cuda.is_available():
            return None

        try:
            device_index = torch.cuda.current_device()
            device_uuid = torch.cuda.get_device_properties(device_index).uuid
            return nvml.nvmlDeviceGetHandleByUUID(device_uuid)
        except Exception:
            try:
                return nvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
            except Exception:
                return None

    @staticmethod
    def _get_accelerator_utilization_percent() -> float | None:
        handle = TrainingMetricsCallback._get_cuda_nvml_handle()
        if handle is None:
            return None

        nvml = TrainingMetricsCallback._get_nvml()
        try:
            return float(nvml.nvmlDeviceGetUtilizationRates(handle).gpu)
        except Exception:
            return None

    @staticmethod
    def _get_accelerator_power_w() -> float | None:
        handle = TrainingMetricsCallback._get_cuda_nvml_handle()
        if handle is None:
            return None

        nvml = TrainingMetricsCallback._get_nvml()
        try:
            return float(nvml.nvmlDeviceGetPowerUsage(handle)) / 1000.0
        except Exception:
            return None

    @staticmethod
    def _get_lr(trainer: Trainer) -> float:
        try:
            return float(trainer.optimizers[0].param_groups[0]["lr"])
        except (IndexError, KeyError):
            return float("nan")

    @staticmethod
    def _get_grad_norm(pl_module: LightningModule) -> float:
        total_norm = 0.0
        for parameter in pl_module.parameters():
            if parameter.grad is not None:
                total_norm += parameter.grad.detach().data.norm(2).item() ** 2
        return total_norm**0.5

    @staticmethod
    def _fractional_epoch(trainer: Trainer) -> float:
        steps_per_epoch = trainer.num_training_batches
        if steps_per_epoch == 0:
            return 0.0
        return trainer.global_step / steps_per_epoch

    @staticmethod
    def _extract_scalar(outputs: Any, key: str) -> float | None:
        if isinstance(outputs, dict):
            value = outputs.get(key)
        elif key == "loss" and isinstance(outputs, torch.Tensor):
            value = outputs
        else:
            return None

        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value.detach().float().item()
        return float(value)

    @staticmethod
    def _should_log_step(trainer: Trainer) -> bool:
        global_step = trainer.global_step
        if global_step <= DENSE_METRIC_LOGGING_STEPS:
            return True

        log_every_n_steps = trainer.log_every_n_steps
        if log_every_n_steps is None or log_every_n_steps <= 0:
            return True
        return global_step % log_every_n_steps == 0

    def on_train_batch_start(
        self,
        trainer: Trainer,  # noqa: ARG002
        pl_module: LightningModule,  # noqa: ARG002
        batch: Any,  # noqa: ARG002
        batch_idx: int,  # noqa: ARG002
    ) -> None:
        self._step_start_time = time.perf_counter()

    @rank_zero_only
    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,  # noqa: ARG002
        batch_idx: int,  # noqa: ARG002
    ) -> None:
        if trainer.logger is None:
            return

        loss = self._extract_scalar(outputs, "loss")
        if loss is not None:
            self._train_loss.add(loss)

        action_error = self._extract_scalar(outputs, "action_error")
        if action_error is not None:
            self._train_action_error.add(action_error)

        if not self._should_log_step(trainer):
            return

        step_time = (
            time.perf_counter() - self._step_start_time if self._step_start_time is not None else float("nan")
        )

        metrics: dict[str, float] = {
            "train/fractional_epoch": self._fractional_epoch(trainer),
            "train/lr": self._get_lr(trainer),
            "system/accelerator_memory_mb": self._get_accelerator_memory_mb(),
            "system/accelerator_memory_total_mb": self._get_accelerator_memory_total_mb(),
            "system/accelerator_memory_percent": self._get_accelerator_memory_percent(),
            "system/step_time_s": step_time,
        }
        accelerator_utilization_percent = self._get_accelerator_utilization_percent()
        if accelerator_utilization_percent is not None:
            metrics["system/accelerator_utilization_percent"] = accelerator_utilization_percent
        accelerator_power_w = self._get_accelerator_power_w()
        if accelerator_power_w is not None:
            metrics["system/accelerator_power_w"] = accelerator_power_w
        if loss is not None:
            metrics["train/loss_step"] = loss
        if action_error is not None:
            metrics["train/action_error_step"] = action_error
        if self.log_grad_norm:
            metrics["train/grad_norm"] = self._get_grad_norm(pl_module)

        trainer.logger.log_metrics(metrics, step=trainer.global_step)

    @rank_zero_only
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:  # noqa: ARG002
        if trainer.logger is None:
            return

        metrics: dict[str, float] = {"train/epoch": float(trainer.current_epoch)}
        train_loss = self._train_loss.get()
        if train_loss is not None:
            metrics["train/loss_epoch"] = train_loss
            self._train_loss.reset()
        train_action_error = self._train_action_error.get()
        if train_action_error is not None:
            metrics["train/action_error_epoch"] = train_action_error
            self._train_action_error.reset()

        trainer.logger.log_metrics(metrics, step=trainer.global_step)

    @rank_zero_only
    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,  # noqa: ARG002
        outputs: Any,
        batch: Any,  # noqa: ARG002
        batch_idx: int,  # noqa: ARG002
        dataloader_idx: int = 0,  # noqa: ARG002
    ) -> None:
        if trainer.sanity_checking:
            return

        loss = self._extract_scalar(outputs, "loss")
        if loss is not None:
            self._val_loss.add(loss)

        action_error = self._extract_scalar(outputs, "action_error")
        if action_error is not None:
            self._val_action_error.add(action_error)

    @rank_zero_only
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:  # noqa: ARG002
        if trainer.sanity_checking or trainer.logger is None:
            return

        metrics: dict[str, float] = {"val/epoch": float(trainer.current_epoch)}
        val_loss = self._val_loss.get()
        if val_loss is not None:
            metrics["val/loss"] = val_loss
            self._val_loss.reset()
        val_action_error = self._val_action_error.get()
        if val_action_error is not None:
            metrics["val/action_error"] = val_action_error
            self._val_action_error.reset()

        trainer.logger.log_metrics(metrics, step=trainer.global_step)
