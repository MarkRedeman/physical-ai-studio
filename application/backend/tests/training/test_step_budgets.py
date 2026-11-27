"""Exercise partial epochs and SnapFlow boundaries with real Lightning on CPU."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import lightning
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from training.job import CHECKPOINT_NAME, SNAPFLOW_CHECKPOINT_NAME, TrainingJobSpec, run_training_job


class TinyPolicy(lightning.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.distilling = False
        self.teacher_batches = 0
        self.distill_batches = 0

    def training_step(self, batch, batch_idx):
        if self.distilling:
            self.distill_batches += 1
        else:
            self.teacher_batches += 1
        return self.weight.square()

    def validation_step(self, batch, batch_idx):
        self.log("val/loss", self.weight.square())

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)

    def enable_snapflow(self, **kwargs):
        self.distilling = True


class TinyDataModule(lightning.LightningDataModule):
    def __init__(self, validation: bool) -> None:
        super().__init__()
        self.validation = validation

    def train_dataloader(self):
        return DataLoader(TensorDataset(torch.ones(20)), batch_size=1)

    def val_dataloader(self):
        return DataLoader(TensorDataset(torch.ones(2)), batch_size=1) if self.validation else []


@pytest.mark.parametrize("distill", [False, True])
@pytest.mark.parametrize("validation", [False, True])
def test_step_limited_run_saves_partial_epoch_checkpoints(tmp_path: Path, distill: bool, validation: bool) -> None:
    policy = TinyPolicy()
    report = MagicMock()
    spec = TrainingJobSpec(
        policy="pi05" if distill else "act",
        max_steps=5,
        snapflow_start_step=3 if distill else None,
        device_type="cpu",
        precision="32-true",
    )
    with (
        patch("physicalai.data.LeRobotDataModule", return_value=TinyDataModule(validation)),
        patch("training.job.build_policy", return_value=policy),
        patch("physicalai.train.callbacks.PolicyDatasetInteraction._interact_policy_dataset"),
        patch("training.job._export_policy", return_value=policy),
    ):
        run_training_job(
            spec,
            dataset_root=tmp_path / "dataset",
            output_dir=tmp_path / "model",
            cache_dir=tmp_path / "cache",
            report=report,
            should_stop=lambda: False,
        )
    assert (policy.teacher_batches, policy.distill_batches) == ((3, 2) if distill else (5, 0))
    assert (tmp_path / "model" / CHECKPOINT_NAME).is_file()
    assert (tmp_path / "model" / SNAPFLOW_CHECKPOINT_NAME).is_file() == distill
    assert any(call.args[0] == 100 for call in report.call_args_list)
