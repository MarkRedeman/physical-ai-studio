# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import tempfile
from pathlib import Path
from unittest.mock import Mock

import lightning
import pytest
import torch

from services.export_checkpoint import export_temporary_files, load_export_policy


class SmallPolicy(lightning.LightningModule):
    def __init__(self, width=4, compile_model=False):
        super().__init__()
        self.save_hyperparameters()
        self.linear = torch.nn.Linear(width, 2, dtype=torch.bfloat16)
        self.loaded_training_state = None

    def forward(self, observation):
        return self.linear(observation)

    def on_load_checkpoint(self, checkpoint):
        self.loaded_training_state = set(checkpoint) & {"optimizer_states", "lr_schedulers"}
        state = checkpoint["state_dict"]
        if "old_weight" in state:
            state["linear.weight"] = state.pop("old_weight")


def save_checkpoint(path, policy, *, legacy=False, old_keys=False):
    state = policy.state_dict()
    if old_keys:
        state["old_weight"] = state.pop("linear.weight")
    data = {
        "pytorch-lightning_version": lightning.__version__,
        "hyper_parameters": dict(policy.hparams),
        "state_dict": state,
        "optimizer_states": [{"state": {0: {"exp_avg": torch.zeros(64, 64)}}}],
        "lr_schedulers": [{"last_epoch": 12}],
    }
    torch.save(data, path, _use_new_zipfile_serialization=not legacy)


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("old_keys", [False, True])
def test_export_loader_matches_lightning_without_retaining_training_state(tmp_path, legacy, old_keys):
    torch.manual_seed(12)
    original = SmallPolicy(compile_model=True)
    checkpoint = tmp_path / "model.ckpt"
    save_checkpoint(checkpoint, original, legacy=legacy, old_keys=old_keys)
    before = checkpoint.read_bytes()
    standard = SmallPolicy.load_from_checkpoint(checkpoint, map_location="cpu", compile_model=False)
    exported = load_export_policy(checkpoint, SmallPolicy)
    assert exported.loaded_training_state == set()
    assert standard.loaded_training_state == {"optimizer_states", "lr_schedulers"}
    assert exported.hparams.compile_model is False
    for name, expected in standard.state_dict().items():
        actual = exported.state_dict()[name]
        assert actual.dtype == expected.dtype
        assert actual.device.type == "cpu"
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    observation = torch.randn(1, 4, dtype=torch.bfloat16)
    torch.testing.assert_close(exported(observation), standard(observation), rtol=0, atol=0)
    # The loaded policy has private model storage, not writable views of the checkpoint.
    with torch.no_grad():
        exported.linear.weight.add_(1)
    assert checkpoint.read_bytes() == before


@pytest.mark.parametrize("legacy", [False, True])
def test_checkpoint_is_loaded_once_with_cpu_mapping(tmp_path, monkeypatch, legacy):
    checkpoint = tmp_path / "model.ckpt"
    save_checkpoint(checkpoint, SmallPolicy(), legacy=legacy)
    load = Mock(wraps=torch.load)
    monkeypatch.setattr(torch, "load", load)
    load_export_policy(checkpoint, SmallPolicy)
    load.assert_called_once_with(checkpoint, map_location="cpu", mmap=not legacy, weights_only=False)


def test_shape_mismatch_is_not_silently_ignored(tmp_path):
    checkpoint = tmp_path / "model.ckpt"
    policy = SmallPolicy()
    save_checkpoint(checkpoint, policy)
    data = torch.load(checkpoint, weights_only=False)
    data["state_dict"]["linear.weight"] = torch.ones(10, 10)
    torch.save(data, checkpoint)
    with pytest.raises(RuntimeError, match="size mismatch"):
        load_export_policy(checkpoint, SmallPolicy)


@pytest.mark.parametrize("data", [{}, {"state_dict": []}])
def test_non_lightning_checkpoint_is_rejected(tmp_path, data):
    checkpoint = tmp_path / "model.ckpt"
    torch.save(data, checkpoint)
    with pytest.raises(ValueError, match="Lightning checkpoint"):
        load_export_policy(checkpoint, SmallPolicy)


@pytest.mark.parametrize("failed", [False, True])
def test_export_cleans_sidecars_and_restores_temp_directory_on_success_or_failure(tmp_path, failed):
    previous = tempfile.tempdir
    unrelated = tmp_path / "unrelated.onnx.data"
    unrelated.write_bytes(b"keep")

    def run_export():
        with export_temporary_files(tmp_path):
            directory = Path(tempfile.gettempdir())
            assert directory.parent == tmp_path
            with tempfile.NamedTemporaryFile(suffix=".onnx") as model:
                sidecar = Path(f"{model.name}.data")
                sidecar.write_bytes(b"weights")
            assert sidecar.exists()  # NamedTemporaryFile only deletes the .onnx file.
            if failed:
                raise RuntimeError("Conversion failed")
        assert not directory.exists()

    if failed:
        with pytest.raises(RuntimeError, match="Conversion failed"):
            run_export()
    else:
        run_export()
    assert tempfile.tempdir == previous
    assert unrelated.read_bytes() == b"keep"
    assert not list(tmp_path.glob(".export-*"))
