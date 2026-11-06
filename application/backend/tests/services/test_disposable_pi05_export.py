# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Exercise the real Pi05 checkpoint/export wrapper with a tiny, download-free inner model."""

from unittest.mock import Mock
from uuid import uuid4

import lightning
import numpy as np
import pytest
import torch
from physicalai.policies.mixins.rtc import RTCModelMixin
from physicalai.policies.pi05 import Pi05

from schemas.job import OptimizeModelJobPayload
from schemas.model import Model
from schemas.model_variant import VariantInfo, read_package
from services.model_optimization_service import ModelOptimizationService


class TinyPi05Model(RTCModelMixin, torch.nn.Module):
    def __init__(self, dataset_stats, **kwargs):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.chunk_size = kwargs["chunk_size"]

    def get_default_peft_targets(self):
        return "linear"

    def forward(self, batch):
        action = self.linear(batch["state"]).unsqueeze(1).expand(-1, self.chunk_size, -1)
        if self.enable_rtc:
            action = (
                action
                + batch["prev_chunk_left_over"] * batch["max_guidance_weight"]
                + batch["inference_delay"].float() * 0.001
                + batch["execution_horizon"].float() * 0.002
            )
        return action


@pytest.mark.parametrize("lora,dora", [(False, False), (True, False), (True, True)])
def test_disposable_pi05_export_preserves_predictions_without_lora_snapshots(tmp_path, monkeypatch, lora, dora):
    ov = pytest.importorskip("openvino")
    monkeypatch.setattr("physicalai.policies.pi05.policy.Pi05Model", TinyPi05Model)
    stats = {
        name: {"name": name, "type": kind, "shape": [2], "q01": [-1.0, -1.0], "q99": [1.0, 1.0]}
        for name, kind in (("observation.state", "STATE"), ("action", "ACTION"))
    }
    original = Pi05(
        dataset_stats=stats,
        chunk_size=4,
        n_action_steps=4,
        max_state_dim=2,
        max_action_dim=2,
        lora_enabled=lora,
        lora_use_dora=dora,
        lora_target_modules="linear",
        lora_rank=2,
    ).eval()
    if lora:
        with torch.no_grad():
            original.model.linear.lora_A["default"].weight.fill_(0.1)
            original.model.linear.lora_B["default"].weight.fill_(0.2)
    checkpoint = tmp_path / "model.ckpt"
    data = {
        "pytorch-lightning_version": lightning.__version__,
        "hyper_parameters": dict(original.hparams),
        "state_dict": original.state_dict(),
        "rtc_enabled": False,
        "optimizer_states": [{"state": {0: {"exp_avg": torch.zeros(32, 32)}}}],
    }
    torch.save(data, checkpoint)
    before = checkpoint.read_bytes()
    original.rtc_enabled = True
    observation = {
        "state": torch.tensor([[0.25, -0.5]]),
        "prev_chunk_left_over": torch.ones(1, 4, 2),
        "inference_delay": torch.tensor(1),
        "execution_horizon": torch.tensor(2),
        "max_guidance_weight": torch.tensor(0.5),
    }
    with torch.no_grad():
        expected = original.model(observation).numpy()

    # Keep the real Pi05 export route (including ONNX -> OpenVINO), but do not
    # download a tokenizer or trace the multi-billion-parameter backbone.
    parameters = Pi05.extra_export_args.fget

    def extra_args(policy):
        args = parameters(policy)
        args["openvino"].export_tokenizer = False
        args["openvino"].preprocessors_specs = [
            spec for spec in args["openvino"].preprocessors_specs if spec.type != "ov_tokenizer"
        ]
        args["openvino"].compress_to_fp16 = False
        args["openvino"].inputs = list(observation)
        return args

    monkeypatch.setattr(Pi05, "extra_export_args", property(extra_args))
    monkeypatch.setattr(Pi05, "_get_default_export_input_sample", lambda _self: observation)
    snapshot = Mock(side_effect=AssertionError("Disposable export must not snapshot base weights"))
    monkeypatch.setattr("physicalai.policies.mixins.peft.policy.merged_lora_scope", snapshot)
    model = Model(path=str(tmp_path), name="Tiny Pi05", policy="pi05", project_id=uuid4(), properties={})
    payload = OptimizeModelJobPayload(model_id=model.id, operation="rtc", target_variant="rtc")
    info = VariantInfo(id="original")
    destination = tmp_path / "export"
    ModelOptimizationService._export_checkpoint(model, payload, info, destination)
    manifest = read_package(destination, "openvino")
    declared = {item["init_args"]["name"] for item in manifest["model"]["input_features"]}
    assert {"prev_chunk_left_over", "inference_delay", "execution_horizon", "max_guidance_weight"} <= declared
    compiled = ov.Core().compile_model(destination / "pi05.xml", "CPU")
    actual = compiled({key: value.numpy() for key, value in observation.items()})[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
    snapshot.assert_not_called()
    assert checkpoint.read_bytes() == before
    assert info.rtc_enabled
    assert not list(destination.glob(".export-*"))
    # The original live policy is not the disposable export instance.
    if lora:
        assert hasattr(original.model.linear, "lora_A")
    with torch.no_grad():
        np.testing.assert_array_equal(original.model(observation).numpy(), expected)
