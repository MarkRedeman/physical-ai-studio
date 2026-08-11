# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for LeRobot policy export support."""

from __future__ import annotations

from pathlib import Path

import pytest

# Skip all tests if lerobot not installed
pytest.importorskip("lerobot")

import torch  # noqa: E402
from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from physicalai.config.serializable import dataclass_to_dict  # noqa: E402
from physicalai.export.backends import ExportBackend  # noqa: E402
from physicalai.export.mixin_policy import CONFIG_KEY, POLICY_NAME_KEY  # noqa: E402
from physicalai.policies.lerobot.export import (  # noqa: E402
    ExportableLeRobotPolicy,
    _LerobotInferenceWrapper,
    build_inputs_schema,
    build_outputs_schema,
    export_lerobot_policy,
)
from physicalai.policies.lerobot.policy import LeRobotPolicy  # noqa: E402


@pytest.fixture(scope="module")
def act_policy() -> ExportableLeRobotPolicy:
    """Build a tiny exportable ACT policy from a synthetic config (no dataset)."""
    return ExportableLeRobotPolicy(
        "act",
        input_features={
            "observation.images.top": PolicyFeature(FeatureType.VISUAL, (3, 64, 64)),
            "observation.state": PolicyFeature(FeatureType.STATE, (12,)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (12,))},
        device="cpu",
    )


class TestExportableLeRobotPolicy:
    def test_advertises_torch_and_openvino_backends(self, act_policy) -> None:
        backends = act_policy.get_supported_export_backends()
        assert ExportBackend.TORCH in backends
        assert ExportBackend.OPENVINO in backends
        assert ExportBackend.ONNX in backends

    def test_inputs_schema_describes_observations(self, act_policy) -> None:
        schema = build_inputs_schema(act_policy)
        by_name = {f.name: f for f in schema}
        assert by_name["observation.state"].shape == (12,)
        assert by_name["observation.images.top"].shape == (3, 64, 64)

    def test_outputs_schema_includes_action_chunk(self, act_policy) -> None:
        schema = build_outputs_schema(act_policy)
        assert len(schema) == 1
        action = schema[0]
        assert action.name == "action"
        assert action.shape == (act_policy._config.n_action_steps, 12)

    def test_exportable_requires_policy_name(self) -> None:
        with pytest.raises(ValueError):
            ExportableLeRobotPolicy(config=None)

    def test_rejects_bare_lerobot_policy(self, tmp_path: Path) -> None:
        bare = LeRobotPolicy(
            "act",
            input_features={
                "observation.images.top": PolicyFeature(FeatureType.VISUAL, (3, 64, 64)),
                "observation.state": PolicyFeature(FeatureType.STATE, (12,)),
            },
            output_features={"action": PolicyFeature(FeatureType.ACTION, (12,))},
            device="cpu",
        )
        with pytest.raises(TypeError):
            export_lerobot_policy(bare, str(tmp_path))


class TestLoadFromCheckpointDeviceAlignment:
    def test_map_location_cpu_aligns_config_device(self, act_policy, tmp_path: Path) -> None:
        """Loading a GPU-trained checkpoint with ``map_location="cpu"`` must yield an all-CPU policy.

        LeRobot's config persists the training device, and its preprocessor
        (``DeviceProcessorStep``) plus some policy forward passes target
        ``config.device``. Loading to CPU without aligning the config device
        leaves the trace sample on the accelerator while the weights sit on CPU,
        which makes ``torch.export`` fail with "Unhandled FakeTensor Device
        Propagation".
        """
        accelerator = "cuda" if torch.cuda.is_available() else ("xpu" if torch.xpu.is_available() else None)
        if accelerator is None:
            pytest.skip("No accelerator available to simulate a GPU-trained checkpoint")

        config_dict = dataclass_to_dict(act_policy._config)
        config_dict["device"] = accelerator

        ckpt_path = tmp_path / "model.ckpt"
        checkpoint = {
            "state_dict": act_policy.state_dict(),
            CONFIG_KEY: config_dict,
            POLICY_NAME_KEY: "act",
        }
        # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
        torch.save(checkpoint, ckpt_path)  # nosec B614

        loaded = ExportableLeRobotPolicy.load_from_checkpoint(ckpt_path, map_location="cpu")

        assert loaded._config.device == "cpu"
        sample = loaded._get_default_export_input_sample()
        assert all(v.device.type == "cpu" for v in sample.values())

        wrapper = _LerobotInferenceWrapper(loaded).eval()
        torch.export.export(wrapper, args=(), kwargs={"observation": sample})


class TestExportLerobotPolicy:
    def test_torch_export_writes_checkpoint_and_manifest(self, act_policy, tmp_path: Path) -> None:
        export_lerobot_policy(act_policy, str(tmp_path), backends=["torch"])
        torch_dir = tmp_path / "torch"
        assert (torch_dir / "lerobotpolicy.pt").is_file()
        assert (torch_dir / "manifest.json").is_file()
