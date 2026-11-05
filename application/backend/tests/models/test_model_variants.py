# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import numpy as np
import pytest

from schemas.job import OptimizeModelJobPayload
from schemas.model import Model
from schemas.model_variant import VariantInfo, export_directory, package_files, read_package, target_variant
from services.model_optimization_service import ModelOptimizationService
from workers.model_worker import Inference, InferRequest


def package(directory, info=None):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "model.xml").write_text("graph")
    (directory / "model.bin").write_bytes(b"weights")
    manifest = {
        "format": "policy_package",
        "policy": {"name": "smolvla"},
        "model": {"artifacts": {"openvino": "model.xml"}},
    }
    if info is not None:
        manifest["studio"] = {"variant": info.model_dump(mode="json")}
    (directory / "manifest.json").write_text(json.dumps(manifest))


@pytest.mark.parametrize(
    "source,operation,target",
    [
        ("original", "int8_sym", "int8"),
        ("original", "rtc", "rtc"),
        ("original", "snapflow", "snapflow"),
        ("rtc", "int8_sym", "rtc-int8"),
        ("int8", "rtc", "rtc-int8"),
        ("snapflow", "rtc", "snapflow-rtc"),
        ("snapflow", "int8_sym", "snapflow-int8"),
        ("rtc-int8", "snapflow", "snapflow-rtc-int8"),
    ],
)
def test_canonical_variant_identity(source, operation, target):
    info = VariantInfo(
        id=source,
        checkpoint_kind="snapflow" if "snapflow" in source else "ordinary",
        rtc_enabled="rtc" in source,
        weight_compression="int8_sym" if "int8" in source else None,
    )
    assert target_variant(info, operation) == target


@pytest.mark.parametrize(
    "backend,variant", [("../openvino", "original"), ("openvino", "../int8"), ("openvino", "/int8")]
)
def test_export_resolution_rejects_path_injection(tmp_path, backend, variant):
    with pytest.raises(ValueError):
        export_directory(tmp_path, backend, variant)


def test_export_and_dependency_symlinks_are_rejected(tmp_path):
    root = tmp_path / "model"
    (root / "exports").mkdir(parents=True)
    (root / "exports/openvino").symlink_to(tmp_path / "outside", target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic"):
        export_directory(root, "openvino")
    directory = tmp_path / "package"
    package(directory)
    (directory / "model.bin").unlink()
    (directory / "model.bin").symlink_to(tmp_path / "outside.bin")
    with pytest.raises(ValueError, match="symbolic"):
        read_package(directory, "openvino")


def test_variant_only_backend_is_discovered_without_counting_it_as_original(tmp_path):
    model = Model(path=str(tmp_path), name="Policy", policy="smolvla", project_id=uuid4(), properties={})
    package(tmp_path / "exports/openvino/variants/int8", VariantInfo(id="int8", weight_compression="int8_sym"))
    assert model.available_backends == ["openvino"]
    assert set(model.export_packages()) == {("openvino", "int8")}
    assert package_files(tmp_path / "exports/openvino") == []


@pytest.mark.parametrize(
    "operation,source,kind,rtc,compressed",
    [
        ("rtc", "original", "ordinary", True, False),
        ("rtc", "int8", "ordinary", True, True),
        ("rtc", "snapflow", "snapflow", True, False),
        ("snapflow", "rtc-int8", "snapflow", True, True),
    ],
)
def test_graph_variants_reload_the_matching_checkpoint_before_compression(
    tmp_path, monkeypatch, operation, source, kind, rtc, compressed
):
    model = Model(path=str(tmp_path), name="Policy", policy="smolvla", project_id=uuid4(), properties={})
    info = VariantInfo(
        id=source,
        checkpoint_kind="snapflow" if "snapflow" in source else "ordinary",
        rtc_enabled="rtc" in source,
        weight_compression="int8_sym" if "int8" in source else None,
    )
    package(export_directory(tmp_path, "openvino", source), info)
    (tmp_path / "model.ckpt").write_bytes(b"ordinary")
    (tmp_path / "snapflow-model.ckpt").write_bytes(b"distilled")
    from physicalai.export import ExportablePolicyMixin
    from physicalai.policies.mixins import RTCPolicyMixin

    class FakePolicy(RTCPolicyMixin, ExportablePolicyMixin):
        pass

    policy = FakePolicy()
    policy.config = SimpleNamespace(snapflow_enabled=kind == "snapflow")
    policy.eval = MagicMock()

    def export(directory, backend):
        assert backend == "openvino"
        assert policy.rtc_enabled is rtc
        package(directory)

    policy.export = MagicMock(side_effect=export)
    loader = MagicMock(return_value=policy)
    monkeypatch.setattr(
        "physicalai.policies.get_physicalai_policy_class", lambda name: SimpleNamespace(load_from_checkpoint=loader)
    )
    compress = MagicMock()
    monkeypatch.setattr("physicalai.export.hooks.compress_weights_openvino_int8_sym", compress)
    payload = OptimizeModelJobPayload(
        model_id=model.id, source_variant=source, operation=operation, target_variant=target_variant(info, operation)
    )
    destination = tmp_path / "staging"
    result = ModelOptimizationService.build_variant(model, payload, destination, uuid4())
    assert Path(loader.call_args.args[0]).name == ("snapflow-model.ckpt" if kind == "snapflow" else "model.ckpt")
    assert result.rtc_enabled is rtc
    assert result.checkpoint_kind == kind
    assert bool(result.weight_compression) is compressed
    assert compress.call_count == int(compressed)


def test_rtc_controls_reach_worker_with_correct_dtypes():
    model = MagicMock()
    model.predict_action_chunk.return_value = np.zeros((2, 3), dtype=np.float32)
    inference = Inference(model=model, store=MagicMock(), images={})
    request = InferRequest(
        state=[1, 2, 3],
        prev_chunk_left_over=[[[1, 2, 3], [4, 5, 6]]],
        inference_delay=2,
        execution_horizon=3,
        max_guidance_weight=4.5,
    )
    inference.infer(request)
    observation = model.predict_action_chunk.call_args.args[0]
    assert observation["prev_chunk_left_over"].shape == (1, 2, 3)
    assert observation["prev_chunk_left_over"].dtype == np.float32
    assert observation["inference_delay"].dtype == np.int64
    assert observation["execution_horizon"].dtype == np.int64
    assert observation["max_guidance_weight"].dtype == np.float32


def test_real_openvino_weight_compression_preserves_loadable_artifact(tmp_path):
    ov = pytest.importorskip("openvino")
    pytest.importorskip("nncf")
    from physicalai.export.hooks import compress_weights_openvino_int8_sym

    rng = np.random.default_rng(42)
    parameter = ov.opset13.parameter([1, 64], np.float32, name="state")
    weights = ov.opset13.constant(rng.standard_normal((64, 64)).astype(np.float32))
    graph = ov.Model([ov.opset13.matmul(parameter, weights, False, False)], [parameter])
    path = tmp_path / "model.xml"
    ov.save_model(graph, path)
    del graph
    compress_weights_openvino_int8_sym(str(path))
    compiled = ov.Core().compile_model(path, "CPU")
    assert compiled([np.ones((1, 64), dtype=np.float32)])[0].shape == (1, 64)


def test_portable_rtc_launcher_shares_latency_tracker(monkeypatch):
    from services import rtc_runtime_launcher as launcher

    predictor = MagicMock()
    model = MagicMock(return_value=predictor)
    execution = MagicMock()
    source = MagicMock()
    runtime = MagicMock()
    monkeypatch.setattr(launcher, "InferenceModel", model)
    monkeypatch.setattr(launcher, "RTCExecution", execution)
    monkeypatch.setattr(launcher, "PolicySource", source)
    monkeypatch.setattr(launcher, "RobotRuntime", runtime)
    launcher.build_runtime(
        {
            "init_args": {
                "fps": 30,
                "robot": None,
                "action_source": {
                    "init_args": {
                        "model": {"init_args": {"export_dir": "exports/openvino", "backend": "openvino"}},
                        "execution": {"init_args": {"chunk_size": 50, "execution_horizon": 15}},
                    }
                },
            }
        }
    )
    tracker = model.call_args.kwargs["callbacks"][0]
    assert execution.call_args.kwargs["latency_tracker"] is tracker
    assert execution.call_args.kwargs["chunk_size"] == 50
    assert source.call_args.args[0] is predictor
