# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the LeRobot training engine.

The engine trains with lerobot's own stack when the user picks LeRobot; what is
asserted here is the Studio side of that contract: how a spec maps onto a
``TrainPipelineConfig``, which devices are allowed, what a run publishes, and
how cancellation is honored. The lerobot training loop itself is mocked out;
the loop's shape is verified end to end by the manual smoke script.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from services.training_backends._log_format import render_progress_log
from training import TrainingJobSpec
from training.job import CHECKPOINT_NAME, EXPORTS_DIRNAME, run_training_job
from training.lerobot import (
    _build_config,
    _latest_checkpoint,
    _lerobot_rename_map,
    _MetricsWriter,
    _resolve_device,
    _resolve_resume_checkpoint,
    _total_frames,
)

if TYPE_CHECKING:
    from pathlib import Path


def _snapshot(tmp_path: Path, total_frames: int = 600) -> Path:
    """Build a minimal LeRobot snapshot (meta/info.json only)."""
    snapshot = tmp_path / "snapshot"
    (snapshot / "meta").mkdir(parents=True)
    (snapshot / "meta" / "info.json").write_text(json.dumps({"total_frames": total_frames}))
    return snapshot


class TestTrainingJobSpecEngine:
    def test_training_engine_defaults_to_physicalai(self) -> None:
        spec = TrainingJobSpec(policy="act")
        assert spec.training_engine == "physicalai"

    def test_training_engine_round_trips_through_json(self) -> None:
        spec = TrainingJobSpec(policy="diffusion", training_engine="lerobot")
        assert TrainingJobSpec.model_validate_json(spec.model_dump_json()) == spec


class TestDispatch:
    def test_lerobot_engine_routes_to_the_lerobot_runner(self, tmp_path: Path) -> None:
        """The lerobot engine must not construct the Lightning stack at all."""
        with (
            patch("training.lerobot.run_lerobot_training_job") as lerobot_runner,
            patch("physicalai.data.LeRobotDataModule") as datamodule,
            patch("physicalai.train.trainer.Trainer") as trainer,
        ):
            run_training_job(
                TrainingJobSpec(policy="act", training_engine="lerobot"),
                dataset_root=tmp_path / "snapshot",
                output_dir=tmp_path / "model",
                cache_dir=tmp_path / "cache" / "job",
                report=MagicMock(),
                should_stop=lambda: False,
            )

        lerobot_runner.assert_called_once()
        assert not datamodule.called
        assert not trainer.called

    def test_physicalai_engine_does_not_touch_lerobot(self, tmp_path: Path) -> None:
        with (
            patch("training.lerobot.run_lerobot_training_job") as lerobot_runner,
            patch("physicalai.data.LeRobotDataModule"),
            patch("training.job.build_policy"),
            patch("physicalai.train.trainer.Trainer"),
        ):
            run_training_job(
                TrainingJobSpec(policy="act"),
                dataset_root=tmp_path / "snapshot",
                output_dir=tmp_path / "model",
                cache_dir=tmp_path / "cache" / "job",
                report=MagicMock(),
                should_stop=lambda: False,
            )

        lerobot_runner.assert_not_called()


class TestDeviceResolution:
    def test_unknown_device_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unsupported device"):
            _resolve_device(TrainingJobSpec(policy="act", training_engine="lerobot", device_type="npu"))

    def test_cpu_when_no_accelerator_available(self) -> None:
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.xpu.is_available", return_value=False),
        ):
            assert _resolve_device(TrainingJobSpec(policy="act", training_engine="lerobot")) == _cpu()

    def test_cuda_when_available(self) -> None:
        with patch("torch.cuda.is_available", return_value=True):
            assert _resolve_device(TrainingJobSpec(policy="act", training_engine="lerobot")) == _cuda()

    def test_xpu_when_cuda_unavailable(self) -> None:
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.xpu.is_available", return_value=True),
        ):
            assert _resolve_device(TrainingJobSpec(policy="act", training_engine="lerobot")) == _xpu()

    def test_explicit_cuda_device_index(self) -> None:
        with patch("torch.cuda.is_available", return_value=True):
            device = _resolve_device(
                TrainingJobSpec(policy="act", training_engine="lerobot", device_type="cuda", device_index=1)
            )
        assert str(device) == "cuda:1"

    def test_cuda_requested_but_unavailable(self) -> None:
        with patch("torch.cuda.is_available", return_value=False), pytest.raises(ValueError, match="not available"):
            _resolve_device(TrainingJobSpec(policy="act", training_engine="lerobot", device_type="cuda"))

    def test_explicit_xpu_device_index(self) -> None:
        with patch("torch.xpu.is_available", return_value=True):
            device = _resolve_device(
                TrainingJobSpec(policy="act", training_engine="lerobot", device_type="xpu", device_index=1)
            )
        assert str(device) == "xpu:1"

    def test_xpu_requested_but_unavailable(self) -> None:
        with patch("torch.xpu.is_available", return_value=False), pytest.raises(ValueError, match="not available"):
            _resolve_device(TrainingJobSpec(policy="act", training_engine="lerobot", device_type="xpu"))


def _cpu() -> object:
    import torch

    return torch.device("cpu")


def _cuda() -> object:
    import torch

    return torch.device("cuda")


def _xpu() -> object:
    import torch

    return torch.device("xpu")


class TestTotalFrames:
    def test_reads_total_frames_from_info_json(self, tmp_path: Path) -> None:
        assert _total_frames(_snapshot(tmp_path, total_frames=42)) == 42


class TestRenameMap:
    def test_converts_short_camera_names_to_full_lerobot_keys(self) -> None:
        assert _lerobot_rename_map({"camera1": "front", "camera2": "left"}) == {
            "observation.images.front": "observation.images.camera1",
            "observation.images.left": "observation.images.camera2",
        }

    def test_omits_empty_camera_slots(self) -> None:
        assert _lerobot_rename_map({"camera1": "front", "camera2": None}) == {
            "observation.images.front": "observation.images.camera1",
        }

    def test_empty_map_stays_empty(self) -> None:
        assert _lerobot_rename_map({}) == {}


class TestConfigDerivation:
    def test_steps_are_derived_from_epochs_and_dataset(self, tmp_path: Path) -> None:
        snapshot = _snapshot(tmp_path, total_frames=600)
        cfg = _build_config(
            TrainingJobSpec(policy="act", training_engine="lerobot", max_epochs=3),
            dataset_root=snapshot,
            device=_cpu(),
            cache_dir=tmp_path / "cache",
            resume_checkpoint=None,
        )

        assert cfg.steps == 3 * ((600 + 7) // 8)
        assert cfg.batch_size == 8

    @pytest.mark.parametrize(
        ("policy", "lr", "weight_decay", "grad_clip"),
        [
            ("act", 1e-4, 1e-4, 10.0),
            ("diffusion", 1e-4, 1e-2, 10.0),
            ("smolvla", 2e-5, 1e-2, 1.0),
            ("pi05", 1e-4, 1e-2, 1.0),
        ],
    )
    def test_policy_defaults(
        self, tmp_path: Path, policy: str, lr: float, weight_decay: float, grad_clip: float
    ) -> None:
        cfg = _build_config(
            TrainingJobSpec(policy=policy, training_engine="lerobot"),
            dataset_root=_snapshot(tmp_path),
            device=_cpu(),
            cache_dir=tmp_path / "cache",
            resume_checkpoint=None,
        )

        assert cfg.batch_size == 8
        assert cfg.optimizer.lr == lr
        assert cfg.optimizer.weight_decay == weight_decay
        assert cfg.optimizer.grad_clip_norm == grad_clip

    def test_scheduler_decay_matches_the_step_budget(self, tmp_path: Path) -> None:
        cfg = _build_config(
            TrainingJobSpec(policy="act", training_engine="lerobot"),
            dataset_root=_snapshot(tmp_path, total_frames=600),
            device=_cpu(),
            cache_dir=tmp_path / "cache",
            resume_checkpoint=None,
        )

        assert cfg.scheduler.num_decay_steps == cfg.steps
        assert cfg.scheduler.peak_lr == cfg.optimizer.lr
        assert cfg.scheduler.num_warmup_steps > 0

    def test_val_split_maps_to_eval_split(self, tmp_path: Path) -> None:
        cfg = _build_config(
            TrainingJobSpec(policy="act", training_engine="lerobot", val_split=0.25),
            dataset_root=_snapshot(tmp_path),
            device=_cpu(),
            cache_dir=tmp_path / "cache",
            resume_checkpoint=None,
        )

        assert cfg.dataset.eval_split == 0.25
        assert cfg.eval_steps > 0

    def test_zero_val_split_disables_eval(self, tmp_path: Path) -> None:
        cfg = _build_config(
            TrainingJobSpec(policy="act", training_engine="lerobot", val_split=0.0),
            dataset_root=_snapshot(tmp_path),
            device=_cpu(),
            cache_dir=tmp_path / "cache",
            resume_checkpoint=None,
        )

        assert cfg.eval_steps == 0

    def test_policy_device_comes_from_the_spec(self, tmp_path: Path) -> None:
        cfg = _build_config(
            TrainingJobSpec(policy="act", training_engine="lerobot"),
            dataset_root=_snapshot(tmp_path),
            device=_cpu(),
            cache_dir=tmp_path / "cache",
            resume_checkpoint=None,
        )

        assert cfg.policy.device == "cpu"

    def test_resume_loads_the_checkpoints_config(self, tmp_path: Path) -> None:
        # A published lerobot model: the raw checkpoint under lerobot/.
        model = tmp_path / "model"
        pretrained = model / "lerobot" / "pretrained_model"
        pretrained.mkdir(parents=True)
        snapshot = _snapshot(tmp_path, total_frames=600)
        original = _build_config(
            TrainingJobSpec(policy="act", training_engine="lerobot"),
            dataset_root=snapshot,
            device=_cpu(),
            cache_dir=tmp_path / "original-cache",
            resume_checkpoint=None,
        )
        original.save_pretrained(pretrained)

        cfg = _build_config(
            TrainingJobSpec(policy="act", training_engine="lerobot"),
            dataset_root=snapshot,
            device=_cpu(),
            cache_dir=tmp_path / "cache",
            resume_checkpoint=model / "lerobot",
        )

        assert cfg.resume is True
        assert cfg.checkpoint_path == model / "lerobot"
        assert cfg.dataset.root == str(snapshot)
        assert cfg.output_dir == tmp_path / "cache"
        assert cfg.wandb.enable is False
        assert cfg.policy.device == "cpu"
        # Batch size / steps / LR survive from the checkpoint's own config.
        assert cfg.steps == original.steps
        assert cfg.batch_size == original.batch_size


class TestResumeCheckpoint:
    def test_resolves_lerobot_subdir_of_the_model(self, tmp_path: Path) -> None:
        model = tmp_path / "model"
        (model / "lerobot" / "pretrained_model").mkdir(parents=True)

        assert _resolve_resume_checkpoint(model) == model / "lerobot"

    def test_returns_none_without_a_lerobot_checkpoint(self, tmp_path: Path) -> None:
        model = tmp_path / "model"
        model.mkdir()

        assert _resolve_resume_checkpoint(model) is None
        assert _resolve_resume_checkpoint(None) is None


class TestLatestCheckpoint:
    def test_picks_the_highest_zero_padded_step_dir(self, tmp_path: Path) -> None:
        checkpoints = tmp_path / "checkpoints"
        checkpoints.mkdir()
        (checkpoints / "000100").mkdir()
        (checkpoints / "000010").mkdir()

        assert _latest_checkpoint(checkpoints) == checkpoints / "000100"

    def test_accepts_older_step_prefix_layout(self, tmp_path: Path) -> None:
        checkpoints = tmp_path / "checkpoints"
        checkpoints.mkdir()
        (checkpoints / "step_0100").mkdir()

        assert _latest_checkpoint(checkpoints) == checkpoints / "step_0100"

    def test_none_when_no_checkpoints(self, tmp_path: Path) -> None:
        assert _latest_checkpoint(tmp_path / "missing") is None


class TestMetricsWriter:
    def test_writes_lightning_style_csv(self, tmp_path: Path) -> None:
        writer = _MetricsWriter(
            report=MagicMock(),
            cache_dir=tmp_path / "job",
            max_steps=100,
            steps_per_epoch=10,
        )

        writer.on_log_step(10, 0.5)
        writer.on_eval(20, 0.25, elapsed_s=1.5)
        writer.close()

        csv_path = tmp_path / "job" / "version_0" / "metrics.csv"
        lines = csv_path.read_text().splitlines()
        assert lines[0] == "epoch,step,train/loss_step,val/loss"
        assert "1,10,0.500000," in lines
        assert "2,20,,0.250000" in lines

    def test_reports_progress_in_the_shared_schema(self, tmp_path: Path) -> None:
        report = MagicMock()
        writer = _MetricsWriter(report=report, cache_dir=tmp_path / "job", max_steps=200, steps_per_epoch=50)

        writer.on_log_step(50, 0.5)
        _, _, extra_info = report.call_args.args
        assert render_progress_log(extra_info) == ("Training progress: step=50/200 (25%), train/loss_step=0.5")

        writer.on_eval(100, 0.25, elapsed_s=2.0)
        _, _, extra_info = report.call_args.args
        assert render_progress_log(extra_info) == ("Validation finished at step=100, val/loss=0.25, elapsed=2.0s")

    def test_running_progress_caps_at_99(self, tmp_path: Path) -> None:
        report = MagicMock()
        writer = _MetricsWriter(report=report, cache_dir=tmp_path / "job", max_steps=10, steps_per_epoch=2)

        writer.on_log_step(10, 0.1)

        assert report.call_args.args[0] == 99


class TestPublish:
    def test_publishes_checkpoint_as_lerobot_model_layout(self, tmp_path: Path) -> None:
        from training.lerobot import _publish

        snapshot = _snapshot(tmp_path)
        cache = tmp_path / "cache" / "job"
        checkpoints = cache / "checkpoints"
        (checkpoints / "000100" / "pretrained_model").mkdir(parents=True)
        (checkpoints / "000100" / "training_state").mkdir()
        (checkpoints / "000100" / "pretrained_model" / "config.json").write_text('{"type": "act"}')
        (checkpoints / "last").symlink_to("000100")

        cfg = _build_config(
            TrainingJobSpec(policy="act", training_engine="lerobot"),
            dataset_root=snapshot,
            device=_cpu(),
            cache_dir=cache,
            resume_checkpoint=None,
        )
        output_dir = tmp_path / "model"

        with (
            patch("physicalai.policies.lerobot.utils.checkpoint_converter.lerobot_to_lightning") as convert,
            patch("training.lerobot._export_backends") as export_backends,
        ):
            _publish(cfg, cache_dir=cache, output_dir=output_dir, report=MagicMock())

        # The raw checkpoint is kept under lerobot/ for native resume.
        assert (output_dir / "lerobot" / "pretrained_model" / "config.json").is_file()
        assert (output_dir / "lerobot" / "training_state").is_dir()
        # The checkpoints/ tree is collapsed into the single raw checkpoint.
        assert not (output_dir / "checkpoints").exists()
        # The checkpoint converts to the shared Lightning artifact (in cache,
        # before the cache moves to its final location).
        convert.assert_called_once()
        assert convert.call_args.args[0] == cache / "lerobot" / "pretrained_model"
        assert convert.call_args.args[1] == cache / CHECKPOINT_NAME
        # Torch export runs from the published model dir.
        export_backends.assert_called_once_with(output_dir, EXPORTS_DIRNAME, None)
        # The cache moved; nothing is left behind.
        assert not cache.exists()

    def test_no_checkpoint_raises(self, tmp_path: Path) -> None:
        from training.lerobot import _publish

        cache = tmp_path / "cache" / "job"
        cfg = MagicMock()

        with pytest.raises(RuntimeError, match="No checkpoint"):
            _publish(cfg, cache_dir=cache, output_dir=tmp_path / "model", report=MagicMock())


class TestJobLifecycle:
    def test_cancel_after_training_skips_publish(self, tmp_path: Path) -> None:
        from training.lerobot import run_lerobot_training_job

        with (
            patch("training.lerobot._build_config", return_value=MagicMock()),
            patch("training.lerobot._train") as train,
            patch("training.lerobot._publish") as publish,
            patch("torch.cuda.is_available", return_value=False),
        ):
            run_lerobot_training_job(
                TrainingJobSpec(policy="act", training_engine="lerobot"),
                dataset_root=tmp_path / "snapshot",
                output_dir=tmp_path / "model",
                cache_dir=tmp_path / "cache" / "job",
                report=MagicMock(),
                should_stop=lambda: True,
            )

        train.assert_called_once()
        publish.assert_not_called()

    def test_unknown_device_fails_before_training(self, tmp_path: Path) -> None:
        from training.lerobot import run_lerobot_training_job

        with (
            patch("training.lerobot._train") as train,
            patch("training.lerobot._publish") as publish,
            pytest.raises(ValueError, match="Unsupported device"),
        ):
            run_lerobot_training_job(
                TrainingJobSpec(policy="act", training_engine="lerobot", device_type="npu"),
                dataset_root=tmp_path / "snapshot",
                output_dir=tmp_path / "model",
                cache_dir=tmp_path / "cache" / "job",
                report=MagicMock(),
                should_stop=lambda: False,
            )

        train.assert_not_called()
        publish.assert_not_called()
