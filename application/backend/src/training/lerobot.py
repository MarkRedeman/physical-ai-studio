# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""LeRobot training engine.

Trains a policy with LeRobot's own stack (``lerobot.scripts.lerobot_train``
semantics) when the user picks LeRobot as the training engine, instead of the
physicalai-train Lightning stack. The user only chooses the policy; batch size,
step budget, optimizer, and LR schedule are derived from LeRobot's best
practices (see the upstream ``AGENT_GUIDE.md``, section "Train a policy").

This is a lean vendored loop, not a subprocess around the LeRobot CLI: it must
stay cooperative with the job's ``report``/``should_stop`` callbacks and emit
the same progress telemetry contract as the physicalai path. The loop mirrors
``lerobot.scripts.lerobot_train.train`` for a single process: same dataset
factory, sampler, optimizer/scheduler construction, checkpoint layout, and
step-by-step update order, minus accelerate and its distributed sharding.

After training, the final checkpoint is published like a physicalai model:

- ``model.ckpt`` — the checkpoint converted to Lightning format
  (``lerobot_to_lightning``), the shared load contract with the Runtime.
- ``lerobot/`` — the raw LeRobot checkpoint (``pretrained_model/`` +
  ``training_state/``), so later runs can resume natively with LeRobot.
- ``exports/torch/`` — the torch export, via the LeRobotPolicy wrapper (the
  only export backend it supports).

The engine runs on CUDA, XPU, or CPU, using the job's epoch budget
(``max_epochs``) and batch size. It derives the corresponding step budget
from the dataset size.
"""

from __future__ import annotations

import csv
import logging
import math
import os
import shutil
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from lerobot.common.train_utils import (
    get_step_checkpoint_dir,
    load_training_batch_size,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.configs.default import DatasetConfig, WandBConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets import EpisodeAwareSampler, compute_sampler_state
from lerobot.datasets.factory import make_train_eval_datasets
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.policies.factory import make_policy_config
from lerobot.utils.collate import lerobot_collate_fn
from lerobot.utils.constants import PRETRAINED_MODEL_DIR
from lerobot.utils.utils import cycle

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from lerobot.policies.pretrained import PreTrainedPolicy
    from physicalai.train.callbacks import ReportFn, StopFn

    from training.job import ExportBackends, TrainingJobSpec

logger = logging.getLogger(__name__)

# Metric-log cadence contract, kept in sync with the physicalai engine's
# ``ProgressReportingCallback``: log every 10 steps for the first 10k steps so
# job metrics graphs fill in quickly, then fall back to the adaptive cadence.
_METRIC_LOG_EARLY_CADENCE = 10
_METRIC_LOG_EARLY_STEPS = 10_000

# Upper bound for the auto batch-size search (power-of-two doubling).
_MAX_AUTO_SCALE_BATCH = 512

_DATASET_REPO_ID = "snapshot"
"""Placeholder repo id: datasets are always loaded from a local root here."""


def run_lerobot_training_job(
    spec: TrainingJobSpec,
    *,
    dataset_root: Path | str,
    output_dir: Path | str,
    cache_dir: Path | str,
    report: ReportFn,
    should_stop: StopFn,
    resume_from: Path | str | None = None,
) -> None:
    """Train one policy end to end with LeRobot's stack.

    On success ``output_dir`` holds ``model.ckpt``, the raw ``lerobot/``
    checkpoint, ``exports/torch/``, and the ``version_0/metrics.csv`` log.
    Cancellation is cooperative: ``should_stop`` is polled every step, and a
    canceled run returns without writing any artifact (the final checkpoint is
    only written at the end of a completed run).
    """
    device = _resolve_device(spec)
    output_dir, cache_dir = Path(output_dir), Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if spec.rename_map:
        logger.warning(
            "Camera rename_map is not supported by the LeRobot engine; training on the dataset's cameras as-is"
        )

    resume_checkpoint = _resolve_resume_checkpoint(resume_from)
    cfg = _build_config(
        spec, dataset_root=Path(dataset_root), device=device, cache_dir=cache_dir, resume_checkpoint=resume_checkpoint
    )

    _train(
        cfg,
        device=device,
        report=report,
        should_stop=should_stop,
        max_epochs=spec.max_epochs,
        auto_scale_batch_size=spec.auto_scale_batch_size,
    )
    if should_stop():
        logger.info("LeRobot training canceled; skipping publish")
        return

    _publish(cfg, cache_dir=cache_dir, output_dir=output_dir, report=report, export_backends=spec.export_backends)


def _resolve_device(spec: TrainingJobSpec) -> torch.device:
    """Return the device to train on.

    LeRobot runs on CUDA (preferred), XPU, or CPU. ``device_type`` is None when
    the UI lets the machine decide (prefer CUDA, then XPU, then CPU).
    """
    requested = spec.device_type
    if requested is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.xpu.is_available():
            return torch.device("xpu")
        return torch.device("cpu")
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested for LeRobot training but is not available.")
        return torch.device(f"cuda:{spec.device_index or 0}")
    if requested == "xpu":
        if not torch.xpu.is_available():
            raise ValueError("XPU was requested for LeRobot training but is not available.")
        return torch.device(f"xpu:{spec.device_index or 0}")
    msg = f"Unsupported device for LeRobot training: {requested!r}"
    raise ValueError(msg)


def _resolve_num_workers(spec: TrainingJobSpec) -> int:
    """Resolve the dataloader worker count, capping ``auto`` like the UI."""
    if spec.num_workers == "auto":
        return min(8, os.cpu_count() or 1)
    return spec.num_workers


def _plan_extra_info(
    cfg: TrainPipelineConfig,
    train_dataset: Any,
    eval_dataset: Any,
    steps_per_epoch: int,
    max_epochs: int,
) -> dict[str, object]:
    """Build the ``train_event: "plan"`` telemetry for the LeRobot engine.

    Mirrors the ``ProgressReportingCallback`` plan payload so local and remote
    backends render the same ``Training plan: ...`` job-log line regardless of
    which engine trained the policy. Emitted once before the training loop.
    """
    val_frames = getattr(eval_dataset, "num_frames", None) if eval_dataset is not None else None
    return {
        "train_event": "plan",
        "batch_size": cfg.batch_size,
        "steps_per_epoch": steps_per_epoch,
        "train_frames": getattr(train_dataset, "num_frames", None),
        "val_frames": val_frames,
        "epochs": max_epochs,
        "total_steps": cfg.steps,
        "num_workers": cfg.num_workers,
    }


def _tensor_stats(tensor: torch.Tensor) -> dict[str, list[float]]:
    """Per-channel ``mean/std/min/max`` of a ``(B, C, ...)`` tensor.

    Computed on the CPU with gradients detached so the diagnostic is cheap and
    does not perturb the training graph.
    """
    t = tensor.detach().float().cpu()
    flat = t.reshape(t.shape[0], t.shape[1], -1)
    return {
        "mean": flat.mean(dim=(0, 2)).round(decimals=4).tolist(),
        "std": flat.std(dim=(0, 2)).round(decimals=4).tolist(),
        "min": flat.min(dim=2).values.min(dim=0).values.round(decimals=4).tolist(),
        "max": flat.max(dim=2).values.max(dim=0).values.round(decimals=4).tolist(),
    }


def _input_sanity_flags(
    raw_stats: dict[str, list[float]] | None,
    norm_stats: dict[str, list[float]] | None,
) -> list[str]:
    """Return anomaly flags for a camera's raw and normalized image stats.

    Flags are informational: they surface NaN/Inf decoding, all-constant
    frames, or a zero-variance normalized channel without aborting training.
    """
    flags: list[str] = []
    for label, stats in (("raw", raw_stats), ("norm", norm_stats)):
        if stats is None:
            continue
        values = stats["mean"] + stats["std"] + stats["min"] + stats["max"]
        if any(not math.isfinite(v) for v in values):
            flags.append(f"{label}_non_finite")
    if norm_stats is not None:
        if max(norm_stats["std"]) <= 1e-6:
            flags.append("norm_degenerate_std")
        if norm_stats["min"] == norm_stats["max"]:
            flags.append("norm_degenerate_constant")
    return flags


def _report_input_sanity(
    report: ReportFn,
    raw_batch: Mapping[str, Any],
    batch: Mapping[str, Any],
    camera_keys: list[str],
) -> None:
    """Emit per-camera image statistics for the first training batch.

    The raw statistics capture the collated frames (before preprocessing), the
    normalized statistics capture the preprocessor output, and anomaly flags
    call out NaN/Inf or degenerate frames. Consumers render this via
    ``render_progress_log`` into the job log.
    """
    cameras: dict[str, dict[str, Any]] = {}
    for cam_key in camera_keys:
        if cam_key not in batch:
            continue
        entry: dict[str, Any] = {}
        raw = raw_batch.get(cam_key)
        if isinstance(raw, torch.Tensor):
            entry["raw"] = _tensor_stats(raw)
            entry["raw_dtype"] = str(raw.dtype)
        norm = batch[cam_key]
        if isinstance(norm, torch.Tensor):
            entry["norm"] = _tensor_stats(norm)
        entry["flags"] = _input_sanity_flags(entry.get("raw"), entry.get("norm"))
        cameras[str(cam_key)] = entry
    report(0, None, {"train_event": "input_sanity", "cameras": cameras})


def _resolve_optimizer_and_scheduler(cfg: TrainPipelineConfig) -> None:
    """Populate the optimizer/scheduler from the policy's lerobot presets.

    Mirrors ``TrainPipelineConfig.validate``, which the lerobot CLI calls but the
    studio's direct-construction path does not: with ``use_policy_training_preset``
    the policy config supplies the optimizer and (optional) scheduler, so ACT
    trains at its documented constant ``optimizer_lr`` with no LR schedule.
    """
    if cfg.policy is None or not cfg.use_policy_training_preset:
        return
    if cfg.optimizer is None:
        cfg.optimizer = cfg.trainable_config.get_optimizer_preset()
    if cfg.scheduler is None:
        cfg.scheduler = cfg.trainable_config.get_scheduler_preset()


def _training_budget(*, total_steps: int, steps_per_epoch: int, eval_split: float) -> tuple[int, int, int]:
    """Return the physicalai-synced cadence for a run.

    Mirrors the physicalai engine's ``ProgressReportingCallback``: an adaptive
    log cadence (~1000 entries, capped at every 100 steps, with the early
    every-10-steps phase applied in the training loop), an eval-loss pass once
    per epoch, and a single final checkpoint. ``steps_per_epoch`` derives from
    the train split only.

    Returns:
        Tuple of ``(log_freq, eval_steps, save_freq)``.
    """
    log_freq = max(1, min(100, total_steps // 1000))
    eval_steps = steps_per_epoch if eval_split > 0 else 0
    save_freq = total_steps
    return log_freq, eval_steps, save_freq


def _build_config(
    spec: TrainingJobSpec,
    *,
    dataset_root: Path,
    device: torch.device,
    cache_dir: Path,
    resume_checkpoint: Path | None,
) -> TrainPipelineConfig:
    """Build the LeRobot ``TrainPipelineConfig``.

    Only the job's explicit choices override lerobot defaults: the dataset
    (with its eval split), the policy, the batch size, and the dataloader
    workers. Optimizer, scheduler, learning rate, seed, and cadence all follow
    lerobot's own defaults; the physicalai-synced step/cadence budget is
    applied in ``_train`` once the train-split size is known. Resumed runs load
    the checkpoint's ``train_config.json`` so optimizer, scheduler, and sampler
    semantics come from the original run.
    """
    if resume_checkpoint is not None:
        return _resume_config(resume_checkpoint, dataset_root=dataset_root, device=device, cache_dir=cache_dir)

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id=_DATASET_REPO_ID,
            root=str(dataset_root),
            eval_split=spec.val_split,
        ),
        policy=make_policy_config(spec.policy, device=device.type),
        output_dir=cache_dir,
        job_name=f"lerobot-{spec.policy}",
        resume=False,
        num_workers=_resolve_num_workers(spec),
        batch_size=spec.batch_size,
        rename_map={},
        use_policy_training_preset=True,
        wandb=WandBConfig(enable=False),
    )
    _resolve_optimizer_and_scheduler(cfg)
    return cfg


def _resume_config(
    resume_checkpoint: Path,
    *,
    dataset_root: Path,
    device: torch.device,
    cache_dir: Path,
) -> TrainPipelineConfig:
    """Load the checkpoint's config and point it at the current job."""
    cfg = TrainPipelineConfig.from_pretrained(str(resume_checkpoint / PRETRAINED_MODEL_DIR))
    assert cfg.policy is not None  # noqa: S101
    cfg.resume = True
    cfg.checkpoint_path = resume_checkpoint
    cfg.policy.pretrained_path = resume_checkpoint / PRETRAINED_MODEL_DIR
    cfg.dataset.root = str(dataset_root)
    cfg.output_dir = cache_dir
    cfg.wandb.enable = False
    cfg.policy.device = device.type
    _resolve_optimizer_and_scheduler(cfg)
    return cfg


def _resolve_resume_checkpoint(resume_from: Path | str | None) -> Path | None:
    """Return the base model's raw LeRobot checkpoint directory, if any.

    Lerobot-trained models keep their raw checkpoint under ``lerobot/`` in the
    model directory, which is exactly the layout ``load_training_state``
    expects.
    """
    if resume_from is None:
        return None
    checkpoint = Path(resume_from) / "lerobot"
    if not (checkpoint / "pretrained_model").is_dir():
        logger.warning("Base model %s has no LeRobot checkpoint to resume from", resume_from)
        return None
    return checkpoint


def _make_probe_batch(
    preprocessor: Any,
    dataset: Any,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any] | None:
    """Collate, preprocess, and move one batch to the device for the batch-size probe."""
    collate_fn = lerobot_collate_fn if dataset.meta.has_language_columns else None
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=collate_fn,
    )
    try:
        batch = next(iter(loader))
    except StopIteration:
        return None
    for cam_key in dataset.meta.camera_keys:
        if cam_key in batch and batch[cam_key].dtype == torch.uint8:
            batch[cam_key] = batch[cam_key].to(dtype=torch.float32) / 255.0
    batch = preprocessor(batch)
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def _scale_probe_batch(probe: dict[str, Any], repeats: int) -> dict[str, Any]:
    """Repeat the probe batch to simulate a larger batch size for memory probing."""
    if repeats <= 1:
        return probe
    return {
        k: torch.cat([v] * repeats, dim=0) if isinstance(v, torch.Tensor) and v.ndim >= 1 else v
        for k, v in probe.items()
    }


def _auto_scale_batch_size(
    cfg: TrainPipelineConfig,
    policy: Any,
    preprocessor: Any,
    dataset: Any,
    device: torch.device,
    report: ReportFn,
) -> int:
    """Return the largest power-of-two batch size that fits on the accelerator.

    Mirrors the physicalai engine's ``auto_scale_batch_size`` (Lightning's
    ``BatchSizeFinder``): probe real collated data with a forward/backward pass
    (no optimizer step), doubling until out of memory or a 90% VRAM watermark.
    Skipped on non-CUDA devices, where memory is not the binding constraint.
    """
    if device.type != "cuda":
        return cfg.batch_size
    base = max(1, cfg.batch_size)
    probe = _make_probe_batch(preprocessor, dataset, base, device)
    if probe is None:
        return cfg.batch_size
    try:
        total_memory = torch.cuda.get_device_properties(device).total_memory
    except (AttributeError, RuntimeError):
        total_memory = None

    candidate = base
    best = base
    try:
        while candidate <= _MAX_AUTO_SCALE_BATCH:
            batch = _scale_probe_batch(probe, candidate // base)
            policy.train()
            loss, _output = policy.forward(batch)
            loss.backward()
            policy.zero_grad(set_to_none=True)
            if total_memory is not None and torch.cuda.memory_allocated(device) > 0.9 * total_memory:
                break
            best = candidate
            candidate *= 2
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        logger.warning("Auto batch size: out of memory at %d; using %d", candidate, best)

    if best != base:
        report(0, f"Auto-scaled batch size to {best}", {})
        logger.info("Auto-scaled LeRobot batch size from %d to %d", base, best)
    return best


def _train(  # noqa: C901, PLR0912, PLR0915
    cfg: TrainPipelineConfig,
    *,
    device: torch.device,
    report: ReportFn,
    should_stop: StopFn,
    max_epochs: int,
    auto_scale_batch_size: bool = False,
) -> None:
    """Run the vendored single-process training loop."""
    assert cfg.policy is not None  # noqa: S101
    assert cfg.dataset.root is not None  # noqa: S101
    assert cfg.output_dir is not None  # noqa: S101
    report(0, "Preparing LeRobot dataset", {})
    raw_train, raw_eval = make_train_eval_datasets(cfg)
    dataset: Any = raw_train
    eval_dataset: Any = raw_eval

    report(0, "Creating LeRobot policy", {})
    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta, rename_map={})
    policy = policy.to(device)

    processor_kwargs: dict[str, Any] = {}
    if (cfg.policy.pretrained_path and not cfg.resume) or not cfg.policy.pretrained_path:
        processor_kwargs["dataset_stats"] = dataset.meta.stats
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=str(cfg.policy.pretrained_path) if cfg.policy.pretrained_path is not None else None,
        pretrained_revision=getattr(cfg.policy, "pretrained_revision", None),
        **processor_kwargs,
    )

    if auto_scale_batch_size and not cfg.resume:
        cfg.batch_size = _auto_scale_batch_size(cfg, policy, preprocessor, dataset, device, report)

    # Step budget mirrors the physicalai engine: max_epochs x train batches per
    # epoch, where train batches derive from the train split only (eval_split
    # holds episodes out). Logging/eval/checkpoint cadence follows the same
    # contract (see ``_training_budget``).
    steps_per_epoch = max(1, math.ceil(dataset.num_frames / cfg.batch_size))
    if not cfg.resume:
        cfg.steps = max(1, max_epochs * steps_per_epoch)

    report(0, "Creating optimizer and scheduler", {})
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    step = 0
    checkpoint_path: Path | None = None
    if cfg.resume:
        assert cfg.checkpoint_path is not None  # noqa: S101
        checkpoint_path = cfg.checkpoint_path
        step, optimizer, lr_scheduler = load_training_state(checkpoint_path, optimizer, lr_scheduler)
        # A resumed run continues for the job's epoch budget, same basis as a fresh run.
        cfg.steps = step + max_epochs * steps_per_epoch
        logger.info("Resuming LeRobot training at step %d (total %d)", step, cfg.steps)

    cfg.log_freq, cfg.eval_steps, cfg.save_freq = _training_budget(
        total_steps=cfg.steps, steps_per_epoch=steps_per_epoch, eval_split=cfg.dataset.eval_split
    )

    # Data order is a pure function of (seed, epoch); resume is sample-exact.
    sampler = EpisodeAwareSampler(
        dataset.meta.episodes["dataset_from_index"],
        dataset.meta.episodes["dataset_to_index"],
        episode_indices_to_use=dataset.episodes,
        drop_n_last_frames=getattr(cfg.policy, "drop_n_last_frames", 0),
        shuffle=True,
        seed=cfg.seed if cfg.seed is not None else 0,
        absolute_to_relative_idx=dataset.absolute_to_relative_idx,
    )
    if checkpoint_path is not None and step > 0:
        saved_batch_size = load_training_batch_size(checkpoint_path) or cfg.batch_size
        sampler_state = compute_sampler_state(step, len(sampler), saved_batch_size, 1)
        sampler.load_state_dict(sampler_state)

    collate_fn = lerobot_collate_fn if dataset.meta.has_language_columns else None
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=False,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
        collate_fn=collate_fn,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
        persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
    )

    eval_dataloader: torch.utils.data.DataLoader[Any] | None = None
    if eval_dataset is not None:
        eval_collate_fn = lerobot_collate_fn if dataset.meta.has_language_columns else None
        eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset,  # type: ignore[arg-type]
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
            collate_fn=eval_collate_fn,
            prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
            persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
        )

    metrics = _MetricsWriter(
        report=report,
        cache_dir=cfg.output_dir,
        max_steps=cfg.steps,
        steps_per_epoch=steps_per_epoch,
    )
    autocast_ctx = _autocast_context(cfg, device)

    dl_iter = cycle(dataloader)
    policy.train()
    report(0, "Training model", _plan_extra_info(cfg, dataset, eval_dataset, steps_per_epoch, max_epochs))
    for current in range(step, cfg.steps):
        if should_stop():
            logger.info("LeRobot training canceled at step %d", current)
            return

        batch = next(dl_iter)
        raw_batch = batch
        for cam_key in dataset.meta.camera_keys:
            if cam_key in batch and batch[cam_key].dtype == torch.uint8:
                batch[cam_key] = batch[cam_key].to(dtype=torch.float32) / 255.0
        batch = preprocessor(batch)
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        if current == step:
            _report_input_sanity(report, raw_batch, batch, dataset.meta.camera_keys)

        with autocast_ctx or nullcontext():
            loss, _output_dict = policy.forward(batch)

        loss.backward()
        grad_clip_norm = getattr(cfg.optimizer, "grad_clip_norm", 0.0)
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip_norm)
        optimizer.step()
        optimizer.zero_grad()
        if lr_scheduler is not None:
            lr_scheduler.step()
        update = getattr(policy, "update", None)
        if callable(update):
            update()

        step_done = current + 1
        cadence = min(cfg.log_freq, _METRIC_LOG_EARLY_CADENCE) if step_done <= _METRIC_LOG_EARLY_STEPS else cfg.log_freq
        if cfg.log_freq > 0 and step_done % cadence == 0:
            metrics.on_log_step(step_done, loss.item())
        if cfg.eval_steps > 0 and eval_dataloader is not None and step_done % cfg.eval_steps == 0:
            eval_loss = _evaluate(policy, preprocessor, eval_dataloader, dataset.meta.camera_keys, autocast_ctx, device)
            metrics.on_eval(step_done, eval_loss)
        if cfg.save_checkpoint and (step_done % cfg.save_freq == 0 or step_done == cfg.steps):
            checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step_done)
            save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                step=step_done,
                cfg=cfg,
                policy=policy,
                optimizer=optimizer,
                scheduler=lr_scheduler,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                num_processes=1,
                batch_size=cfg.batch_size,
            )
            update_last_checkpoint(checkpoint_dir)
            logger.info("Saved checkpoint at step %d", step_done)
    metrics.close()


def _evaluate(
    policy: PreTrainedPolicy,
    preprocessor: Any,
    eval_dataloader: Any,
    camera_keys: list[str],
    autocast_ctx: Any,
    device: torch.device,
) -> float:
    """Compute mean eval loss over the held-out dataloader."""
    policy.eval()
    eval_loss_sum = 0.0
    n_batches = 0
    with torch.no_grad(), autocast_ctx or nullcontext():
        for eval_batch in eval_dataloader:
            for cam_key in camera_keys:
                if cam_key in eval_batch and eval_batch[cam_key].dtype == torch.uint8:
                    eval_batch[cam_key] = eval_batch[cam_key].to(dtype=torch.float32) / 255.0
            preprocessed_batch = preprocessor(eval_batch)
            preprocessed_batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in preprocessed_batch.items()
            }
            loss, _ = policy.forward(preprocessed_batch)
            eval_loss_sum += loss.item()
            n_batches += 1
    policy.train()
    return eval_loss_sum / max(n_batches, 1)


def _autocast_context(cfg: TrainPipelineConfig, device: torch.device) -> Any:
    """Return the mixed-precision context the policy's dtype calls for, if any.

    Mirrors accelerate's autocast activation in ``lerobot_train.py``: bf16/fp16
    policy dtypes enable autocast, float32 (or unset) does not.
    """
    policy_dtype = getattr(cfg.policy, "dtype", None)
    if policy_dtype == "bfloat16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    if policy_dtype == "float16":
        return torch.autocast(device_type=device.type, dtype=torch.float16)
    return None


class _MetricsWriter:
    """Write progress telemetry and a Lightning-style ``metrics.csv``.

    Consumes the same ``report`` contract as the physicalai path
    (``(progress, message, extra_info)`` with the fields
    :func:`services.training_backends._log_format.render_progress_log`
    understands), so LeRobot job logs and metrics graphs look identical.
    """

    def __init__(self, *, report: Callable[..., None], cache_dir: Path, max_steps: int, steps_per_epoch: int) -> None:
        self.report = report
        self.max_steps = max_steps
        self.steps_per_epoch = max(1, steps_per_epoch)
        self.csv_path = Path(cache_dir) / "version_0" / "metrics.csv"
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._csv = self.csv_path.open("w", newline="")
        self._writer = csv.writer(self._csv)
        self._writer.writerow(["epoch", "step", "train/loss_step", "val/loss"])
        self._csv.flush()
        self._epoch = 0

    def on_log_step(self, step: int, loss: float) -> None:
        self._epoch = step // self.steps_per_epoch
        self._writer.writerow([self._epoch, step, f"{loss:.6f}", ""])
        self._csv.flush()
        self.report(
            min(99, int(step / self.max_steps * 100)),
            None,
            {
                "global_step": step,
                "max_steps": self.max_steps,
                "epoch": self._epoch,
                "train/loss_step": loss,
            },
        )

    def on_eval(self, step: int, val_loss: float, elapsed_s: float | None = None) -> None:
        self._epoch = step // self.steps_per_epoch
        self._writer.writerow([self._epoch, step, "", f"{val_loss:.6f}"])
        self._csv.flush()
        elapsed_s = elapsed_s if elapsed_s is not None else 0.0
        self.report(
            min(99, int(step / self.max_steps * 100)),
            None,
            {
                "val_event": "end",
                "global_step": step,
                "max_steps": self.max_steps,
                "val/loss": val_loss,
                "val_elapsed_s": elapsed_s,
            },
        )

    def close(self) -> None:
        self._csv.close()


def _publish(
    cfg: TrainPipelineConfig,
    *,
    cache_dir: Path,
    output_dir: Path,
    report: ReportFn,
    export_backends: ExportBackends | None = None,
) -> None:
    """Move the finished run into its final location and export it."""
    from physicalai.policies.lerobot.utils.checkpoint_converter import lerobot_to_lightning

    from training.job import CHECKPOINT_NAME, EXPORTS_DIRNAME

    checkpoints_dir = cache_dir / "checkpoints"
    final = _latest_checkpoint(checkpoints_dir)
    if final is None:
        msg = "No checkpoint was produced by LeRobot training"
        raise RuntimeError(msg)
    assert cfg.policy is not None  # noqa: S101

    report(99, "Converting checkpoint", {})
    lerobot_dir = cache_dir / "lerobot"
    shutil.move(str(final), str(lerobot_dir))
    # The `last` symlink (and the now-empty checkpoints/ tree) only point at
    # checkpoints that no longer exist; drop them so the published model holds
    # a single raw checkpoint under lerobot/.
    if (checkpoints_dir / "last").is_symlink():
        (checkpoints_dir / "last").unlink()
    if checkpoints_dir.exists() and not any(checkpoints_dir.iterdir()):
        checkpoints_dir.rmdir()

    lerobot_to_lightning(lerobot_dir / PRETRAINED_MODEL_DIR, cache_dir / CHECKPOINT_NAME, policy_name=cfg.policy.type)

    report(99, "Publishing model", {})
    _move_to_output(cache_dir, output_dir)

    report(100, "Exporting model", {})
    _export_backends(output_dir, EXPORTS_DIRNAME, export_backends)
    report(100, "Model saved", {})


def _latest_checkpoint(checkpoints_dir: Path) -> Path | None:
    """Return the highest checkpoint directory, if any.

    LeRobot names step dirs as zero-padded digits (``000010``); older layouts
    used a ``step_`` prefix, which is accepted as a fallback. The ``last``
    symlink is excluded either way.
    """
    if not checkpoints_dir.is_dir():
        return None
    candidates = sorted(checkpoints_dir.glob("[0-9]*"))
    if not candidates:
        candidates = sorted(checkpoints_dir.glob("step_*"))
    return candidates[-1] if candidates else None


def _move_to_output(cache_dir: Path, output_dir: Path) -> None:
    """Move the finished training cache into its final location."""
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    shutil.move(str(cache_dir), str(output_dir))


def _export_backends(output_dir: Path, exports_dirname: str, export_backends: ExportBackends | None) -> None:
    """Export the trained policy to the requested backends, best-effort.

    Uses :class:`ExportableLeRobotPolicy` so torch/onnx/openvino can be
    produced. A failed export must not abort the job, since ``model.ckpt`` is
    already the primary loadable artifact.
    """
    from physicalai.export.backends import ExportBackend
    from physicalai.policies.lerobot.export import ExportableLeRobotPolicy

    from training.job import CHECKPOINT_NAME

    try:
        policy = ExportableLeRobotPolicy.load_from_checkpoint(output_dir / CHECKPOINT_NAME)
        supported = [ExportBackend(b) for b in policy.get_supported_export_backends()]
    except Exception:
        logger.exception("Failed to load LeRobot policy for export")
        return

    selected = [ExportBackend(b) for b in export_backends] if export_backends else supported
    for backend in selected:
        if backend not in supported:
            logger.warning("Skipping %s export: policy does not support it", backend.value)
            continue
        try:
            policy.export(output_dir / exports_dirname / backend.value, backend=backend)
        except Exception:
            logger.exception("Failed exporting model to %s format", backend.value)
