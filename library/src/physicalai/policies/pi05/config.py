# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Configuration for Pi05 model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from physicalai.config import Config


def _metadata(description: str, *, group: str, group_title: str, title: str | None = None, **validation: object) -> dict[str, object]:
    """Build pydantic.Field-compatible metadata plus UI grouping hints."""
    json_schema_extra = {"group": group, "group_title": group_title}
    if isinstance(validation.get("json_schema_extra"), dict):
        json_schema_extra.update(validation.pop("json_schema_extra"))
    metadata = {"description": description, "json_schema_extra": json_schema_extra, **validation}
    if title is not None:
        metadata["title"] = title
    return metadata


@dataclass(frozen=True)
class Pi05Config(Config):
    """Configuration for Pi05 flow matching model."""

    paligemma_variant: Literal["gemma_300m", "gemma_2b"] = field(
        default="gemma_2b",
        metadata=_metadata(
            "Gemma variant used for the vision-language backbone.",
            group="backbone",
            group_title="Backbone",
            json_schema_extra={"group": "backbone", "group_title": "Backbone", "enum": ["gemma_300m", "gemma_2b"]},
        ),
    )
    action_expert_variant: Literal["gemma_300m", "gemma_2b"] = field(
        default="gemma_300m",
        metadata=_metadata(
            "Gemma variant used for the action expert.",
            group="backbone",
            group_title="Backbone",
            json_schema_extra={"group": "backbone", "group_title": "Backbone", "enum": ["gemma_300m", "gemma_2b"]},
        ),
    )
    dtype: Literal["bfloat16", "float32"] = field(
        default="bfloat16",
        metadata=_metadata(
            "Model weight precision.",
            group="backbone",
            group_title="Backbone",
            json_schema_extra={"group": "backbone", "group_title": "Backbone", "enum": ["bfloat16", "float32"]},
        ),
    )

    n_obs_steps: int = field(default=1, metadata=_metadata("Number of observation steps to use.", group="io", group_title="Input / Output", ge=1))
    chunk_size: int = field(default=50, metadata=_metadata("Number of action steps predicted per chunk.", group="io", group_title="Input / Output", ge=1))
    n_action_steps: int = field(
        default=50,
        metadata=_metadata("Number of predicted actions to execute before querying again. Must be <= chunk_size.", group="io", group_title="Input / Output", ge=1),
    )
    max_state_dim: int = field(default=32, metadata=_metadata("Maximum state vector dimension; shorter vectors are padded.", group="io", group_title="Input / Output", ge=1))
    max_action_dim: int = field(default=32, metadata=_metadata("Maximum action vector dimension; shorter vectors are padded.", group="io", group_title="Input / Output", ge=1))

    num_inference_steps: int = field(default=10, metadata=_metadata("Number of denoising steps for inference.", group="flow_matching", group_title="Flow Matching", ge=1))
    time_sampling_beta_alpha: float = field(default=1.5, metadata=_metadata("Alpha parameter for beta-distribution timestep sampling.", group="flow_matching", group_title="Flow Matching", gt=0.0))
    time_sampling_beta_beta: float = field(default=1.0, metadata=_metadata("Beta parameter for beta-distribution timestep sampling.", group="flow_matching", group_title="Flow Matching", gt=0.0))
    time_sampling_scale: float = field(default=0.999, metadata=_metadata("Scale factor for timestep sampling.", group="flow_matching", group_title="Flow Matching"))
    time_sampling_offset: float = field(default=0.001, metadata=_metadata("Offset applied to timestep sampling.", group="flow_matching", group_title="Flow Matching"))
    min_period: float = field(default=4e-3, metadata=_metadata("Minimum period for sine-cosine timestep positional encoding.", group="flow_matching", group_title="Flow Matching", gt=0.0))
    max_period: float = field(default=4.0, metadata=_metadata("Maximum period for sine-cosine timestep positional encoding.", group="flow_matching", group_title="Flow Matching", gt=0.0))
    use_random_input_noise: bool = field(default=True, metadata=_metadata("Use random noise as the initial denoising input. If false, zeros are used.", group="flow_matching", group_title="Flow Matching"))

    image_resolution: tuple[int, int] = field(default=(224, 224), metadata=_metadata("Target image resolution as (height, width).", group="preprocessing", group_title="Preprocessing", min_length=2, max_length=2))
    empty_cameras: int = field(default=0, metadata=_metadata("Number of empty camera slots to add.", group="preprocessing", group_title="Preprocessing", ge=0))
    tokenizer_max_length: int = field(default=200, metadata=_metadata("Maximum tokenized language sequence length.", group="preprocessing", group_title="Preprocessing", ge=1))
    normalization_mode: Literal["MEAN_STD", "QUANTILES"] = field(
        default="QUANTILES",
        metadata=_metadata(
            "State/action normalization mode. QUANTILES is robust to outliers; MEAN_STD uses mean/std.",
            group="preprocessing",
            group_title="Preprocessing",
            json_schema_extra={"group": "preprocessing", "group_title": "Preprocessing", "enum": ["MEAN_STD", "QUANTILES"]},
        ),
    )

    gradient_checkpointing: bool = field(default=True, metadata=_metadata("Enable gradient checkpointing to reduce memory usage.", group="training", group_title="Training"))
    compile_model: bool = False
    compile_mode: str = field(default="max-autotune", metadata=_metadata("Torch compile mode.", group="training", group_title="Training"))
    freeze_vision_encoder: bool = field(default=False, metadata=_metadata("Freeze the vision encoder during fine-tuning.", group="training", group_title="Training"))
    train_expert_only: bool = field(default=False, metadata=_metadata("Train only the action expert while freezing other trainable components.", group="training", group_title="Training"))

    optimizer_lr: float = field(default=2.5e-5, metadata=_metadata("Learning rate.", group="optimizer", group_title="Optimizer", gt=0.0))
    optimizer_betas: tuple[float, float] = field(default=(0.9, 0.95), metadata=_metadata("Adam beta coefficients as (beta1, beta2).", group="optimizer", group_title="Optimizer", min_length=2, max_length=2))
    optimizer_eps: float = field(default=1e-8, metadata=_metadata("Optimizer epsilon.", group="optimizer", group_title="Optimizer", gt=0.0))
    optimizer_weight_decay: float = field(default=0.01, metadata=_metadata("Weight decay coefficient.", group="optimizer", group_title="Optimizer", ge=0.0))
    optimizer_grad_clip_norm: float = field(default=1.0, metadata=_metadata("Maximum gradient norm.", group="optimizer", group_title="Optimizer", gt=0.0))
    scheduler_warmup_steps: int = field(default=1_000, metadata=_metadata("Linear warmup steps.", group="optimizer", group_title="Optimizer", ge=0))
    scheduler_decay_steps: int | None = field(default=30_000, metadata=_metadata("Cosine decay horizon in steps. Null auto-scales to total training steps.", group="optimizer", group_title="Optimizer", ge=1))
    scheduler_decay_lr: float = field(default=2.5e-6, metadata=_metadata("Final learning rate after decay.", group="optimizer", group_title="Optimizer", ge=0.0))

    def __post_init__(self) -> None:
        """Validate configuration parameters after initialization."""
        if self.n_action_steps > self.chunk_size:
            msg = f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            raise ValueError(msg)
        if self.paligemma_variant not in {"gemma_300m", "gemma_2b"}:
            msg = f"Invalid paligemma_variant: {self.paligemma_variant}"
            raise ValueError(msg)
        if self.action_expert_variant not in {"gemma_300m", "gemma_2b"}:
            msg = f"Invalid action_expert_variant: {self.action_expert_variant}"
            raise ValueError(msg)
        if self.dtype not in {"bfloat16", "float32"}:
            msg = f"Invalid dtype: {self.dtype}"
            raise ValueError(msg)
