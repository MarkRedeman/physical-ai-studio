# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Configuration for SmolVLA model."""

from __future__ import annotations

from dataclasses import dataclass, field

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
class SmolVLAConfig(Config):
    """Configuration for SmolVLA flow matching model."""

    n_obs_steps: int = field(default=1, metadata=_metadata("Number of observation steps to use.", group="io", group_title="Input / Output", ge=1))
    chunk_size: int = field(default=50, metadata=_metadata("Number of action steps predicted per chunk.", group="io", group_title="Input / Output", ge=1))
    n_action_steps: int = field(default=50, metadata=_metadata("Number of predicted actions to execute before querying again. Must be <= chunk_size.", group="io", group_title="Input / Output", ge=1))
    max_state_dim: int = field(default=32, metadata=_metadata("Maximum state vector dimension; shorter vectors are padded.", group="io", group_title="Input / Output", ge=1))
    max_action_dim: int = field(default=32, metadata=_metadata("Maximum action vector dimension; shorter vectors are padded.", group="io", group_title="Input / Output", ge=1))

    resize_imgs_with_padding: tuple[int, int] = field(default=(512, 512), metadata=_metadata("Target padded image resolution as (height, width).", group="preprocessing", group_title="Preprocessing", min_length=2, max_length=2))
    empty_cameras: int = field(default=0, metadata=_metadata("Number of empty camera images to add for datasets with missing camera slots.", group="preprocessing", group_title="Preprocessing", ge=0))
    adapt_to_pi_aloha: bool = field(default=False, metadata=_metadata("Convert Aloha joint/gripper values to pi internal runtime space.", group="preprocessing", group_title="Preprocessing"))
    tokenizer_max_length: int = field(default=48, metadata=_metadata("Maximum tokenized language length.", group="preprocessing", group_title="Preprocessing", ge=1))
    pad_language_to: str = field(default="max_length", metadata=_metadata("Language token padding strategy, typically 'max_length' for export compatibility.", group="preprocessing", group_title="Preprocessing"))

    vlm_model_name: str = field(default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct", metadata=_metadata("Name or path of the VLM backbone model.", group="architecture", group_title="Architecture"))
    load_vlm_weights: bool = field(default=False, metadata=_metadata("Load pretrained VLM weights. Useful when training the expert from scratch.", group="architecture", group_title="Architecture"))
    add_image_special_tokens: bool = field(default=False, metadata=_metadata("Add special tokens around image features.", group="architecture", group_title="Architecture"))
    attention_mode: str = field(default="cross_attn", metadata=_metadata("Attention mechanism used by the model.", group="architecture", group_title="Architecture"))
    prefix_length: int = field(default=-1, metadata=_metadata("Attention prefix length. Negative values use default behavior.", group="architecture", group_title="Architecture"))
    num_expert_layers: int = field(default=-1, metadata=_metadata("Number of action expert layers. Values <= 0 match the VLM layer count.", group="architecture", group_title="Architecture"))
    num_vlm_layers: int = field(default=16, metadata=_metadata("Number of VLM layers to use.", group="architecture", group_title="Architecture", ge=1))
    self_attn_every_n_layers: int = field(default=2, metadata=_metadata("Frequency for interleaving self-attention layers.", group="architecture", group_title="Architecture", ge=1))
    expert_width_multiplier: float = field(default=0.75, metadata=_metadata("Action expert hidden width as a multiplier of the VLM hidden width.", group="architecture", group_title="Architecture", gt=0.0))

    min_period: float = field(default=4e-3, metadata=_metadata("Minimum period for sine-cosine timestep positional encoding.", group="flow_matching", group_title="Flow Matching", gt=0.0))
    max_period: float = field(default=4.0, metadata=_metadata("Maximum period for sine-cosine timestep positional encoding.", group="flow_matching", group_title="Flow Matching", gt=0.0))
    use_random_input_noise: bool = field(default=True, metadata=_metadata("Use random noise as the initial denoising input. If false, zeros are used.", group="flow_matching", group_title="Flow Matching"))
    num_steps: int = field(default=10, metadata=_metadata("Number of flow matching decoding steps.", group="flow_matching", group_title="Flow Matching", ge=1))
    use_cache: bool = field(default=True, metadata=_metadata("Enable attention cache during decoding.", group="flow_matching", group_title="Flow Matching"))

    freeze_vision_encoder: bool = field(default=True, metadata=_metadata("Freeze the vision encoder during fine-tuning.", group="fine_tuning", group_title="Fine Tuning"))
    train_expert_only: bool = field(default=True, metadata=_metadata("Train only expert layers.", group="fine_tuning", group_title="Fine Tuning"))
    train_state_proj: bool = field(default=True, metadata=_metadata("Train state projection layers.", group="fine_tuning", group_title="Fine Tuning"))
    compile_model: bool = False

    optimizer_lr: float = field(default=1e-4, metadata=_metadata("Learning rate.", group="optimizer", group_title="Optimizer", gt=0.0))
    optimizer_betas: tuple[float, float] = field(default=(0.9, 0.95), metadata=_metadata("AdamW beta coefficients as (beta1, beta2).", group="optimizer", group_title="Optimizer", min_length=2, max_length=2))
    optimizer_eps: float = field(default=1e-8, metadata=_metadata("Optimizer epsilon.", group="optimizer", group_title="Optimizer", gt=0.0))
    optimizer_weight_decay: float = field(default=1e-10, metadata=_metadata("Weight decay coefficient.", group="optimizer", group_title="Optimizer", ge=0.0))
    optimizer_grad_clip_norm: float = field(default=10, metadata=_metadata("Maximum gradient norm.", group="optimizer", group_title="Optimizer", gt=0.0))
    scheduler_warmup_steps: int = field(default=1_000, metadata=_metadata("Linear warmup steps.", group="optimizer", group_title="Optimizer", ge=0))
    scheduler_decay_steps: int = field(default=30_000, metadata=_metadata("Cosine decay horizon in steps.", group="optimizer", group_title="Optimizer", ge=1))
    scheduler_decay_lr: float = field(default=2.5e-6, metadata=_metadata("Final learning rate after decay.", group="optimizer", group_title="Optimizer", ge=0.0))

    def __post_init__(self) -> None:
        """Validate configuration parameters after initialization."""
        if self.n_action_steps > self.chunk_size:
            msg = (
                "The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
            raise ValueError(msg)
