# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Configuration for SmolVLA model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from physicalai.config import Config


def _metadata(description: str, **validation: object) -> dict[str, object]:
    """Build field metadata for docs/schema adapters."""
    return {"description": description, **validation}


@dataclass(frozen=True)
class SmolVLAInputOutputConfig(Config):
    """Input/output dimensions and action chunking settings for SmolVLA."""

    n_obs_steps: int = field(default=1, metadata=_metadata("Number of observation steps to use.", ge=1))
    chunk_size: int = field(default=50, metadata=_metadata("Number of action steps predicted per chunk.", ge=1))
    n_action_steps: int = field(
        default=50,
        metadata=_metadata("Number of predicted actions to execute before querying again. Must be <= chunk_size.", ge=1),
    )
    max_state_dim: int = field(
        default=32,
        metadata=_metadata("Maximum state vector dimension; shorter vectors are padded.", ge=1),
    )
    max_action_dim: int = field(
        default=32,
        metadata=_metadata("Maximum action vector dimension; shorter vectors are padded.", ge=1),
    )

    def __post_init__(self) -> None:
        """Validate input/output dimensions."""
        if self.n_action_steps > self.chunk_size:
            msg = (
                "The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
            raise ValueError(msg)
        if self.n_obs_steps <= 0 or self.chunk_size <= 0 or self.n_action_steps <= 0:
            msg = "Observation and action step counts must be positive."
            raise ValueError(msg)
        if self.max_state_dim <= 0 or self.max_action_dim <= 0:
            msg = "State and action dimensions must be positive."
            raise ValueError(msg)


@dataclass(frozen=True)
class SmolVLAPreprocessingConfig(Config):
    """Image, tokenization, and dataset-space preprocessing settings for SmolVLA."""

    resize_imgs_with_padding: tuple[int, int] = field(
        default=(512, 512),
        metadata=_metadata("Target padded image resolution as (height, width).", min_items=2, max_items=2),
    )
    empty_cameras: int = field(
        default=0,
        metadata=_metadata("Number of empty camera images to add for datasets with missing camera slots.", ge=0),
    )
    adapt_to_pi_aloha: bool = field(
        default=False,
        metadata=_metadata("Convert Aloha joint/gripper values to pi internal runtime space."),
    )
    tokenizer_max_length: int = field(default=48, metadata=_metadata("Maximum tokenized language length.", ge=1))
    pad_language_to: str = field(
        default="max_length",
        metadata=_metadata("Language token padding strategy, typically 'max_length' for export compatibility."),
    )

    def __post_init__(self) -> None:
        """Validate preprocessing parameters."""
        if len(self.resize_imgs_with_padding) != 2 or any(size <= 0 for size in self.resize_imgs_with_padding):
            msg = f"`resize_imgs_with_padding` must contain two positive integers. Got {self.resize_imgs_with_padding}."
            raise ValueError(msg)
        if self.empty_cameras < 0:
            msg = "`empty_cameras` must be non-negative."
            raise ValueError(msg)
        if self.tokenizer_max_length <= 0:
            msg = "`tokenizer_max_length` must be positive."
            raise ValueError(msg)


@dataclass(frozen=True)
class SmolVLAArchitectureConfig(Config):
    """VLM and action expert architecture settings for SmolVLA."""

    vlm_model_name: str = field(
        default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        metadata=_metadata("Name or path of the VLM backbone model."),
    )
    load_vlm_weights: bool = field(
        default=False,
        metadata=_metadata("Load pretrained VLM weights. Useful when training the expert from scratch."),
    )
    add_image_special_tokens: bool = field(
        default=False,
        metadata=_metadata("Add special tokens around image features."),
    )
    attention_mode: str = field(default="cross_attn", metadata=_metadata("Attention mechanism used by the model."))
    prefix_length: int = field(default=-1, metadata=_metadata("Attention prefix length. Negative values use default behavior."))
    num_expert_layers: int = field(
        default=-1,
        metadata=_metadata("Number of action expert layers. Values <= 0 match the VLM layer count."),
    )
    num_vlm_layers: int = field(default=16, metadata=_metadata("Number of VLM layers to use.", ge=1))
    self_attn_every_n_layers: int = field(
        default=2,
        metadata=_metadata("Frequency for interleaving self-attention layers.", ge=1),
    )
    expert_width_multiplier: float = field(
        default=0.75,
        metadata=_metadata("Action expert hidden width as a multiplier of the VLM hidden width.", gt=0.0),
    )

    def __post_init__(self) -> None:
        """Validate architecture parameters."""
        if self.num_vlm_layers <= 0:
            msg = "`num_vlm_layers` must be positive."
            raise ValueError(msg)
        if self.self_attn_every_n_layers <= 0:
            msg = "`self_attn_every_n_layers` must be positive."
            raise ValueError(msg)
        if self.expert_width_multiplier <= 0:
            msg = "`expert_width_multiplier` must be positive."
            raise ValueError(msg)


@dataclass(frozen=True)
class SmolVLAFlowMatchingConfig(Config):
    """Flow matching and decoding settings for SmolVLA."""

    num_steps: int = field(default=10, metadata=_metadata("Number of flow matching decoding steps.", ge=1))
    use_cache: bool = field(default=True, metadata=_metadata("Enable attention cache during decoding."))
    min_period: float = field(
        default=4e-3,
        metadata=_metadata("Minimum period for sine-cosine timestep positional encoding.", gt=0.0),
    )
    max_period: float = field(
        default=4.0,
        metadata=_metadata("Maximum period for sine-cosine timestep positional encoding.", gt=0.0),
    )
    use_random_input_noise: bool = field(
        default=True,
        metadata=_metadata("Use random noise as the initial denoising input. If false, zeros are used."),
    )

    def __post_init__(self) -> None:
        """Validate flow matching parameters."""
        if self.num_steps <= 0:
            msg = "`num_steps` must be positive."
            raise ValueError(msg)
        if self.min_period <= 0 or self.max_period <= 0:
            msg = "Period bounds must be positive."
            raise ValueError(msg)
        if self.min_period > self.max_period:
            msg = "`min_period` cannot be greater than `max_period`."
            raise ValueError(msg)


@dataclass(frozen=True)
class SmolVLAFineTuningConfig(Config):
    """Fine-tuning and compilation settings for SmolVLA."""

    freeze_vision_encoder: bool = field(
        default=True,
        metadata=_metadata("Freeze the vision encoder during fine-tuning."),
    )
    train_expert_only: bool = field(default=True, metadata=_metadata("Train only expert layers."))
    train_state_proj: bool = field(default=True, metadata=_metadata("Train state projection layers."))
    compile_model: bool = field(default=False, metadata=_metadata("Enable torch.compile for the model."))


@dataclass(frozen=True)
class SmolVLAOptimizerConfig(Config):
    """Optimizer and scheduler settings for SmolVLA."""

    optimizer_lr: float = field(default=1e-4, metadata=_metadata("Learning rate.", gt=0.0))
    optimizer_betas: tuple[float, float] = field(
        default=(0.9, 0.95),
        metadata=_metadata("AdamW beta coefficients as (beta1, beta2).", min_items=2, max_items=2),
    )
    optimizer_eps: float = field(default=1e-8, metadata=_metadata("Optimizer epsilon.", gt=0.0))
    optimizer_weight_decay: float = field(default=1e-10, metadata=_metadata("Weight decay coefficient.", ge=0.0))
    optimizer_grad_clip_norm: float = field(default=10, metadata=_metadata("Maximum gradient norm.", gt=0.0))
    scheduler_warmup_steps: int = field(default=1_000, metadata=_metadata("Linear warmup steps.", ge=0))
    scheduler_decay_steps: int = field(default=30_000, metadata=_metadata("Cosine decay horizon in steps.", ge=1))
    scheduler_decay_lr: float = field(default=2.5e-6, metadata=_metadata("Final learning rate after decay.", ge=0.0))

    def __post_init__(self) -> None:
        """Validate optimizer and scheduler parameters."""
        if self.optimizer_lr <= 0:
            msg = "`optimizer_lr` must be positive."
            raise ValueError(msg)
        if len(self.optimizer_betas) != 2 or any(not 0 <= beta < 1 for beta in self.optimizer_betas):
            msg = f"`optimizer_betas` must contain two values in [0, 1). Got {self.optimizer_betas}."
            raise ValueError(msg)
        if self.optimizer_eps <= 0:
            msg = "`optimizer_eps` must be positive."
            raise ValueError(msg)
        if self.optimizer_weight_decay < 0:
            msg = "`optimizer_weight_decay` must be non-negative."
            raise ValueError(msg)
        if self.optimizer_grad_clip_norm <= 0:
            msg = "`optimizer_grad_clip_norm` must be positive."
            raise ValueError(msg)
        if self.scheduler_warmup_steps < 0:
            msg = "`scheduler_warmup_steps` must be non-negative."
            raise ValueError(msg)
        if self.scheduler_decay_steps <= 0:
            msg = "`scheduler_decay_steps` must be positive."
            raise ValueError(msg)
        if self.scheduler_decay_lr < 0:
            msg = "`scheduler_decay_lr` must be non-negative."
            raise ValueError(msg)


@dataclass(frozen=True, init=False)
class SmolVLAConfig(Config):
    """Configuration for SmolVLA flow matching model.

    The preferred shape groups related settings into nested config objects. The constructor also accepts legacy flat
    SmolVLA keyword arguments for checkpoint compatibility, HuggingFace config loading, and existing callers.
    """

    io: SmolVLAInputOutputConfig = field(
        default_factory=SmolVLAInputOutputConfig,
        metadata=_metadata("Input/output dimensions and action chunking settings."),
    )
    preprocessing: SmolVLAPreprocessingConfig = field(
        default_factory=SmolVLAPreprocessingConfig,
        metadata=_metadata("Image, tokenization, and dataset-space preprocessing settings."),
    )
    architecture: SmolVLAArchitectureConfig = field(
        default_factory=SmolVLAArchitectureConfig,
        metadata=_metadata("VLM and action expert architecture settings."),
    )
    flow_matching: SmolVLAFlowMatchingConfig = field(
        default_factory=SmolVLAFlowMatchingConfig,
        metadata=_metadata("Flow matching and decoding settings."),
    )
    fine_tuning: SmolVLAFineTuningConfig = field(
        default_factory=SmolVLAFineTuningConfig,
        metadata=_metadata("Fine-tuning and compilation settings."),
    )
    optimizer: SmolVLAOptimizerConfig = field(
        default_factory=SmolVLAOptimizerConfig,
        metadata=_metadata("Optimizer and scheduler settings."),
    )

    def __init__(  # noqa: PLR0913
        self,
        *,
        io: SmolVLAInputOutputConfig | None = None,
        preprocessing: SmolVLAPreprocessingConfig | None = None,
        architecture: SmolVLAArchitectureConfig | None = None,
        flow_matching: SmolVLAFlowMatchingConfig | None = None,
        fine_tuning: SmolVLAFineTuningConfig | None = None,
        optimizer: SmolVLAOptimizerConfig | None = None,
        n_obs_steps: int = 1,
        chunk_size: int = 50,
        n_action_steps: int = 50,
        max_state_dim: int = 32,
        max_action_dim: int = 32,
        resize_imgs_with_padding: tuple[int, int] = (512, 512),
        empty_cameras: int = 0,
        adapt_to_pi_aloha: bool = False,
        tokenizer_max_length: int = 48,
        pad_language_to: str = "max_length",
        vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        load_vlm_weights: bool = False,
        add_image_special_tokens: bool = False,
        attention_mode: str = "cross_attn",
        prefix_length: int = -1,
        num_expert_layers: int = -1,
        num_vlm_layers: int = 16,
        self_attn_every_n_layers: int = 2,
        expert_width_multiplier: float = 0.75,
        min_period: float = 4e-3,
        max_period: float = 4.0,
        use_random_input_noise: bool = True,
        num_steps: int = 10,
        use_cache: bool = True,
        freeze_vision_encoder: bool = True,
        train_expert_only: bool = True,
        train_state_proj: bool = True,
        compile_model: bool = False,
        optimizer_lr: float = 1e-4,
        optimizer_betas: tuple[float, float] = (0.9, 0.95),
        optimizer_eps: float = 1e-8,
        optimizer_weight_decay: float = 1e-10,
        optimizer_grad_clip_norm: float = 10,
        scheduler_warmup_steps: int = 1_000,
        scheduler_decay_steps: int = 30_000,
        scheduler_decay_lr: float = 2.5e-6,
    ) -> None:
        """Initialize SmolVLA config from grouped configs or legacy flat keyword arguments."""
        object.__setattr__(
            self,
            "io",
            io
            or SmolVLAInputOutputConfig(
                n_obs_steps=n_obs_steps,
                chunk_size=chunk_size,
                n_action_steps=n_action_steps,
                max_state_dim=max_state_dim,
                max_action_dim=max_action_dim,
            ),
        )
        object.__setattr__(
            self,
            "preprocessing",
            preprocessing
            or SmolVLAPreprocessingConfig(
                resize_imgs_with_padding=resize_imgs_with_padding,
                empty_cameras=empty_cameras,
                adapt_to_pi_aloha=adapt_to_pi_aloha,
                tokenizer_max_length=tokenizer_max_length,
                pad_language_to=pad_language_to,
            ),
        )
        object.__setattr__(
            self,
            "architecture",
            architecture
            or SmolVLAArchitectureConfig(
                vlm_model_name=vlm_model_name,
                load_vlm_weights=load_vlm_weights,
                add_image_special_tokens=add_image_special_tokens,
                attention_mode=attention_mode,
                prefix_length=prefix_length,
                num_expert_layers=num_expert_layers,
                num_vlm_layers=num_vlm_layers,
                self_attn_every_n_layers=self_attn_every_n_layers,
                expert_width_multiplier=expert_width_multiplier,
            ),
        )
        object.__setattr__(
            self,
            "flow_matching",
            flow_matching
            or SmolVLAFlowMatchingConfig(
                num_steps=num_steps,
                use_cache=use_cache,
                min_period=min_period,
                max_period=max_period,
                use_random_input_noise=use_random_input_noise,
            ),
        )
        object.__setattr__(
            self,
            "fine_tuning",
            fine_tuning
            or SmolVLAFineTuningConfig(
                freeze_vision_encoder=freeze_vision_encoder,
                train_expert_only=train_expert_only,
                train_state_proj=train_state_proj,
                compile_model=compile_model,
            ),
        )
        object.__setattr__(
            self,
            "optimizer",
            optimizer
            or SmolVLAOptimizerConfig(
                optimizer_lr=optimizer_lr,
                optimizer_betas=optimizer_betas,
                optimizer_eps=optimizer_eps,
                optimizer_weight_decay=optimizer_weight_decay,
                optimizer_grad_clip_norm=optimizer_grad_clip_norm,
                scheduler_warmup_steps=scheduler_warmup_steps,
                scheduler_decay_steps=scheduler_decay_steps,
                scheduler_decay_lr=scheduler_decay_lr,
            ),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SmolVLAConfig:
        """Reconstruct SmolVLAConfig from nested or legacy flat dictionaries."""
        grouped = {"io", "preprocessing", "architecture", "flow_matching", "fine_tuning", "optimizer"}
        flat_keys = {
            "n_obs_steps",
            "chunk_size",
            "n_action_steps",
            "max_state_dim",
            "max_action_dim",
            "resize_imgs_with_padding",
            "empty_cameras",
            "adapt_to_pi_aloha",
            "tokenizer_max_length",
            "pad_language_to",
            "vlm_model_name",
            "load_vlm_weights",
            "add_image_special_tokens",
            "attention_mode",
            "prefix_length",
            "num_expert_layers",
            "num_vlm_layers",
            "self_attn_every_n_layers",
            "expert_width_multiplier",
            "min_period",
            "max_period",
            "use_random_input_noise",
            "num_steps",
            "use_cache",
            "freeze_vision_encoder",
            "train_expert_only",
            "train_state_proj",
            "compile_model",
            "optimizer_lr",
            "optimizer_betas",
            "optimizer_eps",
            "optimizer_weight_decay",
            "optimizer_grad_clip_norm",
            "scheduler_warmup_steps",
            "scheduler_decay_steps",
            "scheduler_decay_lr",
        }
        if grouped.intersection(data):
            data = dict(data)
            for key, config_cls in {
                "io": SmolVLAInputOutputConfig,
                "preprocessing": SmolVLAPreprocessingConfig,
                "architecture": SmolVLAArchitectureConfig,
                "flow_matching": SmolVLAFlowMatchingConfig,
                "fine_tuning": SmolVLAFineTuningConfig,
                "optimizer": SmolVLAOptimizerConfig,
            }.items():
                if isinstance(data.get(key), dict):
                    data[key] = config_cls.from_dict(data[key])
            return cls(**{key: value for key, value in data.items() if key in grouped})
        return cls(**{key: value for key, value in data.items() if key in flat_keys})

    def to_flat_dict(self) -> dict[str, Any]:
        """Return legacy flat SmolVLA config keys used by existing checkpoints and constructors."""
        return {
            "n_obs_steps": self.n_obs_steps,
            "chunk_size": self.chunk_size,
            "n_action_steps": self.n_action_steps,
            "max_state_dim": self.max_state_dim,
            "max_action_dim": self.max_action_dim,
            "resize_imgs_with_padding": self.resize_imgs_with_padding,
            "empty_cameras": self.empty_cameras,
            "adapt_to_pi_aloha": self.adapt_to_pi_aloha,
            "tokenizer_max_length": self.tokenizer_max_length,
            "pad_language_to": self.pad_language_to,
            "vlm_model_name": self.vlm_model_name,
            "load_vlm_weights": self.load_vlm_weights,
            "add_image_special_tokens": self.add_image_special_tokens,
            "attention_mode": self.attention_mode,
            "prefix_length": self.prefix_length,
            "num_expert_layers": self.num_expert_layers,
            "num_vlm_layers": self.num_vlm_layers,
            "self_attn_every_n_layers": self.self_attn_every_n_layers,
            "expert_width_multiplier": self.expert_width_multiplier,
            "min_period": self.min_period,
            "max_period": self.max_period,
            "use_random_input_noise": self.use_random_input_noise,
            "num_steps": self.num_steps,
            "use_cache": self.use_cache,
            "freeze_vision_encoder": self.freeze_vision_encoder,
            "train_expert_only": self.train_expert_only,
            "train_state_proj": self.train_state_proj,
            "compile_model": self.compile_model,
            "optimizer_lr": self.optimizer_lr,
            "optimizer_betas": self.optimizer_betas,
            "optimizer_eps": self.optimizer_eps,
            "optimizer_weight_decay": self.optimizer_weight_decay,
            "optimizer_grad_clip_norm": self.optimizer_grad_clip_norm,
            "scheduler_warmup_steps": self.scheduler_warmup_steps,
            "scheduler_decay_steps": self.scheduler_decay_steps,
            "scheduler_decay_lr": self.scheduler_decay_lr,
        }

    @property
    def n_obs_steps(self) -> int:
        return self.io.n_obs_steps

    @property
    def chunk_size(self) -> int:
        return self.io.chunk_size

    @property
    def n_action_steps(self) -> int:
        return self.io.n_action_steps

    @property
    def max_state_dim(self) -> int:
        return self.io.max_state_dim

    @property
    def max_action_dim(self) -> int:
        return self.io.max_action_dim

    @property
    def resize_imgs_with_padding(self) -> tuple[int, int]:
        return self.preprocessing.resize_imgs_with_padding

    @property
    def empty_cameras(self) -> int:
        return self.preprocessing.empty_cameras

    @property
    def adapt_to_pi_aloha(self) -> bool:
        return self.preprocessing.adapt_to_pi_aloha

    @property
    def tokenizer_max_length(self) -> int:
        return self.preprocessing.tokenizer_max_length

    @property
    def pad_language_to(self) -> str:
        return self.preprocessing.pad_language_to

    @property
    def vlm_model_name(self) -> str:
        return self.architecture.vlm_model_name

    @property
    def load_vlm_weights(self) -> bool:
        return self.architecture.load_vlm_weights

    @property
    def add_image_special_tokens(self) -> bool:
        return self.architecture.add_image_special_tokens

    @property
    def attention_mode(self) -> str:
        return self.architecture.attention_mode

    @property
    def prefix_length(self) -> int:
        return self.architecture.prefix_length

    @property
    def num_expert_layers(self) -> int:
        return self.architecture.num_expert_layers

    @property
    def num_vlm_layers(self) -> int:
        return self.architecture.num_vlm_layers

    @property
    def self_attn_every_n_layers(self) -> int:
        return self.architecture.self_attn_every_n_layers

    @property
    def expert_width_multiplier(self) -> float:
        return self.architecture.expert_width_multiplier

    @property
    def min_period(self) -> float:
        return self.flow_matching.min_period

    @property
    def max_period(self) -> float:
        return self.flow_matching.max_period

    @property
    def use_random_input_noise(self) -> bool:
        return self.flow_matching.use_random_input_noise

    @property
    def num_steps(self) -> int:
        return self.flow_matching.num_steps

    @property
    def use_cache(self) -> bool:
        return self.flow_matching.use_cache

    @property
    def freeze_vision_encoder(self) -> bool:
        return self.fine_tuning.freeze_vision_encoder

    @property
    def train_expert_only(self) -> bool:
        return self.fine_tuning.train_expert_only

    @property
    def train_state_proj(self) -> bool:
        return self.fine_tuning.train_state_proj

    @property
    def compile_model(self) -> bool:
        return self.fine_tuning.compile_model

    @property
    def optimizer_lr(self) -> float:
        return self.optimizer.optimizer_lr

    @property
    def optimizer_betas(self) -> tuple[float, float]:
        return self.optimizer.optimizer_betas

    @property
    def optimizer_eps(self) -> float:
        return self.optimizer.optimizer_eps

    @property
    def optimizer_weight_decay(self) -> float:
        return self.optimizer.optimizer_weight_decay

    @property
    def optimizer_grad_clip_norm(self) -> float:
        return self.optimizer.optimizer_grad_clip_norm

    @property
    def scheduler_warmup_steps(self) -> int:
        return self.optimizer.scheduler_warmup_steps

    @property
    def scheduler_decay_steps(self) -> int:
        return self.optimizer.scheduler_decay_steps

    @property
    def scheduler_decay_lr(self) -> float:
        return self.optimizer.scheduler_decay_lr
