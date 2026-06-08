# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Configuration for Pi05 model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from physicalai.config import Config


def _metadata(description: str, **validation: object) -> dict[str, object]:
    """Build metadata that can be passed directly to pydantic.Field.

    Dataclasses keep metadata as a plain mapping, while API adapters can later do
    ``Field(**field.metadata)`` without translating common constraint names.
    """
    return {"description": description, **validation}


@dataclass(frozen=True)
class Pi05BackboneConfig(Config):
    """Backbone model selection and precision settings for Pi05."""

    paligemma_variant: Literal["gemma_300m", "gemma_2b"] = field(
        default="gemma_2b",
        metadata=_metadata(
            "Gemma variant used for the vision-language backbone.",
            json_schema_extra={"enum": ["gemma_300m", "gemma_2b"]},
        ),
    )
    action_expert_variant: Literal["gemma_300m", "gemma_2b"] = field(
        default="gemma_300m",
        metadata=_metadata(
            "Gemma variant used for the action expert.",
            json_schema_extra={"enum": ["gemma_300m", "gemma_2b"]},
        ),
    )
    dtype: Literal["bfloat16", "float32"] = field(
        default="bfloat16",
        metadata=_metadata("Model weight precision.", json_schema_extra={"enum": ["bfloat16", "float32"]}),
    )

    def __post_init__(self) -> None:
        """Validate backbone parameters."""
        if self.paligemma_variant not in {"gemma_300m", "gemma_2b"}:
            msg = f"Invalid paligemma_variant: {self.paligemma_variant}"
            raise ValueError(msg)
        if self.action_expert_variant not in {"gemma_300m", "gemma_2b"}:
            msg = f"Invalid action_expert_variant: {self.action_expert_variant}"
            raise ValueError(msg)
        if self.dtype not in {"bfloat16", "float32"}:
            msg = f"Invalid dtype: {self.dtype}"
            raise ValueError(msg)


@dataclass(frozen=True)
class Pi05InputOutputConfig(Config):
    """Input/output dimensions and action chunking settings for Pi05."""

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
            msg = f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            raise ValueError(msg)
        if self.n_obs_steps <= 0 or self.chunk_size <= 0 or self.n_action_steps <= 0:
            msg = "Observation and action step counts must be positive."
            raise ValueError(msg)
        if self.max_state_dim <= 0 or self.max_action_dim <= 0:
            msg = "State and action dimensions must be positive."
            raise ValueError(msg)


@dataclass(frozen=True)
class Pi05FlowMatchingConfig(Config):
    """Flow matching and denoising parameters for Pi05."""

    num_inference_steps: int = field(default=10, metadata=_metadata("Number of denoising steps for inference.", ge=1))
    time_sampling_beta_alpha: float = field(
        default=1.5,
        metadata=_metadata("Alpha parameter for beta-distribution timestep sampling.", gt=0.0),
    )
    time_sampling_beta_beta: float = field(
        default=1.0,
        metadata=_metadata("Beta parameter for beta-distribution timestep sampling.", gt=0.0),
    )
    time_sampling_scale: float = field(default=0.999, metadata=_metadata("Scale factor for timestep sampling."))
    time_sampling_offset: float = field(default=0.001, metadata=_metadata("Offset applied to timestep sampling."))
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
        if self.num_inference_steps <= 0:
            msg = "`num_inference_steps` must be positive."
            raise ValueError(msg)
        if self.time_sampling_beta_alpha <= 0 or self.time_sampling_beta_beta <= 0:
            msg = "Beta distribution parameters must be positive."
            raise ValueError(msg)
        if self.min_period <= 0 or self.max_period <= 0:
            msg = "Period bounds must be positive."
            raise ValueError(msg)
        if self.min_period > self.max_period:
            msg = "`min_period` cannot be greater than `max_period`."
            raise ValueError(msg)


@dataclass(frozen=True)
class Pi05PreprocessingConfig(Config):
    """Image, camera, tokenization, and normalization settings for Pi05."""

    image_resolution: tuple[int, int] = field(
        default=(224, 224),
        metadata=_metadata("Target image resolution as (height, width).", min_length=2, max_length=2),
    )
    empty_cameras: int = field(default=0, metadata=_metadata("Number of empty camera slots to add.", ge=0))
    tokenizer_max_length: int = field(
        default=200,
        metadata=_metadata("Maximum tokenized language sequence length.", ge=1),
    )
    normalization_mode: Literal["MEAN_STD", "QUANTILES"] = field(
        default="QUANTILES",
        metadata=_metadata(
            "State/action normalization mode. QUANTILES is robust to outliers; MEAN_STD uses mean/std.",
            json_schema_extra={"enum": ["MEAN_STD", "QUANTILES"]},
        ),
    )

    def __post_init__(self) -> None:
        """Validate preprocessing parameters."""
        if len(self.image_resolution) != 2 or any(size <= 0 for size in self.image_resolution):
            msg = f"`image_resolution` must contain two positive integers. Got {self.image_resolution}."
            raise ValueError(msg)
        if self.empty_cameras < 0:
            msg = "`empty_cameras` must be non-negative."
            raise ValueError(msg)
        if self.tokenizer_max_length <= 0:
            msg = "`tokenizer_max_length` must be positive."
            raise ValueError(msg)
        if self.normalization_mode not in {"MEAN_STD", "QUANTILES"}:
            msg = f"Invalid normalization_mode: {self.normalization_mode}"
            raise ValueError(msg)


@dataclass(frozen=True)
class Pi05TrainingConfig(Config):
    """Compilation and fine-tuning behavior for Pi05."""

    gradient_checkpointing: bool = field(
        default=True,
        metadata=_metadata("Enable gradient checkpointing to reduce memory usage."),
    )
    compile_model: bool = field(default=False, metadata=_metadata("Enable torch.compile for the model."))
    compile_mode: str = field(default="max-autotune", metadata=_metadata("Torch compile mode."))
    freeze_vision_encoder: bool = field(
        default=False,
        metadata=_metadata("Freeze the vision encoder during fine-tuning."),
    )
    train_expert_only: bool = field(
        default=False,
        metadata=_metadata("Train only the action expert while freezing other trainable components."),
    )


@dataclass(frozen=True)
class Pi05OptimizerConfig(Config):
    """Optimizer and scheduler settings for Pi05."""

    optimizer_lr: float = field(default=2.5e-5, metadata=_metadata("Learning rate.", gt=0.0))
    optimizer_betas: tuple[float, float] = field(
        default=(0.9, 0.95),
        metadata=_metadata("Adam beta coefficients as (beta1, beta2).", min_length=2, max_length=2),
    )
    optimizer_eps: float = field(default=1e-8, metadata=_metadata("Optimizer epsilon.", gt=0.0))
    optimizer_weight_decay: float = field(default=0.01, metadata=_metadata("Weight decay coefficient.", ge=0.0))
    optimizer_grad_clip_norm: float = field(default=1.0, metadata=_metadata("Maximum gradient norm.", gt=0.0))
    scheduler_warmup_steps: int = field(default=1_000, metadata=_metadata("Linear warmup steps.", ge=0))
    scheduler_decay_steps: int | None = field(
        default=30_000,
        metadata=_metadata("Cosine decay horizon in steps. Null auto-scales to total training steps.", ge=1),
    )
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
        if self.scheduler_decay_steps is not None and self.scheduler_decay_steps <= 0:
            msg = "`scheduler_decay_steps` must be positive when provided."
            raise ValueError(msg)
        if self.scheduler_decay_lr < 0:
            msg = "`scheduler_decay_lr` must be non-negative."
            raise ValueError(msg)


@dataclass(frozen=True, init=False)
class Pi05Config(Config):
    """Configuration for Pi05 flow matching model.

    The preferred shape groups related settings into nested config objects. The constructor also accepts legacy flat
    Pi05 keyword arguments for checkpoint compatibility, HuggingFace config loading, and existing callers.
    """

    backbone: Pi05BackboneConfig = field(
        default_factory=Pi05BackboneConfig,
        metadata=_metadata("Backbone model selection and precision settings."),
    )
    io: Pi05InputOutputConfig = field(
        default_factory=Pi05InputOutputConfig,
        metadata=_metadata("Input/output dimensions and action chunking settings."),
    )
    flow_matching: Pi05FlowMatchingConfig = field(
        default_factory=Pi05FlowMatchingConfig,
        metadata=_metadata("Flow matching and denoising parameters."),
    )
    preprocessing: Pi05PreprocessingConfig = field(
        default_factory=Pi05PreprocessingConfig,
        metadata=_metadata("Image, camera, tokenization, and normalization settings."),
    )
    training: Pi05TrainingConfig = field(
        default_factory=Pi05TrainingConfig,
        metadata=_metadata("Compilation and fine-tuning behavior."),
    )
    optimizer: Pi05OptimizerConfig = field(
        default_factory=Pi05OptimizerConfig,
        metadata=_metadata("Optimizer and scheduler settings."),
    )

    def __init__(  # noqa: PLR0913
        self,
        *,
        backbone: Pi05BackboneConfig | None = None,
        io: Pi05InputOutputConfig | None = None,
        flow_matching: Pi05FlowMatchingConfig | None = None,
        preprocessing: Pi05PreprocessingConfig | None = None,
        training: Pi05TrainingConfig | None = None,
        optimizer: Pi05OptimizerConfig | None = None,
        paligemma_variant: Literal["gemma_300m", "gemma_2b"] = "gemma_2b",
        action_expert_variant: Literal["gemma_300m", "gemma_2b"] = "gemma_300m",
        dtype: Literal["bfloat16", "float32"] = "bfloat16",
        n_obs_steps: int = 1,
        chunk_size: int = 50,
        n_action_steps: int = 50,
        max_state_dim: int = 32,
        max_action_dim: int = 32,
        num_inference_steps: int = 10,
        time_sampling_beta_alpha: float = 1.5,
        time_sampling_beta_beta: float = 1.0,
        time_sampling_scale: float = 0.999,
        time_sampling_offset: float = 0.001,
        min_period: float = 4e-3,
        max_period: float = 4.0,
        use_random_input_noise: bool = True,
        image_resolution: tuple[int, int] = (224, 224),
        empty_cameras: int = 0,
        tokenizer_max_length: int = 200,
        gradient_checkpointing: bool = True,
        compile_model: bool = False,
        compile_mode: str = "max-autotune",
        freeze_vision_encoder: bool = False,
        train_expert_only: bool = False,
        normalization_mode: Literal["MEAN_STD", "QUANTILES"] = "QUANTILES",
        optimizer_lr: float = 2.5e-5,
        optimizer_betas: tuple[float, float] = (0.9, 0.95),
        optimizer_eps: float = 1e-8,
        optimizer_weight_decay: float = 0.01,
        optimizer_grad_clip_norm: float = 1.0,
        scheduler_warmup_steps: int = 1_000,
        scheduler_decay_steps: int | None = 30_000,
        scheduler_decay_lr: float = 2.5e-6,
    ) -> None:
        """Initialize Pi05 config from grouped configs or legacy flat keyword arguments."""
        object.__setattr__(
            self,
            "backbone",
            backbone
            or Pi05BackboneConfig(
                paligemma_variant=paligemma_variant,
                action_expert_variant=action_expert_variant,
                dtype=dtype,
            ),
        )
        object.__setattr__(
            self,
            "io",
            io
            or Pi05InputOutputConfig(
                n_obs_steps=n_obs_steps,
                chunk_size=chunk_size,
                n_action_steps=n_action_steps,
                max_state_dim=max_state_dim,
                max_action_dim=max_action_dim,
            ),
        )
        object.__setattr__(
            self,
            "flow_matching",
            flow_matching
            or Pi05FlowMatchingConfig(
                num_inference_steps=num_inference_steps,
                time_sampling_beta_alpha=time_sampling_beta_alpha,
                time_sampling_beta_beta=time_sampling_beta_beta,
                time_sampling_scale=time_sampling_scale,
                time_sampling_offset=time_sampling_offset,
                min_period=min_period,
                max_period=max_period,
                use_random_input_noise=use_random_input_noise,
            ),
        )
        object.__setattr__(
            self,
            "preprocessing",
            preprocessing
            or Pi05PreprocessingConfig(
                image_resolution=image_resolution,
                empty_cameras=empty_cameras,
                tokenizer_max_length=tokenizer_max_length,
                normalization_mode=normalization_mode,
            ),
        )
        object.__setattr__(
            self,
            "training",
            training
            or Pi05TrainingConfig(
                gradient_checkpointing=gradient_checkpointing,
                compile_model=compile_model,
                compile_mode=compile_mode,
                freeze_vision_encoder=freeze_vision_encoder,
                train_expert_only=train_expert_only,
            ),
        )
        object.__setattr__(
            self,
            "optimizer",
            optimizer
            or Pi05OptimizerConfig(
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
    def from_dict(cls, data: dict[str, Any]) -> Pi05Config:
        """Reconstruct Pi05Config from nested or legacy flat dictionaries."""
        grouped = {"backbone", "io", "flow_matching", "preprocessing", "training", "optimizer"}
        flat_keys = {
            "paligemma_variant",
            "action_expert_variant",
            "dtype",
            "n_obs_steps",
            "chunk_size",
            "n_action_steps",
            "max_state_dim",
            "max_action_dim",
            "num_inference_steps",
            "time_sampling_beta_alpha",
            "time_sampling_beta_beta",
            "time_sampling_scale",
            "time_sampling_offset",
            "min_period",
            "max_period",
            "use_random_input_noise",
            "image_resolution",
            "empty_cameras",
            "tokenizer_max_length",
            "gradient_checkpointing",
            "compile_model",
            "compile_mode",
            "freeze_vision_encoder",
            "train_expert_only",
            "normalization_mode",
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
                "backbone": Pi05BackboneConfig,
                "io": Pi05InputOutputConfig,
                "flow_matching": Pi05FlowMatchingConfig,
                "preprocessing": Pi05PreprocessingConfig,
                "training": Pi05TrainingConfig,
                "optimizer": Pi05OptimizerConfig,
            }.items():
                if isinstance(data.get(key), dict):
                    data[key] = config_cls.from_dict(data[key])
            return cls(**{key: value for key, value in data.items() if key in grouped})
        return cls(**{key: value for key, value in data.items() if key in flat_keys})

    def to_flat_dict(self) -> dict[str, Any]:
        """Return legacy flat Pi05 config keys used by existing checkpoints and constructors."""
        return {
            "paligemma_variant": self.paligemma_variant,
            "action_expert_variant": self.action_expert_variant,
            "dtype": self.dtype,
            "n_obs_steps": self.n_obs_steps,
            "chunk_size": self.chunk_size,
            "n_action_steps": self.n_action_steps,
            "max_state_dim": self.max_state_dim,
            "max_action_dim": self.max_action_dim,
            "num_inference_steps": self.num_inference_steps,
            "time_sampling_beta_alpha": self.time_sampling_beta_alpha,
            "time_sampling_beta_beta": self.time_sampling_beta_beta,
            "time_sampling_scale": self.time_sampling_scale,
            "time_sampling_offset": self.time_sampling_offset,
            "min_period": self.min_period,
            "max_period": self.max_period,
            "use_random_input_noise": self.use_random_input_noise,
            "image_resolution": self.image_resolution,
            "empty_cameras": self.empty_cameras,
            "tokenizer_max_length": self.tokenizer_max_length,
            "gradient_checkpointing": self.gradient_checkpointing,
            "compile_model": self.compile_model,
            "compile_mode": self.compile_mode,
            "freeze_vision_encoder": self.freeze_vision_encoder,
            "train_expert_only": self.train_expert_only,
            "normalization_mode": self.normalization_mode,
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
    def paligemma_variant(self) -> Literal["gemma_300m", "gemma_2b"]:
        return self.backbone.paligemma_variant

    @property
    def action_expert_variant(self) -> Literal["gemma_300m", "gemma_2b"]:
        return self.backbone.action_expert_variant

    @property
    def dtype(self) -> Literal["bfloat16", "float32"]:
        return self.backbone.dtype

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
    def num_inference_steps(self) -> int:
        return self.flow_matching.num_inference_steps

    @property
    def time_sampling_beta_alpha(self) -> float:
        return self.flow_matching.time_sampling_beta_alpha

    @property
    def time_sampling_beta_beta(self) -> float:
        return self.flow_matching.time_sampling_beta_beta

    @property
    def time_sampling_scale(self) -> float:
        return self.flow_matching.time_sampling_scale

    @property
    def time_sampling_offset(self) -> float:
        return self.flow_matching.time_sampling_offset

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
    def image_resolution(self) -> tuple[int, int]:
        return self.preprocessing.image_resolution

    @property
    def empty_cameras(self) -> int:
        return self.preprocessing.empty_cameras

    @property
    def tokenizer_max_length(self) -> int:
        return self.preprocessing.tokenizer_max_length

    @property
    def normalization_mode(self) -> Literal["MEAN_STD", "QUANTILES"]:
        return self.preprocessing.normalization_mode

    @property
    def gradient_checkpointing(self) -> bool:
        return self.training.gradient_checkpointing

    @property
    def compile_model(self) -> bool:
        return self.training.compile_model

    @property
    def compile_mode(self) -> str:
        return self.training.compile_mode

    @property
    def freeze_vision_encoder(self) -> bool:
        return self.training.freeze_vision_encoder

    @property
    def train_expert_only(self) -> bool:
        return self.training.train_expert_only

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
    def scheduler_decay_steps(self) -> int | None:
        return self.optimizer.scheduler_decay_steps

    @property
    def scheduler_decay_lr(self) -> float:
        return self.optimizer.scheduler_decay_lr
