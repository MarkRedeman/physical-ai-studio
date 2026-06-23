# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""ACT policy config."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from physicalai.config import Config
from physicalai.data import Feature  # noqa: TC001 - Needed at runtime for type hint resolution


def _metadata(description: str, *, title: str | None = None, **validation: object) -> dict[str, object]:
    """Build metadata that can be passed directly to pydantic.Field.

    Dataclasses keep metadata as a plain mapping, while API adapters can later do
    ``Field(**field.metadata)`` without translating common constraint names.
    """
    metadata = {"description": description, **validation}
    if title is not None:
        metadata["title"] = title
    return metadata


@dataclass(frozen=True)
class ACTInputOutputConfig(Config):
    """Input/output structure for ACT.

    Feature dictionaries are usually inferred from dataset statistics by the policy during training. They are part of
    the model config for checkpoint restoration and lower-level model construction.
    """

    input_features: dict[str, Feature] = field(
        default_factory=dict,
        metadata=_metadata(
            "Input feature definitions keyed by feature name. Usually inferred from the dataset for training.",
        ),
    )
    output_features: dict[str, Feature] = field(
        default_factory=dict,
        metadata=_metadata(
            "Output feature definitions keyed by feature name. Usually inferred from the dataset for training.",
        ),
    )
    n_obs_steps: int = field(
        default=1,
        metadata=_metadata(
            "Number of environment observation steps provided to the policy. ACT currently supports 1.",
            ge=1,
        ),
    )
    chunk_size: int = field(
        default=100,
        metadata=_metadata("Number of future action steps predicted per policy invocation.", ge=1),
    )
    n_action_steps: int = field(
        default=100,
        metadata=_metadata(
            "Number of predicted action steps to execute before querying the policy again. Must be <= chunk_size.",
            ge=1,
        ),
    )

    def __post_init__(self) -> None:
        """Validate input/output timing parameters."""
        if self.n_action_steps > self.chunk_size:
            msg = (
                "The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
            raise ValueError(msg)
        if self.n_obs_steps != 1:
            msg = f"Multiple observation steps not handled yet. Got `nobs_steps={self.n_obs_steps}`"
            raise ValueError(msg)


@dataclass(frozen=True)
class ACTVisionBackboneConfig(Config):
    """Vision backbone settings for ACT image encoding."""

    vision_backbone: str = field(
        default="resnet18",
        metadata=_metadata("Torchvision ResNet backbone used to encode visual observations."),
    )
    pretrained_backbone_weights: str | None = field(
        default="ResNet18_Weights.IMAGENET1K_V1",
        metadata=_metadata("Torchvision pretrained weights identifier. Use null to train the backbone from scratch."),
    )
    replace_final_stride_with_dilation: bool = field(
        default=False,
        metadata=_metadata("Replace the final ResNet 2x2 stride with dilation to preserve spatial resolution."),
    )
    image_size: tuple[int, int] = field(
        default=(512, 512),
        metadata=_metadata("Image preprocessing resolution as (height, width).", min_length=2, max_length=2),
    )

    def __post_init__(self) -> None:
        """Validate vision backbone parameters."""
        if not self.vision_backbone.startswith("resnet"):
            msg = f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            raise ValueError(msg)
        if len(self.image_size) != 2 or any(size <= 0 for size in self.image_size):
            msg = f"`image_size` must contain two positive integers. Got {self.image_size}."
            raise ValueError(msg)


@dataclass(frozen=True)
class ACTTransformerConfig(Config):
    """Transformer architecture settings for ACT."""

    pre_norm: bool = field(
        default=False,
        metadata=_metadata("Use pre-normalization in transformer blocks instead of post-normalization."),
    )
    dim_model: int = field(
        default=512,
        metadata=_metadata("Main hidden dimension of the transformer blocks.", ge=1),
    )
    n_heads: int = field(
        default=8,
        metadata=_metadata("Number of multi-head attention heads.", ge=1),
    )
    dim_feedforward: int = field(
        default=3200,
        metadata=_metadata("Feed-forward expansion dimension inside transformer blocks.", ge=1),
    )
    feedforward_activation: str = field(
        default="relu",
        metadata=_metadata("Activation used in transformer feed-forward layers."),
    )
    n_encoder_layers: int = field(
        default=4,
        metadata=_metadata("Number of transformer encoder layers.", ge=1),
    )
    n_decoder_layers: int = field(
        default=1,
        metadata=_metadata(
            "Number of transformer decoder layers. ACT matches the original implementation with a default of 1.",
            ge=1,
        ),
    )
    dropout: float = field(
        default=0.1,
        metadata=_metadata("Dropout probability used in transformer layers.", ge=0.0, le=1.0),
    )

    def __post_init__(self) -> None:
        """Validate transformer architecture parameters."""
        if self.dim_model <= 0:
            msg = "`dim_model` must be positive."
            raise ValueError(msg)
        if self.n_heads <= 0:
            msg = "`n_heads` must be positive."
            raise ValueError(msg)
        if self.dim_model % self.n_heads != 0:
            msg = f"`dim_model` must be divisible by `n_heads`. Got {self.dim_model} and {self.n_heads}."
            raise ValueError(msg)
        if self.dim_feedforward <= 0:
            msg = "`dim_feedforward` must be positive."
            raise ValueError(msg)
        if self.n_encoder_layers <= 0 or self.n_decoder_layers <= 0:
            msg = "Transformer layer counts must be positive."
            raise ValueError(msg)
        if not 0 <= self.dropout <= 1:
            msg = f"`dropout` must be between 0 and 1. Got {self.dropout}."
            raise ValueError(msg)


@dataclass(frozen=True)
class ACTVAEConfig(Config):
    """Variational objective settings for ACT."""

    use_vae: bool = field(
        default=True,
        metadata=_metadata("Enable the variational training objective."),
    )
    latent_dim: int = field(
        default=32,
        metadata=_metadata("Latent dimension of the VAE branch.", ge=1),
    )
    n_vae_encoder_layers: int = field(
        default=4,
        metadata=_metadata("Number of transformer layers in the VAE encoder.", ge=1),
    )
    kl_weight: float = field(
        default=10.0,
        metadata=_metadata("Weight applied to the KL-divergence loss term when VAE is enabled.", ge=0.0),
    )

    def __post_init__(self) -> None:
        """Validate VAE parameters."""
        if self.latent_dim <= 0:
            msg = "`latent_dim` must be positive."
            raise ValueError(msg)
        if self.n_vae_encoder_layers <= 0:
            msg = "`n_vae_encoder_layers` must be positive."
            raise ValueError(msg)
        if self.kl_weight < 0:
            msg = "`kl_weight` must be non-negative."
            raise ValueError(msg)


@dataclass(frozen=True)
class ACTInferenceConfig(Config):
    """Inference-time ACT behavior."""

    temporal_ensemble_coeff: float | None = field(
        default=None,
        metadata=_metadata(
            "Exponential weighting coefficient for temporal ensembling. Null disables temporal ensembling.",
            ge=0.0,
        ),
    )

    def __post_init__(self) -> None:
        """Validate inference parameters."""
        if self.temporal_ensemble_coeff is not None and self.temporal_ensemble_coeff < 0:
            msg = "`temporal_ensemble_coeff` must be non-negative when provided."
            raise ValueError(msg)


@dataclass(frozen=True)
class ACTOptimizerConfig(Config):
    """Optimizer settings for ACT training."""

    optimizer_lr: float = field(
        default=1e-5,
        metadata=_metadata("Learning rate for the ACT optimizer.", gt=0.0),
    )
    optimizer_weight_decay: float = field(
        default=1e-4,
        metadata=_metadata("Weight decay coefficient for the ACT optimizer.", ge=0.0),
    )
    optimizer_grad_clip_norm: float = field(
        default=10,
        metadata=_metadata("Maximum gradient norm used for gradient clipping.", gt=0.0),
    )

    def __post_init__(self) -> None:
        """Validate optimizer parameters."""
        if self.optimizer_lr <= 0:
            msg = "`optimizer_lr` must be positive."
            raise ValueError(msg)
        if self.optimizer_weight_decay < 0:
            msg = "`optimizer_weight_decay` must be non-negative."
            raise ValueError(msg)
        if self.optimizer_grad_clip_norm <= 0:
            msg = "`optimizer_grad_clip_norm` must be positive."
            raise ValueError(msg)


@dataclass(frozen=True, init=False)
class ACTConfig(Config):
    """Configuration class for the Action Chunking Transformers policy.

    The preferred shape groups related settings into nested config objects. The constructor also accepts the legacy
    flat ACT keyword arguments for checkpoint compatibility and existing callers.
    """

    io: ACTInputOutputConfig = field(
        default_factory=ACTInputOutputConfig,
        metadata=_metadata("Input/output and action chunking settings."),
    )
    vision: ACTVisionBackboneConfig = field(
        default_factory=ACTVisionBackboneConfig,
        metadata=_metadata("Vision backbone and image preprocessing settings."),
    )
    transformer: ACTTransformerConfig = field(
        default_factory=ACTTransformerConfig,
        metadata=_metadata("Transformer architecture settings."),
    )
    vae: ACTVAEConfig = field(
        default_factory=ACTVAEConfig,
        metadata=_metadata("Variational training objective settings."),
    )
    inference: ACTInferenceConfig = field(
        default_factory=ACTInferenceConfig,
        metadata=_metadata("Inference-time ACT behavior settings."),
    )
    optimizer: ACTOptimizerConfig = field(
        default_factory=ACTOptimizerConfig,
        metadata=_metadata("Optimizer and gradient clipping settings."),
    )
    compile_model: bool = field(
        default=False,
        metadata=_metadata("Enable torch.compile for the ACT model."),
    )

    def __init__(  # noqa: PLR0913
        self,
        *,
        io: ACTInputOutputConfig | None = None,
        vision: ACTVisionBackboneConfig | None = None,
        transformer: ACTTransformerConfig | None = None,
        vae: ACTVAEConfig | None = None,
        inference: ACTInferenceConfig | None = None,
        optimizer: ACTOptimizerConfig | None = None,
        compile_model: bool = False,
        input_features: dict[str, Feature] | None = None,
        output_features: dict[str, Feature] | None = None,
        n_obs_steps: int = 1,
        chunk_size: int = 100,
        n_action_steps: int = 100,
        vision_backbone: str = "resnet18",
        pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1",
        replace_final_stride_with_dilation: bool = False,
        image_size: tuple[int, int] = (512, 512),
        pre_norm: bool = False,
        dim_model: int = 512,
        n_heads: int = 8,
        dim_feedforward: int = 3200,
        feedforward_activation: str = "relu",
        n_encoder_layers: int = 4,
        n_decoder_layers: int = 1,
        use_vae: bool = True,
        latent_dim: int = 32,
        n_vae_encoder_layers: int = 4,
        temporal_ensemble_coeff: float | None = None,
        dropout: float = 0.1,
        kl_weight: float = 10.0,
        optimizer_lr: float = 1e-5,
        optimizer_weight_decay: float = 1e-4,
        optimizer_grad_clip_norm: float = 10,
    ) -> None:
        """Initialize ACT config from grouped configs or legacy flat keyword arguments."""
        object.__setattr__(
            self,
            "io",
            io
            or ACTInputOutputConfig(
                input_features=input_features or {},
                output_features=output_features or {},
                n_obs_steps=n_obs_steps,
                chunk_size=chunk_size,
                n_action_steps=n_action_steps,
            ),
        )
        object.__setattr__(
            self,
            "vision",
            vision
            or ACTVisionBackboneConfig(
                vision_backbone=vision_backbone,
                pretrained_backbone_weights=pretrained_backbone_weights,
                replace_final_stride_with_dilation=replace_final_stride_with_dilation,
                image_size=image_size,
            ),
        )
        object.__setattr__(
            self,
            "transformer",
            transformer
            or ACTTransformerConfig(
                pre_norm=pre_norm,
                dim_model=dim_model,
                n_heads=n_heads,
                dim_feedforward=dim_feedforward,
                feedforward_activation=feedforward_activation,
                n_encoder_layers=n_encoder_layers,
                n_decoder_layers=n_decoder_layers,
                dropout=dropout,
            ),
        )
        object.__setattr__(
            self,
            "vae",
            vae
            or ACTVAEConfig(
                use_vae=use_vae,
                latent_dim=latent_dim,
                n_vae_encoder_layers=n_vae_encoder_layers,
                kl_weight=kl_weight,
            ),
        )
        object.__setattr__(
            self,
            "inference",
            inference or ACTInferenceConfig(temporal_ensemble_coeff=temporal_ensemble_coeff),
        )
        object.__setattr__(
            self,
            "optimizer",
            optimizer
            or ACTOptimizerConfig(
                optimizer_lr=optimizer_lr,
                optimizer_weight_decay=optimizer_weight_decay,
                optimizer_grad_clip_norm=optimizer_grad_clip_norm,
            ),
        )
        object.__setattr__(self, "compile_model", compile_model)
        ACTConfig.__post_init__(self)

    def __post_init__(self) -> None:
        """Validate cross-group ACT settings."""
        if self.inference.temporal_ensemble_coeff is not None and self.io.n_action_steps > 1:
            msg = (
                "`n_action_steps` must be 1 when using temporal ensembling. "
                "This is because the policy needs to be queried every step to compute the ensembled action."
            )
            raise NotImplementedError(msg)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ACTConfig:
        """Reconstruct ACTConfig from nested or legacy flat dictionaries."""
        grouped_keys = {"io", "vision", "transformer", "vae", "inference", "optimizer"}
        if grouped_keys.intersection(data):
            data = dict(data)
            for key, config_cls in {
                "io": ACTInputOutputConfig,
                "vision": ACTVisionBackboneConfig,
                "transformer": ACTTransformerConfig,
                "vae": ACTVAEConfig,
                "inference": ACTInferenceConfig,
                "optimizer": ACTOptimizerConfig,
            }.items():
                if isinstance(data.get(key), dict):
                    data[key] = config_cls.from_dict(data[key])
            return cls(**data)
        return cls(**data)

    def to_flat_dict(self) -> dict[str, Any]:
        """Return legacy flat ACT config keys used by existing checkpoints and constructors."""
        return {
            "input_features": self.input_features,
            "output_features": self.output_features,
            "n_obs_steps": self.n_obs_steps,
            "chunk_size": self.chunk_size,
            "n_action_steps": self.n_action_steps,
            "vision_backbone": self.vision_backbone,
            "pretrained_backbone_weights": self.pretrained_backbone_weights,
            "replace_final_stride_with_dilation": self.replace_final_stride_with_dilation,
            "image_size": self.image_size,
            "pre_norm": self.pre_norm,
            "dim_model": self.dim_model,
            "n_heads": self.n_heads,
            "dim_feedforward": self.dim_feedforward,
            "feedforward_activation": self.feedforward_activation,
            "n_encoder_layers": self.n_encoder_layers,
            "n_decoder_layers": self.n_decoder_layers,
            "use_vae": self.use_vae,
            "latent_dim": self.latent_dim,
            "n_vae_encoder_layers": self.n_vae_encoder_layers,
            "temporal_ensemble_coeff": self.temporal_ensemble_coeff,
            "dropout": self.dropout,
            "kl_weight": self.kl_weight,
            "optimizer_lr": self.optimizer_lr,
            "optimizer_weight_decay": self.optimizer_weight_decay,
            "optimizer_grad_clip_norm": self.optimizer_grad_clip_norm,
            "compile_model": self.compile_model,
        }

    @property
    def input_features(self) -> dict[str, Feature]:
        return self.io.input_features

    @property
    def output_features(self) -> dict[str, Feature]:
        return self.io.output_features

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
    def vision_backbone(self) -> str:
        return self.vision.vision_backbone

    @property
    def pretrained_backbone_weights(self) -> str | None:
        return self.vision.pretrained_backbone_weights

    @property
    def replace_final_stride_with_dilation(self) -> bool:
        return self.vision.replace_final_stride_with_dilation

    @property
    def image_size(self) -> tuple[int, int]:
        return self.vision.image_size

    @property
    def pre_norm(self) -> bool:
        return self.transformer.pre_norm

    @property
    def dim_model(self) -> int:
        return self.transformer.dim_model

    @property
    def n_heads(self) -> int:
        return self.transformer.n_heads

    @property
    def dim_feedforward(self) -> int:
        return self.transformer.dim_feedforward

    @property
    def feedforward_activation(self) -> str:
        return self.transformer.feedforward_activation

    @property
    def n_encoder_layers(self) -> int:
        return self.transformer.n_encoder_layers

    @property
    def n_decoder_layers(self) -> int:
        return self.transformer.n_decoder_layers

    @property
    def dropout(self) -> float:
        return self.transformer.dropout

    @property
    def use_vae(self) -> bool:
        return self.vae.use_vae

    @property
    def latent_dim(self) -> int:
        return self.vae.latent_dim

    @property
    def n_vae_encoder_layers(self) -> int:
        return self.vae.n_vae_encoder_layers

    @property
    def kl_weight(self) -> float:
        return self.vae.kl_weight

    @property
    def temporal_ensemble_coeff(self) -> float | None:
        return self.inference.temporal_ensemble_coeff

    @property
    def optimizer_lr(self) -> float:
        return self.optimizer.optimizer_lr

    @property
    def optimizer_weight_decay(self) -> float:
        return self.optimizer.optimizer_weight_decay

    @property
    def optimizer_grad_clip_norm(self) -> float:
        return self.optimizer.optimizer_grad_clip_norm
