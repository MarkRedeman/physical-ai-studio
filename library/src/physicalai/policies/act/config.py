# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""ACT policy config."""

from __future__ import annotations

from dataclasses import dataclass, field

from physicalai.config import Config
from physicalai.data import Feature  # noqa: TC001 - Needed at runtime for type hint resolution


def _metadata(
    description: str,
    *,
    group: str,
    group_title: str,
    title: str | None = None,
    **validation: object,
) -> dict[str, object]:
    """Build pydantic.Field-compatible metadata plus UI grouping hints."""
    json_schema_extra = {"group": group, "group_title": group_title}
    if isinstance(validation.get("json_schema_extra"), dict):
        json_schema_extra.update(validation.pop("json_schema_extra"))
    metadata = {"description": description, "json_schema_extra": json_schema_extra, **validation}
    if title is not None:
        metadata["title"] = title
    return metadata


@dataclass(frozen=True)
class ACTConfig(Config):
    """Configuration class for the Action Chunking Transformers policy."""

    input_features: dict[str, Feature] = field(default_factory=dict)
    output_features: dict[str, Feature] = field(default_factory=dict)

    n_obs_steps: int = field(
        default=1,
        metadata=_metadata(
            "Number of environment observation steps provided to the policy. ACT currently supports 1.",
            group="io",
            group_title="Input / Output",
            ge=1,
        ),
    )
    chunk_size: int = field(
        default=100,
        metadata=_metadata(
            "Number of future action steps predicted per policy invocation.",
            group="io",
            group_title="Input / Output",
            ge=1,
        ),
    )
    n_action_steps: int = field(
        default=100,
        metadata=_metadata(
            "Number of predicted action steps to execute before querying the policy again. Must be <= chunk_size.",
            group="io",
            group_title="Input / Output",
            ge=1,
        ),
    )

    vision_backbone: str = field(
        default="resnet18",
        metadata=_metadata(
            "Torchvision ResNet backbone used to encode visual observations.",
            group="vision",
            group_title="Vision",
        ),
    )
    pretrained_backbone_weights: str | None = field(
        default="ResNet18_Weights.IMAGENET1K_V1",
        metadata=_metadata(
            "Torchvision pretrained weights identifier. Use null to train the backbone from scratch.",
            group="vision",
            group_title="Vision",
        ),
    )
    replace_final_stride_with_dilation: bool = field(
        default=False,
        metadata=_metadata(
            "Replace the final ResNet 2x2 stride with dilation to preserve spatial resolution.",
            group="vision",
            group_title="Vision",
        ),
    )
    image_size: tuple[int, int] = field(
        default=(512, 512),
        metadata=_metadata(
            "Image preprocessing resolution as (height, width).",
            group="vision",
            group_title="Vision",
            min_length=2,
            max_length=2,
        ),
    )

    pre_norm: bool = field(
        default=False,
        metadata=_metadata(
            "Use pre-normalization in transformer blocks instead of post-normalization.",
            group="transformer",
            group_title="Transformer",
        ),
    )
    dim_model: int = field(
        default=512,
        metadata=_metadata("Main hidden dimension of the transformer blocks.", group="transformer", group_title="Transformer", ge=1),
    )
    n_heads: int = field(
        default=8,
        metadata=_metadata("Number of multi-head attention heads.", group="transformer", group_title="Transformer", ge=1),
    )
    dim_feedforward: int = field(
        default=3200,
        metadata=_metadata("Feed-forward expansion dimension inside transformer blocks.", group="transformer", group_title="Transformer", ge=1),
    )
    feedforward_activation: str = field(
        default="relu",
        metadata=_metadata("Activation used in transformer feed-forward layers.", group="transformer", group_title="Transformer"),
    )
    n_encoder_layers: int = field(
        default=4,
        metadata=_metadata("Number of transformer encoder layers.", group="transformer", group_title="Transformer", ge=1),
    )
    n_decoder_layers: int = field(
        default=1,
        metadata=_metadata(
            "Number of transformer decoder layers. ACT matches the original implementation with a default of 1.",
            group="transformer",
            group_title="Transformer",
            ge=1,
        ),
    )

    use_vae: bool = field(
        default=True,
        metadata=_metadata("Enable the variational training objective.", group="vae", group_title="VAE"),
    )
    latent_dim: int = field(
        default=32,
        metadata=_metadata("Latent dimension of the VAE branch.", group="vae", group_title="VAE", ge=1),
    )
    n_vae_encoder_layers: int = field(
        default=4,
        metadata=_metadata("Number of transformer layers in the VAE encoder.", group="vae", group_title="VAE", ge=1),
    )

    temporal_ensemble_coeff: float | None = field(
        default=None,
        metadata=_metadata(
            "Exponential weighting coefficient for temporal ensembling. Null disables temporal ensembling.",
            group="inference",
            group_title="Inference",
            ge=0.0,
        ),
    )

    dropout: float = field(
        default=0.1,
        metadata=_metadata("Dropout probability used in transformer layers.", group="training", group_title="Training", ge=0.0, le=1.0),
    )
    kl_weight: float = field(
        default=10.0,
        metadata=_metadata("Weight applied to the KL-divergence loss term when VAE is enabled.", group="vae", group_title="VAE", ge=0.0),
    )

    optimizer_lr: float = field(
        default=1e-5,
        metadata=_metadata("Learning rate for the ACT optimizer.", group="optimizer", group_title="Optimizer", gt=0.0),
    )
    optimizer_weight_decay: float = field(
        default=1e-4,
        metadata=_metadata("Weight decay coefficient for the ACT optimizer.", group="optimizer", group_title="Optimizer", ge=0.0),
    )
    optimizer_grad_clip_norm: float = field(
        default=10,
        metadata=_metadata("Maximum gradient norm used for gradient clipping.", group="optimizer", group_title="Optimizer", gt=0.0),
    )

    compile_model: bool = False

    def __post_init__(self) -> None:
        """Validate ACT configuration parameters."""
        if self.n_action_steps > self.chunk_size:
            msg = (
                "The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
            raise ValueError(msg)
        if self.n_obs_steps != 1:
            msg = f"Multiple observation steps not handled yet. Got `nobs_steps={self.n_obs_steps}`"
            raise ValueError(msg)
        if not self.vision_backbone.startswith("resnet"):
            msg = f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            raise ValueError(msg)
        if self.temporal_ensemble_coeff is not None and self.n_action_steps > 1:
            msg = "`n_action_steps` must be 1 when using temporal ensembling."
            raise NotImplementedError(msg)
        if self.dim_model % self.n_heads != 0:
            msg = f"`dim_model` must be divisible by `n_heads`. Got {self.dim_model} and {self.n_heads}."
            raise ValueError(msg)
