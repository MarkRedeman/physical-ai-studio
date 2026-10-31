"""Discover trainable policies in the process that will execute the training job."""

from collections.abc import Callable
from importlib import metadata
from typing import Annotated, Literal

from loguru import logger
from packaging.requirements import Requirement
from pydantic import BaseModel, ConfigDict, Field


class PolicyTrainingDefaults(BaseModel):
    """Recommended values expressed in Studio training-payload field names."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    batch_size: int = Field(default=8, ge=1, le=256, description="Samples per training step.")
    max_epochs: int = Field(default=5, ge=1, le=10_000, description="Full passes through the training dataset.")
    num_workers: Annotated[int, Field(ge=0, le=16)] | Literal["auto"] = Field(
        default="auto", description="DataLoader workers (0-16); auto uses available CPUs."
    )
    precision: Literal["32-true", "bf16-mixed", "bf16-true"] = Field(
        default="bf16-mixed", description="Lightning numerical precision mode."
    )
    compile_model: bool = Field(default=False, description="Enable torch.compile for the policy.")
    gradient_checkpointing: bool | None = Field(
        default=None, description="Recompute activations to save memory; null keeps the library default."
    )
    auto_scale_batch_size: bool = Field(default=False, description="Find a fitting batch size before training.")
    augment_images: bool = Field(default=False, description="Apply Studio's default training-only image augmentations.")
    val_split: float = Field(default=0.1, ge=0, lt=1, description="Fraction of episodes held out for validation.")
    val_split_seed: int | None = Field(default=None, description="Validation split RNG seed; null uses the global RNG.")
    lora_enabled: bool = Field(default=False, description="Fine-tune low-rank adapters rather than all model weights.")
    lora_rank: int = Field(default=32, ge=1, le=256, description="LoRA adapter rank.")
    lora_alpha: int | None = Field(default=None, ge=1, description="LoRA scaling numerator; null uses the rank.")
    lora_dropout: float = Field(default=0.05, ge=0, lt=1, description="LoRA adapter input dropout probability.")
    lora_use_dora: bool = Field(default=False, description="Use DoRA instead of plain LoRA; requires LoRA enabled.")
    snapflow_enabled: bool = Field(default=False, description="Append SnapFlow distillation after ordinary training.")
    snapflow_distill_epochs: int = Field(
        default=3, ge=1, le=10_000, description="Additional distillation epochs, not included in max_epochs."
    )


class PolicyTrainingHelp(BaseModel):
    """Optional policy guidance; absent fields use the UI's generic help."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    batch_size: str | None = Field(default=None, description="Batch size recommendation and memory tradeoffs.")
    max_epochs: str | None = Field(default=None, description="Training budget recommendation and dataset assumptions.")
    num_workers: str | None = Field(default=None, description="Data loading worker recommendation.")
    precision: str | None = Field(default=None, description="Policy-specific numerical precision guidance.")
    compile_model: str | None = Field(default=None, description="Compilation speed and startup tradeoffs.")
    gradient_checkpointing: str | None = Field(default=None, description="Activation memory and compute tradeoffs.")
    auto_scale_batch_size: str | None = Field(default=None, description="Automatic batch size guidance.")
    augment_images: str | None = Field(default=None, description="Image augmentation recommendation.")
    lora_enabled: str | None = Field(default=None, description="Adapter fine-tuning recommendation.")
    lora_rank: str | None = Field(default=None, description="Adapter capacity recommendation.")
    lora_alpha: str | None = Field(default=None, description="Adapter scaling guidance.")
    lora_dropout: str | None = Field(default=None, description="Adapter dropout guidance.")
    lora_use_dora: str | None = Field(default=None, description="DoRA guidance.")
    snapflow_enabled: str | None = Field(default=None, description="Distillation recommendation.")
    snapflow_distill_epochs: str | None = Field(default=None, description="Distillation budget recommendation.")


class PolicyCameraSlot(BaseModel):
    """One camera slot in the pretrained policy's fixed camera order."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(description="Stable pretrained camera slot identifier.")
    name: str = Field(description="Display name describing the camera view.")
    description: str = Field(description="Guidance for assigning a dataset camera to this slot.")
    is_required: bool = Field(description="Whether a camera must fill this slot for sensible training.")


class PolicyTrainingConfiguration(BaseModel):
    """Recommended training values with optional guidance and camera metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    defaults: PolicyTrainingDefaults = Field(description="Editable defaults to include in the training job payload.")
    help: PolicyTrainingHelp = Field(default_factory=PolicyTrainingHelp, description="Optional policy-specific help.")
    camera_slots: list[PolicyCameraSlot] = Field(
        default_factory=list, description="Pretrained camera slots in order; empty means dataset order is used."
    )


POLICY_TRAINING_CONFIGURATIONS = {
    "act": PolicyTrainingConfiguration(
        defaults=PolicyTrainingDefaults(batch_size=32, precision="32-true", augment_images=True),
        help=PolicyTrainingHelp(
            batch_size="The PushT reference recipe uses 32 samples per step. Reduce this if memory is insufficient.",
            max_epochs=(
                "Five epochs is a starting point, not a validated ACT budget. The PushT reference recipe uses "
                "70,000 steps; the equivalent epoch count depends on dataset size and batch size."
            ),
            precision="The ACT PushT recipes use full FP32 precision, including on Intel XPU.",
            augment_images="The reference uses ColorJitter. Studio enables its broader default augmentation pipeline.",
        ),
    ),
    "smolvla": PolicyTrainingConfiguration(
        defaults=PolicyTrainingDefaults(batch_size=64, max_epochs=100, precision="32-true", augment_images=True),
        help=PolicyTrainingHelp(
            batch_size="The PushT reference uses batch size 64. Reduce this if GPU memory is insufficient.",
            max_epochs="The PushT reference recipe trains for 100 epochs. Review validation loss for your own task.",
        ),
        camera_slots=[
            PolicyCameraSlot(
                id="camera1",
                name="Overview",
                is_required=True,
                description=(
                    "The scene view: a fixed camera that sees the workspace and the robot in it. "
                    "SmolVLA is pretrained with this view in its first camera slot, so mapping a wrist "
                    "camera here means the policy sees something quite different from what it was trained on."
                ),
            ),
            PolicyCameraSlot(
                id="camera2",
                name="Gripper",
                is_required=False,
                description=(
                    "The wrist or gripper view: a camera mounted on the arm that sees what the gripper is "
                    "about to grasp. Leave it empty when the robot has no wrist camera — the slot is then "
                    "filled with a masked empty image."
                ),
            ),
            PolicyCameraSlot(
                id="camera3",
                name="Extra",
                is_required=False,
                description=(
                    "A third view — a second scene angle or a second wrist camera. Leave it empty when the "
                    "setup has only two cameras."
                ),
            ),
        ],
    ),
    "pi05": PolicyTrainingConfiguration(
        defaults=PolicyTrainingDefaults(
            batch_size=16,
            max_epochs=25,
            compile_model=True,
            gradient_checkpointing=False,
            lora_enabled=True,
            lora_alpha=32,
            val_split_seed=42,
        ),
        help=PolicyTrainingHelp(
            batch_size="The ALOHA LoRA reference uses batch size 16. Reduce this if GPU memory is insufficient.",
            max_epochs="The ALOHA LoRA reference uses 25 epochs. Review validation loss for your task.",
            gradient_checkpointing="The LoRA recipe disables checkpointing. Enable it to save memory with more compute.",
            lora_enabled="The reference uses LoRA. The library automatically applies a 10x learning-rate multiplier.",
            lora_rank="Start at 32; try 16 if a small dataset overfits, or 64 for more capacity with plenty of data.",
        ),
    ),
    "molmoact2": PolicyTrainingConfiguration(
        defaults=PolicyTrainingDefaults(
            batch_size=16,
            max_epochs=8,
            num_workers=8,
            lora_enabled=True,
            gradient_checkpointing=True,
            val_split_seed=42,
        ),
        help=PolicyTrainingHelp(
            max_epochs=(
                "The SO101 reference recipe uses 8 epochs with LoRA. Studio uses the general pretrained checkpoint; "
                "these defaults do not reproduce the SO101-specific model settings."
            ),
        ),
    ),
    "rldx1": PolicyTrainingConfiguration(
        defaults=PolicyTrainingDefaults(gradient_checkpointing=True),
        help=PolicyTrainingHelp(
            max_epochs="Five epochs is a starting budget, not a validated recipe. Review validation loss for your task."
        ),
    ),
    "xr0": PolicyTrainingConfiguration(
        defaults=PolicyTrainingDefaults(gradient_checkpointing=True, val_split=0.05, val_split_seed=42),
        help=PolicyTrainingHelp(
            max_epochs=(
                "Five epochs is a starting point. The SO101 reference recipe uses 30,000 optimizer steps and "
                "gradient accumulation; these defaults do not reproduce that full recipe."
            ),
        ),
    ),
}

# Hub dependencies are lazy and configuration-dependent. Inspect each policy's
# from_pretrained/hf_hub_download/Auto* calls when changing its default repositories.
HUGGINGFACE_REQUIREMENTS = {
    "molmoact2": (("allenai/MolmoAct2", False),),
    "pi05": (("lerobot/pi05_base", True), ("google/paligemma-3b-pt-224", True)),
    "smolvla": (("lerobot/smolvla_base", False),),
    "rldx1": (("RLWRLD/RLDX-1-PT", False),),
    "xr0": (("XiaomiRobotics/Xiaomi-Robotics-0-Pretrain", False), ("Qwen/Qwen3-VL-4B-Instruct", False)),
}

# These describe the provisioned AWS image, not locally installed capabilities.
AWS_POLICY_INSTANCE_TYPES = {
    "act": "g4dn.2xlarge",
    "pi05": "g6e.2xlarge",
    "smolvla": "g4dn.2xlarge",
    "molmoact2": "g7e.8xlarge",
}
AWS_POLICY_EXPORT_BACKENDS = {
    "act": ["torch", "openvino", "onnx", "executorch"],
    "pi05": ["torch", "openvino"],
    "smolvla": ["torch", "openvino"],
    "molmoact2": ["torch", "openvino"],
}


class AvailablePolicy(BaseModel):
    id: str
    name: str
    description: str
    min_vram: int | None


class PolicyCatalog(BaseModel):
    available: list[AvailablePolicy]
    backends: dict[str, list[str]]


# Only policies whose constructors support Studio's training job are selectable.
_STUDIO_POLICIES = {
    "act": AvailablePolicy(
        id="act",
        name="ACT",
        description="Action Chunking with Transformers, lightweight and fast to train (BSD 3-Clause license)",
        min_vram=2 * 1024**3,
    ),
    "molmoact2": AvailablePolicy(
        id="molmoact2",
        name="MolmoAct2",
        description="Vision-Language-Action model with flow-matching action generation (Apache 2.0 license)",
        min_vram=80 * 1024**3,
    ),
    "pi05": AvailablePolicy(
        id="pi05",
        name="Pi0.5",
        description="Flow-matching VLA with discrete state encoding and longer context (Gemma license)",
        min_vram=40 * 1024**3,
    ),
    "rldx1": AvailablePolicy(
        id="rldx1",
        name="RLDX-1",
        description="General-purpose robot foundation model for dexterous manipulation (RLWRLD Model License v1.0)",
        min_vram=40 * 1024**3,
    ),
    "smolvla": AvailablePolicy(
        id="smolvla",
        name="SmolVLA",
        description="Small Vision-Language-Action model based on SmolVLM2-500M (Apache 2.0 license)",
        min_vram=3 * 1024**3,
    ),
    "xr0": AvailablePolicy(
        id="xr0",
        name="XR0",
        description=(
            "Vision-Language-Action model with a Qwen3-VL backbone and flow-matching action expert (Apache 2.0 license)"
        ),
        min_vram=60 * 1024**3,
    ),
}


def missing_policy_dependencies(name: str) -> list[str]:
    """Check extras without loading deferred model dependencies."""
    missing = []
    for raw in metadata.requires("physicalai-train") or []:
        requirement = Requirement(raw)
        marker = requirement.marker
        if marker is None or not marker.evaluate({"extra": name}) or marker.evaluate({"extra": ""}):
            continue
        try:
            installed_version = metadata.version(requirement.name)
        except metadata.PackageNotFoundError:
            missing.append(requirement.name)
        else:
            if installed_version not in requirement.specifier:
                missing.append(f"{requirement.name}{requirement.specifier} (installed {installed_version})")
    return missing


def available_policy_classes(
    missing_dependencies: Callable[[str], list[str]] = missing_policy_dependencies,
    warning: Callable[..., None] = logger.warning,
) -> dict[str, tuple[type, list[str]]]:
    """Resolve policies and backends installed in this Python environment."""
    from physicalai.policies import get_physicalai_policy_class, list_physicalai_policies

    classes = {}
    for name in list_physicalai_policies():
        if name not in _STUDIO_POLICIES:
            continue
        if missing := missing_dependencies(name):
            warning("Policy {} is unavailable (missing optional dependencies): {}", name, ", ".join(missing))
            continue
        try:
            cls = get_physicalai_policy_class(name)
            backends = (
                [str(b) for b in cls.get_supported_export_backends()]
                if hasattr(cls, "get_supported_export_backends")
                else []
            )
            classes[name] = (cls, backends)
        except ImportError as exc:
            warning("Policy {} is unavailable: {}", name, exc)
    return classes


def policy_catalog(classes: dict[str, tuple[type, list[str]]]) -> PolicyCatalog:
    """Describe the policies and backends of this process."""
    return PolicyCatalog(
        available=[_STUDIO_POLICIES[name] for name in classes],
        backends={name: backends for name, (_, backends) in classes.items()},
    )


def declared_policy_catalog(backends: dict[str, list[str]]) -> PolicyCatalog:
    """Describe an image's declared capabilities without importing local policies."""
    return PolicyCatalog(available=[_STUDIO_POLICIES[name] for name in backends], backends=backends)
