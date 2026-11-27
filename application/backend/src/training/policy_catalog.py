"""Discover trainable policies in the process that will execute the training job."""

from collections.abc import Callable
from importlib import metadata

from loguru import logger
from packaging.requirements import Requirement
from pydantic import BaseModel, Field

from training.policy_training_configuration import (
    BatchSize,
    EpochCount,
    LoraDefaults,
    PolicyTrainingConfiguration,
    PolicyTrainingDefaults,
    Precision,
    StepCount,
    ValidationDefaults,
)


class ACTTrainingDefaults(PolicyTrainingDefaults):
    """ACT defaults validated against the SO101 reference recipe."""

    batch_size: BatchSize = Field(
        default=8,
        description="The ACT baseline uses batch size 8. Larger images and more cameras need more memory.",
    )
    max_epochs: EpochCount | None = Field(
        default=10,
        description=(
            "The 200-episode SO101 baseline uses 10 epochs, roughly 100k steps at batch 8. "
            "Smaller datasets need more epochs for the same update budget. Validate on your robot."
        ),
    )
    precision: Precision = Field(
        default="32-true",
        description="The single-arm ACT baseline uses FP32, without automatic mixed precision.",
    )
    augment_images: bool = Field(
        default=False,
        description="Disabled in the baseline. Enable it only if it improves held-out task performance.",
    )


class MolmoAct2LoraDefaults(LoraDefaults):
    """MolmoAct2 VLM adapter defaults from the native baseline."""

    rank: int = Field(
        default=64,
        ge=1,
        le=256,
        description="Rank 64 and alpha 16 match the native MolmoAct2 baseline.",
    )
    alpha: int | None = Field(default=16, ge=1, description="Scaling numerator; null uses the rank.")


class MolmoAct2TrainingDefaults(PolicyTrainingDefaults):
    """MolmoAct2 single-robot defaults; a candidate, not a validated rollout result."""

    max_epochs: EpochCount | None = Field(
        default=8,
        description=(
            "The single-robot baseline uses 8 epochs on 200 SO101 episodes. This is a candidate, "
            "not a validated rollout result. Adjust the budget for your dataset."
        ),
    )
    lora: MolmoAct2LoraDefaults | None = Field(
        default_factory=MolmoAct2LoraDefaults,
        description="VLM LoRA with a fully trainable action expert, initialized from the general checkpoint.",
    )


class SmolVLATrainingDefaults(PolicyTrainingDefaults):
    """SmolVLA defaults validated against the SO100 pick-and-place reference."""

    batch_size: BatchSize = Field(
        default=64,
        description="The SO100 pick-and-place reference uses batch 64. Reduce it if GPU memory is insufficient.",
    )
    max_epochs: EpochCount | None = Field(
        default=None,
        description=(
            "In epoch mode, roughly 75 epochs on the 50-episode reference or 15 on the 200-episode SO101 "
            "dataset approximates 20k updates at batch 64. Adjust for your training split."
        ),
    )
    max_steps: StepCount | None = Field(
        default=20_000,
        description="The SO100 pick-and-place reference uses 20,000 optimizer updates. Validate on your robot.",
    )
    augment_images: bool = Field(
        default=False,
        description="Disabled in the baseline; some real SO101 runs enable it. Compare physical rollouts.",
    )


POLICY_TRAINING_CONFIGURATIONS: dict[str, PolicyTrainingConfiguration] = {
    "act": PolicyTrainingConfiguration.from_defaults(
        ACTTrainingDefaults(num_workers=4, validation=ValidationDefaults(seed=42))
    ),
    "molmoact2": PolicyTrainingConfiguration.from_defaults(
        MolmoAct2TrainingDefaults(
            batch_size=16,
            num_workers=8,
            gradient_checkpointing=True,
            validation=ValidationDefaults(seed=42),
        )
    ),
    "smolvla": PolicyTrainingConfiguration.from_defaults(
        SmolVLATrainingDefaults(num_workers=4, precision="32-true", validation=ValidationDefaults(seed=42))
    ),
}
"""Per-policy recipe overrides; absent policies use the shared defaults."""

DEFAULT_POLICY_TRAINING_CONFIGURATION = PolicyTrainingConfiguration.from_defaults(PolicyTrainingDefaults())


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
