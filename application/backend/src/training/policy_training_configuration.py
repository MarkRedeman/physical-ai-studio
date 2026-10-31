"""Typed Studio training defaults, policy guidance, and camera slot metadata."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def resolve_training_limit(
    max_epochs: int | None, max_steps: int | None, *, default_epochs: int = 5
) -> tuple[int | None, int | None]:
    """Keep one active limit, preferring epochs for compatibility with older jobs."""
    if max_epochs is not None:
        return max_epochs, None
    if max_steps is not None:
        return None, max_steps
    return default_epochs, None


class ValidationDefaults(BaseModel):
    """Episode validation split configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True, json_schema_serialization_defaults_required=True)

    split: float = Field(default=0.1, ge=0, lt=1, description="Fraction of episodes held out for validation.")
    seed: int | None = Field(default=None, description="Validation split RNG seed; null uses the global RNG.")


class LoraDefaults(BaseModel):
    """Adapter settings; the presence of this configuration enables LoRA."""

    model_config = ConfigDict(extra="forbid", frozen=True, json_schema_serialization_defaults_required=True)

    rank: int = Field(default=32, ge=1, le=256, description="LoRA adapter rank.")
    alpha: int | None = Field(default=None, ge=1, description="Scaling numerator; null uses the rank.")
    dropout: float = Field(default=0.05, ge=0, lt=1, description="Adapter input dropout probability.")
    use_dora: bool = Field(default=False, description="Use weight-decomposed adapters instead of plain LoRA.")


class SnapflowDefaults(BaseModel):
    """Distillation settings; the presence of this configuration enables SnapFlow."""

    model_config = ConfigDict(extra="forbid", frozen=True, json_schema_serialization_defaults_required=True)

    distill_epochs: int | None = Field(default=None, ge=1, le=10_000, description="Additional training epochs.")
    distill_steps: int | None = Field(default=None, ge=1, le=100_000, description="Additional optimizer updates.")

    @model_validator(mode="before")
    @classmethod
    def resolve_limit(cls, data: Any) -> Any:
        """Default to three epochs unless an explicit step budget is given."""
        if not isinstance(data, dict):
            return data
        epochs, steps = resolve_training_limit(data.get("distill_epochs"), data.get("distill_steps"), default_epochs=3)
        return {**data, "distill_epochs": epochs, "distill_steps": steps}


class PolicyTrainingDefaults(BaseModel):
    """Recommended values; grouped options translate to the existing job payload."""

    model_config = ConfigDict(extra="forbid", frozen=True, json_schema_serialization_defaults_required=True)

    batch_size: int = Field(default=8, ge=1, le=256, description="Samples per training step.")
    max_epochs: int | None = Field(default=None, ge=1, le=10_000, description="Dataset passes; preferred over steps.")
    max_steps: int | None = Field(default=None, ge=1, le=100_000, description="Optimizer updates; null uses epochs.")
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
    validation: ValidationDefaults = Field(default_factory=ValidationDefaults, description="Episode validation split.")
    lora: LoraDefaults | None = Field(default=None, description="LoRA/DoRA adapter settings; null disables adapters.")
    snapflow: SnapflowDefaults | None = Field(
        default=None, description="Distillation settings; null disables SnapFlow."
    )

    @model_validator(mode="before")
    @classmethod
    def resolve_limit(cls, data: Any) -> Any:
        """Default to five epochs only when neither limit is supplied."""
        if not isinstance(data, dict):
            return data
        epochs, steps = resolve_training_limit(data.get("max_epochs"), data.get("max_steps"))
        return {**data, "max_epochs": epochs, "max_steps": steps}

    @model_validator(mode="after")
    def validate_distillation_unit(self) -> "PolicyTrainingDefaults":
        """Require the teacher and distillation to use the same unit."""
        if self.snapflow is not None:
            budget = self.snapflow.distill_epochs if self.max_epochs is not None else self.snapflow.distill_steps
            if budget is None:
                raise ValueError("SnapFlow distillation must use the same unit as the training limit.")
        return self


class LoraHelp(BaseModel):
    """Optional adapter guidance; absent values use generic UI help."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    description: str | None = Field(default=None, description="Adapter fine-tuning recommendation.")
    rank: str | None = Field(default=None, description="Adapter capacity recommendation.")
    alpha: str | None = Field(default=None, description="Adapter scaling guidance.")
    dropout: str | None = Field(default=None, description="Adapter dropout guidance.")
    use_dora: str | None = Field(default=None, description="DoRA guidance.")


class SnapflowHelp(BaseModel):
    """Optional distillation guidance; absent values use generic UI help."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    description: str | None = Field(default=None, description="Distillation recommendation.")
    distill_epochs: str | None = Field(default=None, description="Distillation budget recommendation.")
    distill_steps: str | None = Field(default=None, description="Distillation optimizer-step budget recommendation.")


class PolicyTrainingHelp(BaseModel):
    """Optional policy guidance; absent fields use the UI's generic help."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    batch_size: str | None = Field(default=None, description="Batch size recommendation and memory tradeoffs.")
    max_epochs: str | None = Field(default=None, description="Training budget recommendation and dataset assumptions.")
    max_steps: str | None = Field(default=None, description="Optimizer-step budget recommendation.")
    num_workers: str | None = Field(default=None, description="Data loading worker recommendation.")
    precision: str | None = Field(default=None, description="Policy-specific numerical precision guidance.")
    compile_model: str | None = Field(default=None, description="Compilation speed and startup tradeoffs.")
    gradient_checkpointing: str | None = Field(default=None, description="Activation memory and compute tradeoffs.")
    auto_scale_batch_size: str | None = Field(default=None, description="Automatic batch size guidance.")
    augment_images: str | None = Field(default=None, description="Image augmentation recommendation.")
    lora: LoraHelp | None = Field(default=None, description="Optional adapter guidance.")
    snapflow: SnapflowHelp | None = Field(default=None, description="Optional distillation guidance.")


class PolicyCameraSlot(BaseModel):
    """One camera slot in the pretrained policy's fixed camera order."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(description="Stable pretrained camera slot identifier.")
    name: str = Field(description="Display name describing the camera view.")
    description: str = Field(description="Guidance for assigning a dataset camera to this slot.")
    is_required: bool = Field(description="Whether a camera must fill this slot for sensible training.")


class PolicyTrainingConfiguration(BaseModel):
    """Recommended training values with optional guidance and camera metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True, json_schema_serialization_defaults_required=True)

    defaults: PolicyTrainingDefaults = Field(description="Editable defaults to include in the training job payload.")
    help: PolicyTrainingHelp = Field(default_factory=PolicyTrainingHelp, description="Optional policy-specific help.")
    camera_slots: list[PolicyCameraSlot] = Field(
        default_factory=list, description="Pretrained camera slots in order; empty means dataset order is used."
    )
