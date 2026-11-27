"""Typed Studio training defaults and policy guidance."""

from typing import Annotated, Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator

BatchSize = Annotated[int, Field(ge=1, le=256)]
EpochCount = Annotated[int, Field(ge=1, le=10_000)]
StepCount = Annotated[int, Field(ge=1, le=100_000)]
NumWorkers = Annotated[int, Field(ge=0, le=16)] | Literal["auto"]
Precision = Literal["32-true", "bf16-mixed", "bf16-true"]


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

    distill_epochs: EpochCount | None = Field(default=3, description="Additional training epochs.")
    distill_steps: StepCount | None = Field(default=None, description="Additional optimizer updates.")

    @model_validator(mode="before")
    @classmethod
    def resolve_limit(cls, data: Any) -> Any:
        """Keep one active distillation limit; omitted values use the field defaults."""
        if not isinstance(data, dict):
            return data
        if data.get("distill_steps") is not None:
            return {**data, "distill_epochs": None}
        if data.get("distill_epochs") is not None:
            return {**data, "distill_steps": None}
        return {key: value for key, value in data.items() if key not in {"distill_epochs", "distill_steps"}}


class PolicyTrainingDefaults(BaseModel):
    """Recommended values; grouped options translate to the existing job payload."""

    model_config = ConfigDict(extra="forbid", frozen=True, json_schema_serialization_defaults_required=True)

    batch_size: BatchSize = Field(default=8, description="Samples per training step.")
    max_epochs: EpochCount | None = Field(default=5, description="Dataset passes; preferred over steps.")
    max_steps: StepCount | None = Field(default=None, description="Optimizer updates; null uses epochs.")
    num_workers: NumWorkers = Field(default="auto", description="DataLoader workers (0-16); auto uses available CPUs.")
    precision: Precision | None = Field(
        default=None,
        description="Policy-recommended precision; null keeps the device-based recommendation.",
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
        """Keep one active limit; omitted values use the policy's field defaults."""
        if not isinstance(data, dict):
            return data
        if data.get("max_steps") is not None:
            return {**data, "max_epochs": None}
        if data.get("max_epochs") is not None:
            return {**data, "max_steps": None}
        return {key: value for key, value in data.items() if key not in {"max_epochs", "max_steps"}}

    @model_validator(mode="after")
    def validate_distillation_unit(self) -> "PolicyTrainingDefaults":
        """Require the teacher and distillation to use the same unit."""
        if self.snapflow is not None:
            budget = self.snapflow.distill_epochs if self.max_epochs is not None else self.snapflow.distill_steps
            if budget is None:
                raise ValueError("SnapFlow distillation must use the same unit as the training limit.")
        return self


def _model_class(annotation: Any) -> type[BaseModel] | None:
    """Return the nested Pydantic model class in an annotation, if any."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    for argument in get_args(annotation):
        nested = _model_class(argument)
        if nested is not None:
            return nested
    return None


def describe_help_overrides(model: type[BaseModel], base: type[BaseModel], prefix: str = "") -> dict[str, str]:
    """Return policy-specific field guidance keyed by dotted field path."""
    descriptions: dict[str, str] = {}
    for name, field in model.model_fields.items():
        base_field = base.model_fields.get(name)
        path = f"{prefix}{name}"
        base_description = base_field.description if base_field is not None else None
        if field.description and field.description != base_description:
            descriptions[path] = field.description

        nested = _model_class(field.annotation)
        if nested is not None:
            base_nested = _model_class(base_field.annotation) if base_field is not None else None
            descriptions.update(describe_help_overrides(nested, base_nested or nested, prefix=f"{path}."))
    return descriptions


class PolicyCameraSlot(BaseModel):
    """One camera slot in the pretrained policy's fixed camera order."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(description="Stable pretrained camera slot identifier.")
    name: str = Field(description="Display name describing the camera view.")
    description: str = Field(description="Guidance for assigning a dataset camera to this slot.")
    is_required: bool = Field(description="Whether a camera must fill this slot for sensible training.")


class PolicyTrainingConfiguration(BaseModel):
    """Recommended training values with policy guidance and camera metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True, json_schema_serialization_defaults_required=True)

    defaults: PolicyTrainingDefaults = Field(description="Editable defaults to include in the training job payload.")
    help: dict[str, str] = Field(
        default_factory=dict,
        description="Policy-specific field guidance keyed by dotted field path; absent keys use generic UI help.",
    )
    camera_slots: list[PolicyCameraSlot] = Field(
        default_factory=list, description="Pretrained camera slots in order; empty means dataset order is used."
    )

    @classmethod
    def from_defaults(
        cls,
        defaults: PolicyTrainingDefaults,
        *,
        camera_slots: list[PolicyCameraSlot] | None = None,
    ) -> "PolicyTrainingConfiguration":
        """Build a response with guidance derived from the policy's field descriptions."""
        return cls(
            defaults=defaults,
            help=describe_help_overrides(type(defaults), PolicyTrainingDefaults),
            camera_slots=camera_slots or [],
        )
