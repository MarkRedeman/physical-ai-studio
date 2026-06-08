import dataclasses
import types
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

from fastapi import APIRouter
from fastapi.exceptions import HTTPException
from physicalai.policies import ACT, Pi0, Pi0Config, Pi05, SmolVLA
from physicalai.policies.act import ACTConfig
from physicalai.policies.pi05 import Pi05Config
from physicalai.policies.smolvla import SmolVLAConfig
from starlette import status

from schemas.policy import PolicyHyperParameter, PolicyHyperParametersResponse

router = APIRouter(prefix="/api/policies", tags=["Policies"])

_POLICY_CLASSES = {
    "act": ACT,
    "pi0": Pi0,
    "pi05": Pi05,
    "smolvla": SmolVLA,
}

_POLICY_CONFIG_CLASSES = {
    "act": ACTConfig,
    "pi0": Pi0Config,
    "pi05": Pi05Config,
    "smolvla": SmolVLAConfig,
}

_SKIPPED_HYPERPARAMETER_FIELDS = {"compile_model"}


@router.get("/backends")
def get_supported_backends_per_policy() -> dict[str, list[str]]:
    """Return the supported export backends for each policy."""
    return {
        name: [str(b) for b in cls.get_supported_export_backends()]
        if hasattr(cls, "get_supported_export_backends")
        else []
        for name, cls in _POLICY_CLASSES.items()
    }


@router.get("/{policy}/hyper_parameters")
def get_policy_hyper_parameters(policy: str) -> PolicyHyperParametersResponse:
    """Return user-tunable hyperparameters accepted by a policy config."""
    config_cls = _POLICY_CONFIG_CLASSES.get(policy)
    if config_cls is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Policy '{policy}' does not expose hyperparameters.",
        )

    return PolicyHyperParametersResponse(
        policy=policy,
        hyper_parameters=_hyper_parameters_from_config(config_cls()),
    )


def _hyper_parameters_from_config(config: object) -> list[PolicyHyperParameter]:
    """Convert grouped config dataclasses into API hyperparameter descriptors."""
    hyper_parameters: list[PolicyHyperParameter] = []
    type_hints = get_type_hints(type(config))

    for config_field in dataclasses.fields(config):
        value = getattr(config, config_field.name)
        human_name = str(config_field.metadata.get("title") or _humanize_field_name(config_field.name))
        description = str(config_field.metadata.get("description", ""))

        if dataclasses.is_dataclass(value):
            hyper_parameters.append(
                PolicyHyperParameter(
                    name=config_field.name,
                    field_type="group",
                    default_value=None,
                    description=description,
                    human_name=human_name,
                    hyper_parameters=_hyper_parameters_from_config(value),
                ),
            )
            continue

        if config_field.name in _SKIPPED_HYPERPARAMETER_FIELDS:
            continue

        annotation = type_hints.get(config_field.name, config_field.type)
        allowed_values = _allowed_values(annotation, config_field.metadata)
        field_type = "choice" if allowed_values is not None else _field_type(annotation)
        if field_type is None:
            continue

        hyper_parameters.append(
            PolicyHyperParameter(
                name=config_field.name,
                field_type=field_type,
                default_value=value,
                description=description,
                human_name=human_name,
                allowed_values=allowed_values,
            ),
        )

    return hyper_parameters


def _field_type(annotation: object) -> Literal["integer", "boolean", "float", "string"] | None:
    """Map supported Python annotations to API primitive type names."""
    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin in {types.UnionType, Union}:
        non_none_args = [arg for arg in args if arg is not type(None)]
        if len(non_none_args) == 1:
            return _field_type(non_none_args[0])

    if origin is Literal:
        literal_args = [arg for arg in args if arg is not None]
        if not literal_args:
            return None
        return _field_type(type(literal_args[0]))

    if annotation is bool:
        return "boolean"
    if annotation is int:
        return "integer"
    if annotation is float:
        return "float"
    if annotation is str:
        return "string"

    return None


def _allowed_values(annotation: object, metadata: Any) -> list[Any] | None:
    """Return allowed choice values from Literal annotations or schema metadata."""
    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin in {types.UnionType, Union}:
        non_none_args = [arg for arg in args if arg is not type(None)]
        if len(non_none_args) == 1:
            return _allowed_values(non_none_args[0], metadata)

    if origin is Literal:
        literal_args = [arg for arg in args if arg is not None]
        return list(literal_args) if literal_args else None

    json_schema_extra = metadata.get("json_schema_extra")
    if isinstance(json_schema_extra, dict):
        enum_values = json_schema_extra.get("enum")
        if isinstance(enum_values, list):
            return enum_values

    return None


def _humanize_field_name(name: str) -> str:
    """Convert snake_case field names to display labels."""
    replacements = {
        "lr": "LR",
        "vae": "VAE",
        "vlm": "VLM",
        "dtype": "Dtype",
    }
    return " ".join(replacements.get(part, part.capitalize()) for part in name.split("_"))
