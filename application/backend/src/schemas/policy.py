from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field


class BaseHyperParameter(BaseModel):
    name: str = Field(description="Hyperparameter field name")
    description: str = Field(description="Description of the hyperparameter")
    human_name: str = Field(description="Human-friendly display name")


class IntHyperParameter(BaseHyperParameter):
    field_type: Literal["integer"] = "integer"
    default_value: int | None = Field(description="Default integer value")


class BooleanHyperParameter(BaseHyperParameter):
    field_type: Literal["boolean"] = "boolean"
    default_value: bool | None = Field(description="Default boolean value")


class FloatHyperParameter(BaseHyperParameter):
    field_type: Literal["float"] = "float"
    default_value: float | None = Field(description="Default float value")


class StringHyperParameter(BaseHyperParameter):
    field_type: Literal["string"] = "string"
    default_value: str | None = Field(description="Default string value")


class GroupHyperParameter(BaseHyperParameter):
    field_type: Literal["group"] = "group"
    hyper_parameters: list[PolicyHyperParameter] = Field(description="Nested hyperparameters")


class ChoiceHyperParameter(BaseHyperParameter):
    field_type: Literal["choice"] = "choice"
    default_value: Any = Field(description="Default selected value")
    allowed_values: list[Any] = Field(description="Allowed values")


PolicyHyperParameter = Annotated[
    GroupHyperParameter
    | IntHyperParameter
    | BooleanHyperParameter
    | FloatHyperParameter
    | StringHyperParameter
    | ChoiceHyperParameter,
    Field(discriminator="field_type"),
]


class PolicyHyperParametersResponse(BaseModel):
    policy: str
    hyper_parameters: list[PolicyHyperParameter]


GroupHyperParameter.model_rebuild()
PolicyHyperParametersResponse.model_rebuild()
