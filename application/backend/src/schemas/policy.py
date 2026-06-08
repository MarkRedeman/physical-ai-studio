from typing import Any, Literal

from pydantic import BaseModel, Field


class PolicyHyperParameter(BaseModel):
    name: str = Field(description="Hyperparameter field name")
    field_type: Literal["group", "integer", "boolean", "float", "string", "choice"] = Field(
        description="Hyperparameter field type",
    )
    default_value: Any = Field(default=None, description="Default value for the hyperparameter")
    description: str = Field(description="Description of the hyperparameter")
    human_name: str = Field(description="Human-friendly display name")
    allowed_values: list[Any] | None = Field(default=None, description="Allowed values for choice fields")
    hyper_parameters: list["PolicyHyperParameter"] = Field(
        default_factory=list,
        description="Nested hyperparameters for group fields",
    )


class PolicyHyperParametersResponse(BaseModel):
    policy: str
    hyper_parameters: list[PolicyHyperParameter]


PolicyHyperParameter.model_rebuild()
