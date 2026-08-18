"""Typed JSON Schema extensions understood by the Studio robot form."""

from __future__ import annotations

from typing import Literal, TypedDict


class RobotUiInfo(TypedDict, total=False):
    """Read-only informational text shown in the Studio robot form."""

    title: str
    text: str
    variant: Literal["info", "warning"]


class RobotFieldUiOptions(TypedDict, total=False):
    """Presentation options for one Pydantic payload field."""

    group: str
    widget: Literal["device-selector"]
    required: bool


class RobotUiGroupOptions(TypedDict, total=False):
    """Presentation and hardware options for a group of payload fields."""

    title: str
    description: str
    device_discovery: bool
    identify: bool
    connection_key: str
    serial_number_key: str
    manual_entry: bool
    infos: list[RobotUiInfo]


class RobotPayloadUiOptions(TypedDict, total=False):
    """Model-level options embedded in a payload's JSON Schema."""

    groups: dict[str, RobotUiGroupOptions]
    infos: list[RobotUiInfo]


FieldSchemaExtra = TypedDict("FieldSchemaExtra", {"x-physicalai-ui": RobotFieldUiOptions})
ModelSchemaExtra = TypedDict("ModelSchemaExtra", {"x-physicalai-ui": RobotPayloadUiOptions})


def robot_field_ui(options: RobotFieldUiOptions) -> FieldSchemaExtra:
    """Create typed ``Field(json_schema_extra=...)`` metadata."""
    return {"x-physicalai-ui": options}


def robot_payload_ui(options: RobotPayloadUiOptions) -> ModelSchemaExtra:
    """Create typed ``ConfigDict(json_schema_extra=...)`` metadata."""
    return {"x-physicalai-ui": options}
