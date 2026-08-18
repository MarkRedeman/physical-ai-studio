"""Typed JSON Schema extensions understood by the Studio robot form."""

from __future__ import annotations

from typing import Literal, NotRequired, Required, TypedDict


class RobotUiInfo(TypedDict, total=False):
    """Read-only informational text shown in the Studio robot form."""

    title: NotRequired[str]
    text: Required[str]
    variant: NotRequired[Literal["info", "warning"]]


class RobotFieldUiOptions(TypedDict, total=False):
    """Per-field UI overrides understood by the Studio robot form."""

    required: bool


class RobotUiConnectionBinding(TypedDict, total=False):
    """Payload field bindings for the connection control."""

    connection: Required[str]
    serial_number: NotRequired[str]


class RobotUiConnectionControlOptions(TypedDict, total=False):
    """Options for the first-party connection control."""

    kind: Required[Literal["connection"]]
    label: NotRequired[str]
    description: NotRequired[str]
    device_discovery: NotRequired[bool]
    identify: NotRequired[bool]
    manual_entry: NotRequired[bool]
    infos: NotRequired[list[RobotUiInfo]]
    bind: Required[RobotUiConnectionBinding]


RobotUiControlOptions = RobotUiConnectionControlOptions


class RobotUiSectionOptions(TypedDict, total=False):
    """Presentation options for a section of payload fields and controls."""

    id: Required[str]
    title: NotRequired[str]
    description: NotRequired[str]
    infos: NotRequired[list[RobotUiInfo]]
    fields: NotRequired[list[str]]
    controls: NotRequired[list[RobotUiControlOptions]]


class RobotPayloadUiOptions(TypedDict, total=False):
    """Model-level options embedded in a payload's JSON Schema."""

    infos: NotRequired[list[RobotUiInfo]]
    sections: NotRequired[list[RobotUiSectionOptions]]


FieldSchemaExtra = TypedDict("FieldSchemaExtra", {"x-physicalai-ui": RobotFieldUiOptions})
ModelSchemaExtra = TypedDict("ModelSchemaExtra", {"x-physicalai-ui": RobotPayloadUiOptions})


def robot_field_ui(options: RobotFieldUiOptions) -> FieldSchemaExtra:
    """Create typed ``Field(json_schema_extra=...)`` metadata."""
    return {"x-physicalai-ui": options}


def robot_payload_ui(options: RobotPayloadUiOptions) -> ModelSchemaExtra:
    """Create typed ``ConfigDict(json_schema_extra=...)`` metadata."""
    return {"x-physicalai-ui": options}
