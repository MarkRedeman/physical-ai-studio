"""Hardware-free validation shared by draft setup endpoints."""

from typing import Any

from fastapi.exceptions import RequestValidationError
from physicalai_studio_plugin import RobotCatalogDefinition


def validate_draft_payload(definition: RobotCatalogDefinition, payload: dict[str, Any]) -> dict[str, Any]:
    """Validate plugin configuration, defaults, and Studio-required fields."""
    if definition.robot_payload is None:
        return payload
    model = definition.robot_payload.model_validate(payload)
    values = model.model_dump(mode="json")
    schema = definition.robot_payload.model_json_schema()
    errors: list[dict[str, Any]] = []

    def check(node: dict[str, Any], data: dict[str, Any], path: tuple[str, ...] = ()) -> None:
        for name, raw_field in node.get("properties", {}).items():
            field = raw_field
            if "$ref" in field:
                field = {**schema.get("$defs", {}).get(field["$ref"].split("/")[-1], {}), **field}
            options = field.get("x-physicalai-ui", {})
            required = name in node.get("required", []) or (isinstance(options, dict) and options.get("required"))
            value = data.get(name)
            if required and (value is None or (isinstance(value, str) and not value.strip())):
                errors.append({"loc": ("body", *path, name), "msg": "Field is required", "type": "missing"})
            if isinstance(value, dict):
                check(field, value, (*path, name))

    check(schema, values)
    if errors:
        raise RequestValidationError(errors)
    return values
