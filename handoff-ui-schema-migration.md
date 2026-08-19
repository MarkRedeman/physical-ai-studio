# UI Schema Migration Handoff (v2)

This document describes the pre-release robot-form UI schema used by Studio
plugins. It replaces field-level `group` and `widget` metadata with a recursive,
ordered item tree.

## Core Principle

Pydantic/JSON Schema describes robot configuration data. `robot_payload_ui(...)`
is optional presentation metadata for ordering that data and adding first-party
form components.

If a payload does not use `robot_payload_ui(...)`, Studio renders its JSON
Schema directly, including `title`, `description`, defaults, enums, required
fields, and nested Pydantic models.

## Migration Summary

Remove these field-level options:

- `group`
- `widget`

Keep this field-level option where needed:

- `required`: Studio-only required override for an optional/defaulted field.

Replace `robot_payload_ui({"sections": [...]})` with an item list:

```python
robot_payload_ui([
    {"kind": "section", "id": "setup", "items": [...]},
])
```

## Item Types

Every payload UI is a `list[RobotUiItem]`. Items render in list order.

### `section`

Groups nested items and can recursively contain other sections.

```python
{
    "kind": "section",
    "id": "setup",
    "title": "Setup",
    "description": "Configure the robot before connecting.",
    "items": [...],
}
```

- `id` is a stable local identifier.
- `title` and `description` are optional.
- `items` is required and may contain any item type, including another section.

### `field`

Renders a normal field from the current Pydantic model.

```python
{"kind": "field", "name": "baud_rate"}
```

The field's label, description, default, enum, and validation still come from
its JSON Schema.

### `connection`

Renders Studio's serial-device picker. This item owns multiple related payload
fields, so Studio does not render duplicate raw inputs for those fields.

```python
{
    "kind": "connection",
    "label": "Select robot",
    "device_discovery": True,
    "identify": True,
    "manual_entry": True,
    "bind": {
        "connection": "connection_string",
        "serial_number": "serial_number",
    },
}
```

- `bind.connection` is required and stores the selected or manually entered
  connection value.
- `bind.serial_number` is optional and stores a discovered serial number.
- Selecting a discovered device writes both configured bindings.
- Manual entry writes `connection` and clears `serial_number` to avoid stale
  hardware identity.
- Only use this item for serial device selection. Network/IP configuration uses
  ordinary `field` items.

### `info`

Renders non-editable guidance inline with other items.

```python
{
    "kind": "info",
    "title": "Before connecting",
    "text": "Power on the robot and clear the workspace.",
    "variant": "warning",
}
```

## Default Rendering

Items customize only the fields that they own.

- Fields named by `field` items render at that exact position.
- Fields bound by a first-party item are owned by that item and do not render as
  duplicate raw inputs.
- Fields not named by any item render after all configured top-level items.
- Existing "Show default fields" behavior remains in effect.

This allows plugins to add one connection control while allowing Studio to
render all other payload fields automatically.

## Nested Payloads

Bindings are always relative to the Pydantic model whose UI item list declares
them. Do not use dotted paths such as `left_arm_config.connection_string`.

Instead, each nested Pydantic model owns its own UI item list. Studio renders
nested models recursively.

### SO101 Arm Payload

```python
class SO101Payload(BaseModel):
    connection_string: str = Field(default="", description="Serial port path")
    serial_number: str = Field(default="", description="USB serial number")
    calibration: dict[str, object] | None = Field(default=None, description="Joint calibration")

    model_config = ConfigDict(
        json_schema_extra={
            **robot_payload_ui(
                [
                    {
                        "kind": "section",
                        "id": "connection",
                        "title": "Connection",
                        "items": [
                            {
                                "kind": "connection",
                                "label": "Select robot",
                                "device_discovery": True,
                                "identify": True,
                                "manual_entry": True,
                                "bind": {
                                    "connection": "connection_string",
                                    "serial_number": "serial_number",
                                },
                            }
                        ],
                    },
                    {
                        "kind": "section",
                        "id": "calibration",
                        "title": "Calibration",
                        "items": [{"kind": "field", "name": "calibration"}],
                    },
                ]
            )
        }
    )
```

### Bimanual SO101 Reusing Nested UI

```python
class BimanualSO101Payload(BaseModel):
    left_arm_config: SO101Payload = Field(title="Left arm")
    right_arm_config: SO101Payload = Field(title="Right arm")

    model_config = ConfigDict(
        json_schema_extra={
            **robot_payload_ui(
                [
                    {
                        "kind": "section",
                        "id": "arms",
                        "title": "Arms",
                        "items": [
                            {"kind": "field", "name": "left_arm_config"},
                            {"kind": "field", "name": "right_arm_config"},
                        ],
                    }
                ]
            )
        }
    )
```

Each arm renders its own `SO101Payload` connection/calibration sections. The
parent does not need to understand or address its nested connection fields.

## Composite Robots

A fixed composite with known nested robots works now by using normal nested
Pydantic fields, like the bimanual example.

A future user-configured composite with a dynamic heterogeneous list of robots
will need a dedicated first-party item, for example `kind: "robot-members"`.
That item should own the complete members collection, select each member's robot
type from the catalog, and render each selected payload recursively. It should
not be modeled as path-based bindings.
