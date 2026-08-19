# Robot Catalog Plugins

Robot catalog plugins let Studio discover, configure, connect to, and display
robot drivers without changing Studio itself. Studio discovers installed plugins
through the `physicalai.studio.catalog_plugins` Python entry-point group.

## Plugin Shape

Declare an entry point in the plugin's `pyproject.toml`:

```toml
[project.entry-points."physicalai.studio.catalog_plugins"]
my-robot = "physicalai_my_robot_plugin.studio_catalog:register_physicalai_studio_plugin"
```

The entry point registers one `RobotCatalogDefinition` per robot type:

```python
from physicalai_studio_plugin import RobotCatalogDefinition


def register_physicalai_studio_plugin(registry) -> None:
    registry.register_robot(
        RobotCatalogDefinition(
            type="MyRobot_Follower",
            display_name="My Robot Follower",
            category="My Robot",
            source="external",
            role="follower",
            robot_payload=MyRobotPayload,
            robot_builder=build_my_robot,
            probe=MyRobotProbe(),
        )
    )
```

`type` is a stable persisted identifier and must be unique across all installed
plugins. `display_name` is the label shown to users. `role` is either
`follower` (a robot that executes actions) or `leader` (a teleoperation input).

## Catalog Presentation

`category` groups related cards in the Studio robot-selection dialog. Use a
short, product-level name such as `ReBot`, `LeRobot`, or `SO101`.

`source` describes ownership:

- `internal`: supplied by Studio itself.
- `first_party`: supplied by an Intel or Physical AI maintained plugin.
- `external`: supplied by another integration.

Users can hide external robots in the catalog browser. Third-party plugins
should therefore set `source="external"`.

Set `RobotAsset.preview_thumbnail` to an optional path relative to the asset
root. Studio serves it as the selection-card image and uses a fallback when it
is not provided.

## Payload, Connection, And Assets

`robot_payload` is a Pydantic model describing configuration submitted by the
Studio form. The async `robot_builder` receives this validated payload and a
`CatalogRobotFactory`, and returns the Physical AI driver. A `RobotProbe` can
implement device discovery, identification, and reachability checks.

`asset` is optional. When supplied, a `RobotAsset` identifies a URDF, ROS
package paths, joint mapping, and an optional root resolver. Studio serves the
URDF and associated meshes to its 3D viewer. When a robot has no URDF, set
`asset=None`; users can still select, configure, and connect it, but Studio
shows no 3D preview.

## Configuring The Robot Form UI

Studio renders the robot payload form from your payload model's JSON Schema.
Use standard Pydantic schema fields first:

- `title`: field label shown in the form.
- `description`: help text shown under a field (works for text, number, enum,
  boolean, and first-party controls).
- `default`: default value pre-filled in the form.
- `enum`: renders a dropdown picker.

`robot_payload_ui(...)` is optional. Without it, Studio renders the payload
directly from JSON Schema. Use it when you need ordering, sections, information,
or a first-party multi-field component.

### Field-level UI options (`robot_field_ui`)

Attach to `Field(..., json_schema_extra=robot_field_ui({...}))`.

- `required` (`bool`): treats a field as required in Studio UI even if the
  underlying schema field is optional/defaulted.
- `advanced_configuration` (`bool`): hides an optional field behind the form's
  "Show advanced options" switch until the user enables it. Use for fine-tuning
  fields with a safe default that most users should not need to change.

### Payload UI items (`robot_payload_ui`)

Attach an ordered list of items to `ConfigDict(json_schema_extra={...})`:

- `section`: recursively groups items under an optional heading and description.
- `field`: renders one named schema field.
- `connection`: renders the first-party serial device selector and owns its
  bound payload fields.
- `info`: renders read-only guidance text.

Fields not named by an item and not owned by a control render after configured
items using normal JSON Schema rendering. This lets plugins customize only the
parts of a form that need special UI.

### Section item (`RobotUiSectionOptions`)

Each section item supports:

- `kind` (`"section"`, required): item type.
- `id` (`str`, required): stable section identifier.
- `title` (`str`): optional heading.
- `description` (`str`): optional section help text.
- `items` (`list[RobotUiItem]`, required): recursively rendered children.

### Field item (`RobotUiFieldItem`)

- `kind` (`"field"`, required): item type.
- `name` (`str`, required): field name in the current payload model.

### Connection item (`RobotUiConnectionItem`)

- `kind` (`"connection"`, required): control type.
- `label` (`str`): control label.
- `description` (`str`): control help text.
- `device_discovery` (`bool`): enables discover/refresh device behavior.
- `identify` (`bool`): shows an `Identify` button that calls the probe's
  `identify(...)` with current payload.
- `manual_entry` (`bool`): when `false`, only discovered values can be used;
  when omitted/`true`, user can type a custom value.
- `bind` (`RobotUiConnectionBinding`, required): payload field bindings.

Connection bindings:

- `bind.connection` (`str`, required): payload key that stores selected/manual
  connection value.
- `bind.serial_number` (`str`, optional): payload key that stores selected
  serial number when available.

Connection bindings are relative to the payload object that declares the item.
For nested Pydantic models, define UI items on that nested model rather than
using a dotted field path.

### Info item (`RobotUiInfoItem`)

- `kind` (`"info"`, required): item type.
- `title` (`str`, optional): short heading.
- `text` (`str`, required): body text.
- `variant` (`"info" | "warning"`, optional): visual emphasis.

### Example

```python
from physicalai_studio_plugin import robot_payload_ui
from pydantic import BaseModel, ConfigDict, Field


class MyRobotPayload(BaseModel):
    connection_string: str = Field(
        default="",
        title="Connection",
        description="Select a discovered port or enter one manually.",
    )
    serial_number: str = Field(
        default="",
        title="Serial number",
        description="Optional hardware serial identifier.",
    )
    torque_enabled: bool = Field(
        default=True,
        description="Enable motor torque during startup.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            **robot_payload_ui(
                [
                    {
                        "kind": "section",
                        "id": "connection",
                        "title": "Connection",
                        "description": "Use Refresh to rescan ports.",
                        "items": [
                            {
                                "kind": "info",
                                "title": "Before connecting",
                                "text": "Power on the robot and clear the workspace.",
                            },
                            {
                                "kind": "connection",
                                "label": "Robot connection",
                                "device_discovery": True,
                                "identify": True,
                                "manual_entry": True,
                                "bind": {
                                    "connection": "connection_string",
                                    "serial_number": "serial_number",
                                },
                            },
                            {
                                "kind": "info",
                                "text": "USB hubs can change assigned port names after reboot.",
                                "variant": "warning",
                            },
                        ],
                    },
                    {
                        "kind": "section",
                        "id": "behavior",
                        "title": "Behavior",
                        "items": [{"kind": "field", "name": "torque_enabled"}],
                    },
                ]
            )
        }
    )
```

See `application/plugin/README.md` for the full SDK reference and the ReBot
catalog implementations for complete examples.
