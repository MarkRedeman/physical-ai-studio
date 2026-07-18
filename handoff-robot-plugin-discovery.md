# Handoff: Robot Plugin Discovery & Identification via Catalog

## Goal

Move robot device discovery (`GET /api/hardware/serial_devices`) and robot identification
(`POST /api/hardware/identify`) from `hardware.py` into the robot catalog API
(`robot_catalog.py`), making both **fully pluggable** per robot type via the
`RobotCatalogDefinition`.

Once this is done, each catalog plugin decides how to find its devices and how to
physically identify a connected robot, and the API layer is a thin router that
delegates to the definition.

---

## Design

`RobotCatalogDefinition` (in `robots/catalog/types.py`) gets two new optional
callables:

```
discover(robot_manager: RobotConnectionManager) -> list[SerialPortInfo]
identify(payload: dict[str, Any], robot_manager: RobotConnectionManager | None, joint: str | None) -> None
```

- **`discover`**: returns the list of currently connected devices for this robot type.
- **`identify`**: performs a visual identification (e.g. move a joint or open/close
  gripper) using the connection info from the caller-supplied `payload` dict.

Each definition validates the `payload` dict internally using its own typed
Pydantic model (`SO101RobotPayload`, `TrossenSingleArmPayload`, etc.).

The API endpoints are thin wrappers:

```
GET  /api/robots/catalog/{robot_type}/discover
       → delegates to definition.discover(robot_manager)
       → returns list[SerialPortInfo]

POST /api/robots/catalog/{robot_type}/identify
       → body: dict[str, Any] (the payload for the robot type)
       → query: ?joint=...
       → delegates to definition.identify(payload, robot_manager, joint)
       → returns None
```

---

## Backend changes

### 1. `robots/catalog/types.py` — Extend `RobotCatalogDefinition`

Add `discover` and `identify` callables:

```python
class RobotCatalogDefinition(BaseModel):
    # … existing fields (type, display_name, role, urdf_path, …)

    discover: (
        Callable[[RobotConnectionManager], Awaitable[list[SerialPortInfo]]]
        | None
    ) = Field(default=None, exclude=True)

    identify: (
        Callable[
            [dict[str, Any], RobotConnectionManager | None, str | None],
            Awaitable[None],
        ]
        | None
    ) = Field(default=None, exclude=True)
```

The `exclude=True` keeps them out of the Pydantic serialization (they don't go
over the wire).

### 2. `robots/catalog/so101.py` — Implement for SO-101

**`_discover_so101(manager)`** → calls `manager.find_robots()`, returns the
`SerialPortInfo` list (reuses existing logic from `utils/serial_robot_tools.py`).

**`_identify_so101(payload, manager, joint)`** → parses `payload` as
`SO101RobotPayload`, constructs a minimal `SO101Robot`, and calls
`identify_so101_robot_visually(manager, robot, joint)` (existing).

Wire into both SO101 definitions returned by `get_definitions()`:

```python
RobotCatalogDefinition(
    …
    discover=_discover_so101,
    identify=_identify_so101,
)
```

### 3. `robots/catalog/widowxai.py` — Implement for Trossen

**`_discover_trossen(manager)`** → returns `[]` for now. IP subnet scanning is
future work.

**`_identify_single_arm(payload, manager, joint)`** → parses `payload` as
`TrossenSingleArmPayload`, calls `identify_trossen_robot_visually(robot)`.

**`_identify_bimanual(payload, manager, joint)`** → parses `payload` as
`TrossenBimanualPayload`. May identify left, right, or both arms depending on
convention (UI currently sends separate requests per arm; for now identify
the arm whose `connection_string` is present).

Wire into all four Trossen definitions in `get_definitions()`:

```python
RobotCatalogDefinition(
    …
    discover=_discover_trossen,
    identify=_identify_single_arm,      # for single-arm defs
    # or
    identify=_identify_bimanual,        # for bimanual defs
)
```

### 4. `api/robot_catalog.py` — New endpoints

```python
from typing import Any
from api.dependencies import RobotCatalogServiceDep, RobotConnectionManagerDep
from robots.catalog.types import RobotCatalogDefinition
from schemas import SerialPortInfo
from schemas.robot_type import RobotType

router = APIRouter(prefix="/api/robots/catalog", tags=["Robot Catalog"])

@router.get("/{robot_type}/discover")
async def discover_robots(
    catalog_service: RobotCatalogServiceDep,
    robot_manager: RobotConnectionManagerDep,
    robot_type: RobotType,
) -> list[SerialPortInfo]:
    """Discover connected devices for a robot type."""
    definition = catalog_service.get_definition(robot_type)
    if definition.discover is None:
        return []
    return await definition.discover(robot_manager)


@router.post("/{robot_type}/identify")
async def identify_robot(
    catalog_service: RobotCatalogServiceDep,
    robot_manager: RobotConnectionManagerDep,
    robot_type: RobotType,
    payload: dict[str, Any],
    joint: str | None = None,
) -> None:
    """Visually identify a robot by moving a joint or gripper."""
    definition = catalog_service.get_definition(robot_type)
    if definition.identify is None:
        raise ResourceNotFoundError(
            resource_type=ResourceType.ROBOT,
            resource_id=robot_type,
            message=f"Robot type {robot_type} does not support identification.",
        )
    await definition.identify(payload, robot_manager, joint)
```

### 5. Old endpoints (temporary)

Keep `GET /api/hardware/serial_devices` and `POST /api/hardware/identify` in
`hardware.py` during transition. Remove once UI migrates.

### 6. Tests — `tests/api/test_robot_catalog.py`

- `test_discover_so101` — mock `RobotConnectionManager.find_robots()`, verify
  serial port list returned.
- `test_discover_trossen` — verify `[]`.
- `test_discover_unknown_type` — verify `404`.
- `test_identify_so101` — mock SO101 definition's `identify`, verify called with
  the payload dict and joint.
- `test_identify_trossen` — same for Trossen.
- `test_identify_unknown_type` — verify `404`.
- `test_identify_no_identify_method` — verify graceful `404` when `identify` is None.

---

## UI changes

### 1. Regenerate API types

```sh
cd application/ui
npm run build:api:download && npm run build:api
```

This adds typed paths for:
- `get`, `/api/robots/catalog/{robot_type}/discover`
- `post`, `/api/robots/catalog/{robot_type}/identify`

### 2. New hooks (`robot-catalog.hooks.ts`)

```ts
export const useDiscoverRobotsQuery = (robotType: SchemaRobotType) => {
    return $api.useQuery('get', '/api/robots/catalog/{robot_type}/discover', {
        params: { path: { robot_type: robotType } },
    });
};

export const useCatalogIdentifyMutation = (robotType: SchemaRobotType) => {
    return $api.useMutation(
        'post',
        '/api/robots/catalog/{robot_type}/identify',
        { meta: { skipInvalidation: true } },
    );
};
```

### 3. Per-form updates

Each robot type form (`so101.tsx`, `widowxai.tsx`, `widowxai-bimanual.tsx`):

| Before | After |
|--------|-------|
| `useSuspenseQuery('get', '/api/hardware/serial_devices')` | `useDiscoverRobotsQuery(activeType)` |
| `useIdentifyMutation()` body: full `Robot` | `useCatalogIdentifyMutation(activeType)` body: **payload object** `{ connection_string: …, serial_number: … }` |
| `buildSO101Body`/`buildWidowxBody` (for identify) | Remove identify-specific helpers; creation helpers remain |

### 4. `actions.tsx` cleanup

- `RefreshRobotsButton` → refactor to accept `robotType` prop, refresh type-scoped discover query.
- `useIdentifyMutation` → remove (replaced by `useCatalogIdentifyMutation`).
- `IdentifyRobot` → accept `robotType` prop, use new mutation internally.

### 5. Robot detail page (`robot.tsx`)

- Replace inline `useMutation('post', '/api/hardware/identify', …)` with
  `useCatalogIdentifyMutation(robot.type)`.
- Body: `robot.payload` (or extract `connection_string` / `serial_number` from
  the robot's payload).

### 6. Removal (Phase 2)

After all UI code references only the new paths:
- Delete `GET /api/hardware/serial_devices` and `POST /api/hardware/identify` from `hardware.py`.
- Delete all remaining `'hardware/serial_devices'` and `'hardware/identify'` UI refs.
- Regenerate OpenAPI types.

---

## OpenAPI contract (new)

### `GET /api/robots/catalog/{robot_type}/discover`

| | |
|---|---|
| Path param | `robot_type`: enum of known types |
| Response `200` | `SerialPortInfo[]` — each: `{ connection_string: string\|null, serial_number: string\|null }` |
| Response `404` | Unknown robot type |

### `POST /api/robots/catalog/{robot_type}/identify`

| | |
|---|---|
| Path param | `robot_type`: enum of known types |
| Body | `object` (free-form dict; each catalog definition validates its own schema internally) |
| Query | `joint?: string` — optional joint name to move |
| Response `200` | `{}` (empty body) |
| Response `404` | Unknown robot type or no identify method |
