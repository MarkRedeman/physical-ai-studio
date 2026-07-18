# Handoff: Plugin-Based Robot Schema via Registry

## Goal

Remove the hardcoded dependency of `schemas/robot.py` on concrete robot model classes (`SO101Robot`, `TrossenSingleArmRobot`, `TrossenBimanualRobot`). Instead, the discriminated union `Robot` should be constructed dynamically from `RobotCatalogRegistry`, so that external plugins can register their own robot types without modifying Studio's internal schemas.

## Constraints

- **Do NOT move Pydantic model classes** into `schemas/robot.py`. The model classes (`SO101Robot`, `TrossenSingleArmRobot`, `TrossenBimanualRobot`) stay in their catalog plugin files (`robots/catalog/so101.py`, `robots/catalog/widowxai.py`). Plugins should not depend on Studio-internal schemas.
- **Keep SO101 as a special case** for setup and motor calibration. Items 4 and 6 from the original review are out of scope.

## Three Work Items

### 1. Make `Robot` Union Registration-Based

**Current state** (`schemas/robot.py:5-15`):

```python
from robots.catalog.so101 import SO101Robot
from robots.catalog.widowxai import TrossenBimanualRobot, TrossenSingleArmRobot

Robot = Annotated[
    SO101Robot | TrossenSingleArmRobot | TrossenBimanualRobot,
    Field(discriminator="type"),
]
RobotAdapter: TypeAdapter[Robot] = TypeAdapter(Robot)
```

**Desired state**:

The `RobotCatalogRegistry` constructs `Robot` from registered definitions. The `RobotWithConnectionState` union should also be dynamic.

**Key challenges**:

- Pydantic discriminated unions must be defined at type-check time — you cannot use `Union[tuple(...)]` with dynamically discovered types. You need `Annotated[Union[...], Field(discriminator="type")]` at module level.
- Possible approach: collect model classes at import time into a module-level variable. The registry's `register()` method could update a `_robot_union` global.
- Alternatively, store model classes on each `RobotCatalogDefinition` and build a `TypeAdapter` at runtime each time it's needed. This is simpler but means the `Robot` type alias itself can't be used in type annotations.
- A pragmatic middle ground: the catalog plugin files expose a `Model` class (or `robot_model` attribute) on each `RobotCatalogDefinition`. A helper function `get_robot_adapter()` or `deserialize_robot()` builds the union on first call.

**Files to touch**:

| File | Change |
|------|--------|
| `robots/catalog/types.py` | Add `robot_model: type[BaseRobot]` field to `RobotCatalogDefinition` |
| `robots/catalog/so101.py` | Wire `SO101Robot` into each definition's `robot_model` |
| `robots/catalog/widowxai.py` | Wire `TrossenSingleArmRobot` / `TrossenBimanualRobot` into definitions |
| `robots/catalog/registry.py` | Maintain a `TypeAdapter` that unions all registered `robot_model` classes |
| `schemas/robot.py` | Replace hardcoded imports + union with delegation to registry adapter |
| `robots/robot_client_factory.py` line 9 | Remove `from schemas.robot import Robot, SO101Robot` (only `Robot` needed, `SO101Robot` can come from catalog) |
| All consumers of `Robot` / `RobotAdapter` | Likely no change — they already import `from schemas.robot import Robot` |

**Consumer files of `Robot` / `RobotAdapter`** (search for `from schemas.robot import`):

- `api/robots.py`
- `api/robot_control.py`
- `robots/robot_service.py`
- `repositories/project_robot_repo.py`
- `repositories/mappers/project_robot_mapper.py`
- `schemas/environment.py`

These should mostly work unchanged if `schemas.robot` re-exports `Robot` from the registry.

---

### 2. Open Up `RobotType`

**Current state** (`schemas/robot_type.py:11-17`):

```python
class RobotType(StrEnum):
    SO101_FOLLOWER = "SO101_Follower"
    SO101_LEADER = "SO101_Leader"
    TROSSEN_WIDOWXAI_LEADER = "Trossen_WidowXAI_Leader"
    TROSSEN_WIDOWXAI_FOLLOWER = "Trossen_WidowXAI_Follower"
    TROSSEN_BIMANUAL_WIDOWXAI_LEADER = "Trossen_Bimanual_WidowXAI_Leader"
    TROSSEN_BIMANUAL_WIDOWXAI_FOLLOWER = "Trossen_Bimanual_WidowXAI_Follower"
```

**Desired state**:

`RobotType` is an open-ended type — built-in values use `Literal` aliases within each plugin, and unknown values are just `str`. The DB column stores plain strings.

**Key challenges**:

- The `RobotType` enum is used as a SQLAlchemy `Enum` column type in `db/schema.py:59`. Changing to `str` requires a DB migration.
- Many files do `if robot.type == RobotType.SO101_FOLLOWER` — these comparisons still work with string literals, but the `RobotType.` prefix goes away.
- The type alias `SO101Types = Literal[RobotType.SO101_FOLLOWER, RobotType.SO101_LEADER]` stays in the plugin file or moves alongside the model class.

**Files to touch**:

| File | Change |
|------|--------|
| `schemas/robot_type.py` | Convert `RobotType` from `StrEnum` to a type alias: `RobotType = str` with known literals. Or keep as a `StrEnum` with a mechanism for extensions (e.g., a registry of known values). Simpler: just `RobotType = str`. |
| `db/schema.py:59` | Change `Enum(RobotType)` to just `String()` or `Text()`. Add a migration. |
| `robots/catalog/so101.py` | Define `SO101Types = Literal["SO101_Follower", "SO101_Leader"]` here (or import from schemas). The `SO101Robot.type` field uses this. |
| `robots/catalog/widowxai.py` | Define `TrossenTypes` and `TrossenBimanualTypes` literals here. |
| `schemas/robot_type.py` | Remove `SO101Types`, `TrossenTypes`, `TrossenBimanualTypes` type aliases (no longer needed centrally). |
| `robots/catalog/types.py` | `CatalogRobot.type` annotation changes from `RobotType` to `str`. |
| `robots/catalog/registry.py` | `get_definition` parameter changes from `RobotType` to `str`. |
| `robots/catalog/assets.py` line 22 | Uses `RobotType` indirectly through `get_definition` — may adjust. |
| `api/robot_catalog.py` | Route parameter type changes from `RobotType` to `str`. |
| `api/robot_setup.py` | Enum comparison becomes string comparison. |
| `robots/robot_client.py` | `robot_type` property return type changes from `RobotType` to `str`. |
| `robots/physicalai_adapter.py` | Same — `robot_type` field/property to `str`. |
| `robots/catalog/so101.py` | `_build_so101_driver` line 99: `RobotType.SO101_FOLLOWER` → `"SO101_Follower"`. |
| `robots/catalog/widowxai.py` | Lines 117, 126: `RobotType.TROSSEN_WIDOWXAI_FOLLOWER` → `"Trossen_WidowXAI_Follower"` etc. |
| `services/robot_calibration_service.py` | Lines 95-96: enum comparison becomes string comparison. |
| `repositories/mappers/project_robot_mapper.py` | `RobotType(model.type)` becomes just `model.type`. |

---

### 3. Genericize `CatalogRobotFactory` Protocol

**Current state** (`robots/catalog/types.py:61-67`):

```python
class CatalogRobotFactory(Protocol):
    async def find_so101_port(self, robot: SO101Robot) -> str: ...
    async def find_port_by_serial(self, serial_number: str) -> str | None: ...
    async def get_calibration_by_id(self, calibration_id: UUID | None) -> Calibration | None: ...
```

The `find_so101_port` method is typed to `SO101Robot` (imported from `schemas.robot` under `TYPE_CHECKING`), making the factory protocol SO101-specific.

**Desired state**:

Replace the SO101-specific parameter type with a generic or broader signature. The factory should work with any robot type's payload.

**Key challenges**:

- `find_so101_port` is only meaningful for SO101 robots. Other robot types don't have serial ports. Moving forward, the factory should be generic and plugins should provide their own port-resolution logic via the probe.
- The `SO101Robot` import in `types.py` (line 15) exists only for `TYPE_CHECKING`. If we remove it, the protocol becomes decoupled.
- `robot_client_factory.py:50-55` implements `find_so101_port` for the SO101 case. This implementation still works; only the protocol signature changes.

**Possible approach**:

```python
class CatalogRobotFactory(Protocol):
    async def find_port(self, robot: CatalogRobot[Any]) -> str: ...
    async def find_port_by_serial(self, serial_number: str) -> str | None: ...
    async def get_calibration_by_id(self, calibration_id: UUID | None) -> Calibration | None: ...
```

Or keep `find_so101_port` but type it more broadly:

```python
async def find_so101_port(self, robot: Any) -> str: ...
```

The cleanest approach: each `RobotCatalogDefinition` has a `resolve_port` method (or the probe handles it). The factory only needs `find_port_by_serial` and `get_calibration_by_id`. But since `_build_so101_driver` in `so101.py:88` calls `factory.find_so101_port(robot)`, the method must exist on the protocol — it just doesn't need to be SO101-typed.

**Files to touch**:

| File | Change |
|------|--------|
| `robots/catalog/types.py` | Remove `from schemas.robot import SO101Robot` import. Change `find_so101_port(self, robot: SO101Robot)` to `find_so101_port(self, robot: Any)` or `find_so101_port(self, robot: PayloadContainer[Any])`. |
| `robots/robot_client_factory.py` | Update `find_so101_port` signature to match new protocol type. The `SO101Robot` import on line 9 can remain for internal use. |
| `robots/catalog/so101.py` | `_build_so101_driver` calls `factory.find_so101_port(robot)` — no change needed if robot is typed as `SO101Robot` locally. |

---

## Summary of Files to Touch

| File | Work Item | Effort |
|------|-----------|--------|
| `robots/catalog/types.py` | 1, 2, 3 | Medium — add `robot_model` field, remove `RobotType` & `SO101Robot` deps |
| `robots/catalog/so101.py` | 1, 2 | Low — wire model into definitions, inline type aliases |
| `robots/catalog/widowxai.py` | 1, 2 | Low — wire models into definitions, inline type aliases |
| `robots/catalog/registry.py` | 1, 2 | Medium — build dynamic `Robot` union from registered models |
| `schemas/robot.py` | 1 | High — delegate to registry, remove hardcoded imports |
| `schemas/robot_type.py` | 2 | Medium — convert `StrEnum` to `str`, remove type aliases |
| `db/schema.py` | 2 | Low — change `Enum(RobotType)` to `String()`, add migration |
| `api/robot_catalog.py` | 2 | Low — route param type `str` instead of `RobotType` |
| `api/robot_setup.py` | 2 | Low — string comparison instead of `RobotType.X` |
| `robots/robot_client.py` | 2 | Low — property return type `str` |
| `robots/physicalai_adapter.py` | 2 | Low — field/property type `str` |
| `robots/robot_client_factory.py` | 1, 3 | Low — remove `Robot` import, `find_so101_port` signature |
| `robots/catalog/assets.py` | 2 | Low — may need type adjustment |
| `services/robot_calibration_service.py` | 2 | Low — string comparison |
| `repositories/mappers/project_robot_mapper.py` | 2 | Low — direct string instead of `RobotType(model.type)` |
| `robots/robot_service.py` | 1 | Check — uses `Robot` + `RobotWithConnectionState` |

## Ordering

1. **(2) Open up RobotType** — foundational change. Do this first. It's mechanical and affects many files but each change is simple.
2. **(3) Genericize CatalogRobotFactory** — small change, just the protocol and its implementation.
3. **(1) Registration-based Robot union** — the most impactful. Builds on top of the open `RobotType`.

## Verification

- `prek run --all-files` from `application/backend/` must pass.
- Server must start (`physicalai-studio serve`).
- Run `api/robot_catalog.py` endpoints manually: `GET /api/robots/catalog`, `GET /api/robots/catalog/so101_follower/urdf`.
- Create a robot via `POST /api/projects/{id}/robots` and read it back.
- DB migration for the `ProjectRobotDB.type` column should round-trip old enum values as strings.
