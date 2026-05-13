# Adding Bimanual SO101 Arms

## Decision

For v1, we will **reuse the existing single-arm SO101 setup and calibration flow twice** (left + right), instead of building a dedicated bimanual setup wizard.

## What This Adds

- A new bimanual SO101 robot configuration in backend and UI.
- A backend bimanual adapter that composes two existing `SO101Adapter` instances.
- Discovery and connectivity checks that require both arms to be online.
- UI creation/edit flows that capture left and right SO101 arm details.

## Implementation Plan

1. **Schema and types (backend)**
   - Extend `application/backend/src/schemas/robot.py` with bimanual SO101 robot types.
   - Add a bimanual SO101 payload model with left/right arm connection info (serial, connection string, calibration refs as needed).
   - Extend the discriminated unions (`Robot`, `RobotWithConnectionState`) with the new bimanual SO101 model.

2. **Bimanual SO101 adapter (backend)**
   - Add `application/backend/src/robots/so101/bimanual_adapter.py`.
   - Implement it as a composition wrapper over two `SO101Adapter` instances (`left`, `right`).
   - Prefix exported feature/state keys with `left_` and `right_` (matching existing bimanual key conventions).
   - Reuse existing normalization and motion clamping behavior from `SO101Adapter` by delegating to each arm adapter.

3. **Robot client factory wiring (backend)**
   - Update `application/backend/src/robots/robot_client_factory.py` to route new bimanual SO101 robot types.
   - Resolve port and calibration independently for each arm using existing logic.
   - Construct two `SO101` drivers, wrap each with `SO101Adapter`, then wrap with `BimanualSO101Adapter`.

4. **Discovery and online checks (backend)**
   - Update `application/backend/src/robots/discovery/manager.py` to recognize bimanual SO101 types.
   - Consider a bimanual SO101 robot online only when **both** serials are found.
   - Keep single-arm SO101 and Trossen discovery behavior unchanged.

5. **Identify and setup behavior (backend)**
   - Reuse existing single-arm identify/setup flow (`SO101` setup worker).
   - Trigger setup per arm (left then right, or independently), then save a combined bimanual robot config.
   - Do not introduce a new bimanual setup websocket workflow in v1.

6. **UI integration**
   - Update robot type unions and forms in:
     - `application/ui/src/features/robots/robot-types.ts`
     - `application/ui/src/features/robots/robot-form/provider.tsx`
     - `application/ui/src/features/robots/robot-form/form.tsx`
     - `application/ui/src/features/robots/robots-list.tsx`
     - `application/ui/src/features/robots/robots-configuration.ts`
   - Add bimanual SO101 robot type choices and left/right SO101 arm fields.
   - Reuse the existing per-arm setup/identify UX patterns.

7. **OpenAPI + frontend typings**
   - Regenerate backend OpenAPI schema after backend model changes.
   - Regenerate/update `application/ui/src/api/openapi-spec.json` and `application/ui/src/api/openapi-spec.d.ts` so new bimanual SO101 types are available in UI code.

8. **Tests**
   - Add backend tests for:
     - bimanual SO101 adapter behavior
     - factory creation paths
     - mapper round-trip of new payload
     - online discovery behavior for both arms required
   - Keep existing single-arm SO101 and bimanual WidowX tests passing.

## Acceptance Criteria

- A bimanual SO101 robot can be created and persisted with left/right arm payload data.
- Websocket control works through one bimanual client, with prefixed left/right state keys.
- Online status is `online` only when both SO101 arms are reachable.
- Existing single-arm SO101 setup and calibration flows remain unchanged and reusable.
- UI can create/edit/list bimanual SO101 robots without type errors.
- Relevant backend and frontend tests pass.

## Out of Scope for v1

- A dedicated combined bimanual SO101 setup wizard.
- PhysicalAI-level `BimanualSO101` driver abstraction (can be added later if needed).
