# Plan: Reconfiguring Plugin Hardware from Studio

## Original reason for this plan

We now support uploading calibration JSON into Studio robot payloads (for example SO101), but this introduces a configuration-drift problem:

1. A user can upload a calibration file in Studio that differs from calibration currently stored on the robot control board.
2. A user can later recalibrate or modify the control board directly, without updating Studio payload data.

Today, the plugin contract does not explicitly support checking or reconciling this drift. As a result, Studio cannot reliably answer whether hardware and payload are aligned, and cannot provide a safe, explicit workflow to reconfigure hardware from the saved payload.

## Goal

Add an explicit plugin capability for hardware reconciliation so Studio can:

- detect payload-vs-hardware drift,
- show drift status to users,
- and apply reconfiguration intentionally (never implicitly),

while preserving current robot-builder behavior and safety constraints.

## Scope and non-goals

### In scope

- Extend plugin catalog definition with optional reconfiguration capabilities.
- Add backend API endpoints to check sync status and trigger reconfiguration.
- Add UI states/actions to inspect and apply sync.
- Add tests and docs so plugin authors can adopt the feature.

### Out of scope (initial version)

- Automatic background writes to control boards.
- Mandatory support for all plugin robot types.
- Replacing existing `robot_builder(...)` flow.

## Design principles

1. **Explicit over implicit**: uploading payload data must not automatically write to hardware.
2. **Inspect before mutate**: users should be able to see drift first, then decide to apply.
3. **Capability-based**: plugins opt in; unsupported robots degrade gracefully.
4. **Session safety**: block reconfigure while a runtime owner session is active.

## Proposed plugin contract

Keep `robot_builder(...)` unchanged and add two optional callables to `RobotCatalogDefinition`:

- `reconfigure_status(payload) -> ReconfigureStatus`
- `reconfigure(payload, strategy) -> ReconfigureResult`

Why two methods (instead of only `reconfigure(payload)`):

- safer UX (preview/diff before write),
- enables read-only implementations where writes are not supported,
- handles "board changed outside Studio" diagnostics.

### Suggested typed models

`ReconfigureStatus`:

- `supported: bool`
- `state: "in_sync" | "drifted" | "unknown"`
- `summary: str | None`
- `diff: list[ReconfigureDiffEntry]`
- `warnings: list[str]`

`ReconfigureResult`:

- `state: "applied" | "no_change" | "failed" | "unsupported"`
- `applied_fields: list[str]`
- `skipped_fields: list[str]`
- `warnings: list[str]`
- `errors: list[str]`

`ReconfigureDiffEntry` (example):

- `path: str` (e.g. `calibration.shoulder_pan.homing_offset`)
- `studio_value: Any`
- `hardware_value: Any`
- `status: "different" | "missing_in_hardware" | "missing_in_studio"`

`strategy` request enum (initial):

- `apply_studio_to_hardware` (default)

Future strategies can be added later (e.g. `import_hardware_to_studio`) without breaking the API.

## Proposed backend API

Prefer project+robot scoped endpoints for real saved robots:

- `POST /api/projects/{project_id}/robots/{robot_id}/reconfigure-status`
- `POST /api/projects/{project_id}/robots/{robot_id}/reconfigure`

Optional catalog-scoped variants for payload-only checks:

- `POST /api/robots/catalog/{robot_type}/reconfigure-status`
- `POST /api/robots/catalog/{robot_type}/reconfigure`

### Behavior expectations

- If plugin does not support capability: return a stable "unsupported" response (or 404 with clear message; pick one and standardize).
- If runtime session owns the robot: return `409 Conflict`.
- Validate request payload through existing robot payload schema validation.
- Reconfigure endpoint performs an explicit mutate operation only when user asks.

## UI workflow

Add controls in robot update/details views:

1. **Check hardware sync** action calls `reconfigure-status`.
2. Show status badge: `In sync`, `Drifted`, or `Unknown`.
3. Show compact drift table using returned `diff`.
4. If drifted and supported, show explicit action:
   - `Apply Studio calibration to control board`.
5. On apply success, refresh status and robot data.
6. If unsupported, show non-blocking notice:
   - `This robot plugin does not support hardware sync.`

## Policy decisions for calibration specifically

- Calibration file upload updates Studio payload only.
- No implicit write to control board during create/update.
- Setup wizard skip behavior (when calibration exists) remains explicit and independent.
- Reconcile/write is a separate user action through reconfigure API.

## Rollout phases

### Phase 1: Contract + backend plumbing

- Extend plugin SDK types (`RobotCatalogDefinition`, models/callables).
- Add backend API endpoints and capability routing.
- Add busy-session guard (`409`) and common error mapping.

Done when:

- endpoints are available,
- unsupported robots handled predictably,
- test fixtures can emulate success/failure.

### Phase 2: SO101 reference implementation

- Implement status + reconfigure for built-in SO101 catalog (and/or plugin reference).
- Map calibration payload to board values and return structured diffs.

Done when:

- known drift cases are detected,
- apply operation updates board and reports result.

### Phase 3: UI integration

- Add sync-check and apply actions in robot details/update flow.
- Render drift status and compact diff table.
- Surface errors, warnings, and unsupported states.

Done when:

- user can inspect drift and apply reconfigure without manual API calls.

### Phase 4: Plugin author documentation + adoption

- Document new capability in plugin docs and architecture docs.
- Provide migration snippet + guidance for plugin maintainers.

Done when:

- plugin authors can implement capability without reading backend internals.

## Documentation updates required

1. `application/docs/robot-plugins.md`
   - New section for reconfigure capability.
   - Request/response examples.
   - Guidance on explicit user-triggered writes.

2. `application/docs/explanation/robot-plugin-architecture.md`
   - Lifecycle section for payload/hardware drift handling.
   - Capability flow diagram notes (probe vs builder vs reconfigure).

3. Handoff note for plugin authors (if behavior changes in rollout)
   - Example: `application/docs/handoff-reconfigure-capability.md`
   - Include expected UX, compatibility expectations, and test checklist.

4. Skill docs (already started in this session)
   - Keep `skills/application/studio-adding-robot-form-ui-fields/SKILL.md` aligned where UI metadata and plugin communication overlap.

## Test plan

### Plugin SDK tests

- `application/plugin/tests/test_contracts.py`
  - new callable/model typing acceptance,
  - invalid combinations,
  - validation error messages.

### Backend API tests

- Add tests for:
  - unsupported capability behavior,
  - in-sync/drifted/unknown status responses,
  - successful apply,
  - failed apply,
  - busy runtime session (`409`).

### UI tests

- Unit/integration tests for:
  - status fetching,
  - diff table rendering,
  - apply action,
  - error/unsupported states,
  - refresh after apply.

## Open design questions

1. Response strategy for unsupported robots: stable payload vs HTTP error code.
2. Whether to support reverse sync (`import_hardware_to_studio`) in v1 or defer.
3. Which robot page should host the controls first (update form vs details view).
4. Whether `reconfigure` should support dry-run mode in addition to dedicated status endpoint.

## Recommended initial decision set

- Ship both `reconfigure-status` and `reconfigure`.
- Use project+robot scoped endpoints first.
- Block when runtime session is active (`409`).
- Keep upload and board-write as separate user actions.
- Start with SO101 as reference implementation, then generalize for plugins.
