# Live robot connection during setup

## Status

- [x] Investigate the existing form, plugin schema, observation stream, calibration flows, and driver lifecycle.
- [x] Confirm UX choices with the user.
- [x] Write this implementation plan in the repository root.
- [x] Check the installed/pinned PhysicalAI public lifecycle API and reproduce its cleanup limitation without hardware.
- [x] Replace temporary shared ownership with a direct driver and public hardware claims (user-approved direction).
- [x] Implement draft validation and live observation backend support.
- [x] Implement the form-scoped connection lifecycle.
- [x] Add live preview, read-only joint states, and error UI.
- [x] Integrate verified and offline saving, plus explicit release before calibration navigation.
- [ ] Publish/integrate the public device-claim helper and update Studio's runtime dependency.
- [ ] Complete automated tests and hardware acceptance checks.
- [ ] Follow-up: optional plugin SDK visualization metadata (parked).

The initial implementation is committed in focused Conventional Commits across Studio and PhysicalAI. Existing unrelated working-tree changes are preserved. Hardware acceptance and the runtime dependency integration remain pending.

### Implementation checkpoint: direct driver selected instead of scoped shared ownership

The backend environment resolves PhysicalAI to `0.3.1.dev25+gb4cc2cf`, matching pinned revision `b4cc2cf569c24f92424001dde7aca4081096ebd1`; installed SO101 and Trossen plugins are both `0.2.0`.

A hardware-free probe mocked transport, owner startup, and first observation attachment while exercising the installed SharedRobot connect/disconnect implementation:

- Normal disconnect: owner started once, subscriber closed once, owner stop/wait never called.
- Failed first-state attachment: owner started once, subscriber closed once, owner stop/wait never called.

The user selected direct-driver lifecycle after reviewing this limitation. The scoped SharedRobot experiment has been removed; ordinary shared subscribers and owner behavior are unchanged.

The live endpoint now calls `RobotClientFactory.build_robot_driver`, claims its `device_ids`, connects/reads directly, and acknowledges disconnect only after driver cleanup and claim release. A small public `physicalai.robot.transport.claim_devices` context manager exposes the existing device locks; no new owner/process/session framework is introduced.

The public claim helper is committed in PhysicalAI as `7184e2f` (`feat(robot): add device claims for direct connections`). Studio's runtime dependency pin has not been changed; integrate the helper with a compatible revision that keeps device ownership locks enabled before hardware acceptance.

**Dependency integration remains necessary:** the installed runtime lacks this new public helper and also lacks `RobotCalibrationError`, which existing Studio error translation already imports. Backend verification uses the PhysicalAI reference source through `PYTHONPATH`. The reference checkout contains a committed temporary device-lock bypass (`de330ec`); that user experiment is preserved and must not be included in a production integration that promises exclusivity. Guard tests confirm collision with the shared lock primitive, not the currently bypassed owner loop.

Targeted UI tests, backend fake-driver tests, guard tests, lint and build checks are recorded during implementation. Full TypeScript verification has existing/schema-regeneration-related robot fixture and input/output union errors outside the new feature; do not hide them or refactor unrelated workflows just to make this experiment pass.

### Verification results

- Targeted form/schema/live-preview UI tests: **54 passed**.
- Draft connection plus existing saved-observation backend tests: **20 passed**, using reference runtime/plugin sources and fake hardware.
- Public device claim and existing lock tests: **25 passed**.
- Changed UI/backend/runtime-helper lint: passed; production UI build: passed; targeted circular dependency checks: passed.
- Full UI suite: 575 passed, one skipped, one remote-trainer test timed out. That remote-trainer test file passed all eight tests when rerun separately; no unrelated test code was changed.
- Full TypeScript check remains blocked by existing robot fixtures and regenerated input/output union incompatibilities outside the new live-connect code. The new feature and its mocks produce no TypeScript errors.
- OpenAPI JSON/types were regenerated from a hardware-free backend server. These generated files are git-ignored by the repository; they were not hand-edited or force-added.
- No hardware was connected. The standard installed backend environment is older than current Studio source (missing `RobotCalibrationError` and `SO101SetupSession`). A production-compatible runtime/plugin update, including the new claim helper and enabled device locks, is still required before hardware acceptance.

## Goal

Let users test the actual robot configuration before saving it. Once they explicitly enable Live connect, automatically connect when the configuration is complete and valid, synchronize the preview with measured state, and display joint readings. Make failures understandable and recoverable.

Keep the experiment small and removable: one form-scoped UI feature, one draft observation endpoint, small integration points, and reuse of existing connection/building/error/rendering infrastructure. Do not create a general-purpose robot setup framework.

## Confirmed product decisions

1. **Consent, then automatic connection.** Live connect starts disabled on each form visit. Enabling it requires a clear warning that driver connection/disconnection may move the arm or change torque. After consent, valid configuration changes trigger automatic connection.
2. **Explicit offline-save fallback.** Normal Add/Update requires a healthy connection for the current configuration. A deliberate Save without verification action can bypass hardware verification, but not payload validation. The warning must distinguish untested, failed, and disconnected configurations.
3. **Read-only joint displays.** Sliders show measured values only. This feature sends no movement commands and exposes no torque controls.
4. **Optional SDK visualization metadata is desirable but parked.** Record the follow-up below; do not make the initial experiment depend on SDK/plugin releases.

## Findings and existing reuse points

### UI

- `application/ui/src/features/robots/robot-form/provider.tsx` owns type, name, and payload. `useRobotFormBody` currently checks only type and nonempty name.
- `robot-form/create-form.tsx` routes uncalibrated SO101 robots to setup, and robots offering zero calibration to their calibration wizard. Its UUID is currently regenerated during rendering; the new flow needs a stable draft ID.
- `robot-form/update-form.tsx` saves without hardware verification.
- `robot-form/preview.tsx` renders a static `RobotViewer` for the selected type.
- `controller/robot-viewer.tsx` already accepts feature names/values and uses catalog joint mappings.
- `use-robot-observations.ts` supplies a saved-robot websocket hook and event parser. Reuse the event shape and appropriate parsing utilities, not its unconditional reconnect policy.
- `controller/joint-controls.tsx` has read-only sliders, but is coupled to persisted robot route context. Its range calculation also matches observation names incorrectly and converts already-converted limits again. Extract only a small presentational component if that is simpler than a form-local display; do not adopt the existing calculation unchanged.
- `robot-schema/schema-validation.ts` only checks UI-required overrides and is not integrated into submit gating. Schema readiness must include standard required fields, overrides, nested objects, and defaults.
- `routes/robots/new-layout.tsx` keeps `RobotFormProvider` mounted across form and calibration routes. A live hardware connection must not share that lifetime.

### Backend and plugins

- `application/backend/src/api/robot_observations.py` streams observations from a `SharedRobot` without starting a runtime/action loop. It reports connected state only after the first read and has cancellation-shielded subscriber cleanup.
- `robots/robot_client_factory.py` already calls any installed catalog robot's builder and wraps its exported recipe in `SharedRobot`.
- `robots/shared_robot_errors.py` translates structured PhysicalAI errors, including device ownership and calibration failures.
- `api/robot_catalog.py` exposes payload JSON Schema, discovery, identify, and online probes. Probe success is not driver connection success; probes are optional.
- `workers/robots/robot_zero_calibration_worker.py` demonstrates unsaved payload validation and direct-driver observation streaming, but invokes calibration callbacks. Follow its direct-driver approach without reusing the calibration worker itself. Existing direct probes/wizards do not acquire the new claim helper; this feature explicitly releases before invoking them rather than claiming to retrofit all existing hardware access.
- The SO101 payload uses a Python model validator requiring either a port or serial number. JSON Schema alone cannot express all plugin validation rules.
- Asset-less plugins are valid, but the catalog currently advertises a URDF URL even when no asset exists. The initial feature should make missing assets explicit rather than render a failed/blank model.

### Safety and lifecycle

- In the PhysicalAI reference source, WidowX AI homes on both connect and disconnect. SO101 follower connect enables torque. Read-only observation does not imply motion-free lifecycle.
- SharedRobot enforces physical device ownership across different owner names, including composite device IDs. Do not replace it with an unguarded plain driver.
- `SharedRobot.disconnect()` detaches a subscriber; the owner exits after its idle timeout. Closing a websocket is not proof that hardware is released.
- SharedRobot attachment does not verify the entire requested driver recipe against an existing owner's configuration. Reusing an owner name across changed payloads could falsely verify stale settings.
- SharedRobot can return cached observations when fresh frames stop. Websocket traffic alone is not evidence of a healthy device; use observation timestamps/freshness.
- These findings must be checked against the backend's pinned PhysicalAI revision and resolved plugin versions, not just the reference checkout.

## Proposed UX

The right-hand panel contains the robot preview, a compact Live connect status/control, and an expandable joint-state list.

Suggested states:

- **Off:** static preview; explanation that enabling live connection can move or energize the robot.
- **Waiting for configuration:** list missing fields or show relevant field errors; no hardware attempt.
- **Connecting:** pending indicator and disabled normal save.
- **Connected:** live preview/readings and enabled normal save for the matching payload.
- **Releasing connection:** pending indicator during disconnect/reconfiguration/handoff.
- **Connection failed / lost:** actionable inline error, Retry, and explicit offline-save option.
- **Setup required:** explain why setup/calibration must be completed; keep its entry action available.

Show “No 3D model supplied by this plugin” for asset-less robots; continue to show connection state and joint values. A missing model or mapping must not block a successful connection or save.

After a failure, do not continuously retry the same configuration. Retry explicitly, or attempt again after a meaningful payload change. Changes to the robot name alone must not reconnect hardware. Reset consent when the selected robot type changes; consent does not persist across form visits.

## Architecture and boundaries

### 1. Authoritative draft validation

Add a small catalog payload-validation operation, provisionally `POST /api/robots/catalog/{robot_type}/validate`:

- Accept the plugin payload as a generic JSON object rather than relying on a generated closed union of installed robot types.
- Validate with the installed Pydantic payload model, applying defaults and custom validators.
- Return validated/normalized payload data; surface errors with field locations using the existing API validation conventions.
- Enforce Studio-required UI fields consistently with the form's schema metadata. Do not silently weaken Pydantic-required fields.
- Never build/connect hardware during this operation.

The frontend performs a cheap schema completeness check first, then debounced server validation. Backend validation remains authoritative; do not implement a second complete JSON Schema/Pydantic validator in TypeScript or add a validation library just for this feature.

Validation can run without Live connect consent because it does not touch hardware. Offline save uses the same validation, including for unchanged configurations and plugins with custom validators. Name validation remains separate from connection readiness.

### 2. Form-scoped draft observation websocket

Add a dedicated endpoint, provisionally `/api/projects/{project_id}/robots/draft/observations/ws`, adjacent to the existing observation endpoint:

- Accept a validated-at-source but revalidated-on-server draft configuration in a websocket command/handshake, not in a URL query string.
- Reuse `build_robot_driver`, observation conversion, and structured error translation. Do not wrap the setup driver in SharedRobot.
- Verify project existence and validate command shape/configuration before hardware access.
- Never persist a temporary robot, start a RuntimeSession, call `send_action`, invoke identify, or invoke calibration.
- Preserve the existing observation event shape; add request/generation correlation and connection lifecycle events where needed.
- Declare hardware connected only after a valid, finite, fresh observation matching joint names/dimensions.
- Treat stale observations and read failures as connection loss, not as successful continuous streaming.
- Serialize configuration replacement and cleanup. Monitor client disconnect while a blocking connect runs; a late completion must still be cleaned up. Cancelling `asyncio.to_thread` alone does not stop the underlying driver operation.
- Bound connection/freshness/release waits, preserve actionable errors, and shield cleanup during cancellation.

Factor only genuinely shared streaming/error helpers out of the existing endpoint. Keep the saved-robot endpoint behavior unchanged unless a directly required fix is covered by regression tests.

### 3. Direct ownership and acknowledged release

Each websocket owns one direct driver built from the validated draft. Acquire all canonical `device_ids` through the public claim helper before `connect()`. A conflicting claim rejects the attempt before driver I/O; there is no attachment to or takeover of another runtime. Claims roll back atomically for composite robots.

Retain the driver before connection so partial failures can be disconnected. Serialize connect/read/disconnect off the event loop, and shield cleanup against browser/backend cancellation. A late connection completion must still be cleaned up.

Keep claims until `disconnect()` succeeds. If cleanup fails repeatedly, retain the session/claim until restart and report that safe release is unconfirmed. Do not release the lock merely because a timeout elapsed. Direct driver calls cannot be forcibly interrupted by cancelling an async task: the UI reports connection/release timeouts without pretending the underlying operation stopped.

Use request/generation IDs to reject stale observations and a `disconnected` acknowledgement to serialize reconfiguration, normal save, and calibration handoff. Keep the persisted draft UUID stable independently of these request IDs. No idle-window delays or private owner handles are used.

### 4. Isolated frontend lifecycle

Add a small `robot-form/live-connect/` feature containing the connection hook/provider and panel (co-located tests). Mount it around the create/edit form-plus-preview layout only, inside the existing form/model providers. Do not mount hardware effects in the persistent new-robot layout.

Responsibilities:

- Derive readiness from the selected schema and payload.
- Debounce settled payload edits, initially around 500 ms; keep the interval local and easy to tune.
- Validate the payload and automatically connect only after consent.
- Track a configuration key/generation independently from the name and temporary transport identity.
- Immediately invalidate verified state when the payload/type changes, before the debounce finishes.
- Ignore late validation responses, socket events, and connect completions belonging to superseded generations.
- Keep one active attempt; release the prior connection before another hardware attempt.
- Expose status, latest joint observations, field/connection errors, Retry, disconnect/handoff, and a current-payload verification predicate.
- Use existing TanStack Query/API clients for HTTP and `react-use-websocket` for websocket transport. No new global state or data-fetching library.

Keep high-frequency observations in this feature's context rather than in the form-data provider, to avoid rerendering every input at display frame rate. Start at a bounded display rate (15–30 FPS), not an unbounded stream.

### 5. Joint presentation and model synchronization

- Reuse RobotViewer and catalog joint mappings where their coordinate convention is known to be compatible.
- Resolve ranges through `joint_map`, not by comparing a `.pos` observation key directly to a URDF joint name.
- Convert angular limits exactly once; handle revolute/continuous and prismatic joints separately, including one-to-many mappings and grippers.
- Sliders are read-only. Unknown ranges get numeric values rather than invented limits. Preserve and flag out-of-range measured values instead of silently presenting a clamped value as the reading.
- Show finite real observations only. Clear or explicitly mark stale readings when disconnected; never present default zero poses as live measurements.
- Keep first-party presentation conversions in a small replaceable form-local adapter if needed. Verify known SO101/WidowX conventions with tests and hardware. Normalized SO101 positions, raw ticks, and gripper values must not be labeled as degrees or passed blindly to degree conversion.
- Until the parked SDK metadata exists, unknown plugin units/conversions receive raw readings and an explicit visualization limitation. Do not claim accurate generic URDF synchronization where the plugin contract cannot establish it.
- Check cached mutable URDF model behavior: switching configuration or disconnecting must not leave an old live pose mislabeled as a new/default pose. Avoid a repository-wide model-cache refactor.

## Saving and calibration integration

### Normal saving

- Gate Add/Update and the submit handler on valid form data plus a healthy connection matching the current payload.
- Take an immutable validated snapshot when submitting; prevent edits or concurrent submissions from changing what is being saved.
- Release/handoff the temporary connection using the tested lifecycle before the destination opens a conflicting connection. Keep the user informed while waiting.
- Show save failures inline, retain the form, and support retry. Do not confuse a persistence failure with a connection failure.
- A verified connection is a point-in-time test, not a guarantee of future runtime success. Keep persisted robot/API contracts unchanged; do not add a permanent “verified” flag.

### Offline saving

- Offer a separate Save without verification action and confirmation warning, not an automatic fallback after a failed attempt.
- Require valid name/payload even offline.
- Explain whether the configuration was not tested, could not connect, or lost its connection.
- Do not bypass required calibration data or reinterpret offline saving as successful calibration. Preserve existing setup routing; only configurations already valid for persistence may save directly.

### Setup/calibration

- Keep Begin Setup / Begin Calibration available when connection cannot succeed until setup is complete. A connection-required save rule must not create a calibration deadlock.
- Uncalibrated SO101 should continue through its existing wizard; do not stream raw ticks as a calibrated arm preview or run calibration implicitly during Live connect.
- Stop/release live preview before entering an exclusive setup/calibration flow. Avoid duplicate Identify/probe hardware access while Live connect is connecting/connected/releasing; disable Identify with a clear explanation or release explicitly before running it.
- Existing wizard connections remain wizard-owned. Returning to the form creates a fresh form-scoped live lifecycle.
- Keep calibration workflows out of the main refactor. Change wizard save/handoff checks only where the temporary connection introduces a real integration requirement.

## Implementation sequence

### A. Prove direct lifecycle and write contract tests

- Confirm pinned dependency behavior and direct-driver claim/release approach.
- Write backend tests for draft validation, first observation, stale state, ownership conflict, configuration replacement, client disconnect during connection, and cleanup.
- Integrate the public claim helper into a production-compatible runtime version; keep shared owner behavior unchanged.

### B. Backend vertical slice

- Add payload validation and draft observation routes with small shared helpers.
- Add explicit no-asset catalog representation (for example nullable `urdf_path`) and ensure existing consumers handle it.
- Register/document endpoints and regenerate OpenAPI plus UI types using the repository workflow; never hand-edit generated artifacts.

### C. UI vertical slice

- Add the form-scoped Live connect provider/hook and consent/status UI.
- Add schema completeness, debounced authoritative validation, stable draft ID, and generation-safe lifecycle.
- Wire preview observations and read-only joint readings with honest unit/range fallbacks.

### D. Save and handoff

- Gate normal create/update, add confirmed offline save, and surface save errors.
- Preserve setup/calibration entry; implement tested release/handoff and Identify exclusion.
- Verify route lifetime and cleanup on navigation/unmount.

### E. Acceptance and refinement

- Run targeted tests first, then required backend/UI lint, type, unit, cyclic-dependency, and build checks.
- Perform hardware checks only with explicit operator consent and a clear workspace.
- Review new code for removal boundaries and unnecessary abstraction; keep unrelated cleanup out of the change.

## Test and acceptance checklist

### Automated

- [ ] No hardware I/O before Live connect consent.
- [ ] Required/default/nested/UI-required fields and custom Pydantic validators are honored.
- [ ] `false` and `0` are not treated as missing; unknown schemas/validation failures fail safely.
- [ ] One attempt after settled edits; name-only edits do not reconnect.
- [ ] Old payload success cannot enable saving a new payload, including while validation is pending.
- [ ] Retry is explicit for the same failed payload; no infinite reconnect loop.
- [ ] Connected requires valid fresh observation, not socket-open state or cached traffic.
- [ ] Connect/read/permission/ownership/calibration/transport errors produce actionable UI.
- [ ] Partial connection failure, late connection completion, unmount, and backend cancellation clean up.
- [ ] Different setup generations cannot reuse stale configuration or begin I/O before their prior direct connection releases its claim.
- [ ] Another runtime/robot owner is never stopped or reconfigured.
- [ ] Missing asset, unknown units, unknown range, continuous/prismatic joints, grippers, and one-to-many mappings render honestly.
- [ ] Normal save is gated both visually and in its handler; offline save still validates and requires confirmation.
- [ ] Persistence failure retains data and allows retry without falsely claiming connection failure.
- [ ] Setup/calibration entry remains possible, with no simultaneous preview/Identify/calibration connections.
- [ ] Saved-robot observation behavior remains covered by existing regression tests.
- [ ] Fake drivers assert no `send_action`, identify, calibration, or runtime-loop calls from live preview.

### Hardware/operator acceptance

- [ ] Calibrated SO101 follower and leader: consent, live state, unit handling, unplug/retry, and save.
- [ ] Uncalibrated SO101: existing setup remains accessible and no misleading calibrated preview appears.
- [ ] WidowX AI: warning covers both connection and disconnection homing; no repeated unintended home cycles during editing.
- [ ] Composite robot: all constituent devices are claimed; partial failure releases acquired resources.
- [ ] Installed plugin with no probe and/or no URDF: connection still works and limitations are clear.
- [ ] Active runtime using the same device: preview refuses safely and provides useful busy guidance.
- [ ] Cancel, navigate away, edit configuration, save, and enter calibration: hardware handoff completes within documented bounds.

## Parked follow-up: optional SDK visualization metadata

The present Robot protocol supplies joint positions without a universal unit declaration. RobotAsset supplies name mappings only. Accurate plugin-agnostic visualization therefore needs an additive contract owned by plugins, not more robot-type conditions in Studio.

Future work in `/home/mark/projects/geti/physicalai/packages/physicalai-studio-plugin/` should define optional per-joint presentation metadata:

- Source units and optional display limits.
- Conversion to each mapped URDF coordinate in radians/meters, including scale/offset and one-to-many targets.
- Configuration-dependent metadata for normalized/calibrated SO101 payloads.
- Safe absence behavior: older plugins retain raw readings without guessed conversions.

Prefer a small optional asset/presentation metadata provider evaluated from the validated robot descriptor, returning serializable data. Final naming/shape remains intentionally undecided until this follow-up is resumed. Do not change the RobotObservation/action semantics, add a new robot-form field kind, or require frontend code from plugins.

Implement and test the metadata in SO101 and WidowX AI plugins first, expose it through Studio's catalog/connection contract, replace the experimental form-local adapters, and document plugin adoption in `application/docs/robot-plugins.md` and the SDK reference.

## Scope and removability guardrails

- No DB migrations, temporary persisted robots, or persistent verification state.
- No teleoperation, commanded joint sliders, torque UI, implicit calibration, or passive-driver rewrite.
- No new process/session framework, global store, generic plugin UI system, or broad schema-form refactor.
- Keep live hardware lifetime isolated from form-data and calibration-route lifetime.
- Prefer one small backend module and one UI feature folder, with only necessary route, submit, factory/helper, catalog, documentation, and generated-contract edits.
- Keep timing constants and first-party conversion adapters local and replaceable.
- Tests and safety checks are part of the minimal implementation, not optional follow-ups.
- Removing the experiment should remove its UI wrapper/folder and draft routes, restore ordinary submit integration, and leave persistence, existing observations, and calibration architecture intact. The public claim helper remains a small independent runtime utility.
