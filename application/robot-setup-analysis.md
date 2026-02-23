# Robot Setup Wizard — Refactoring Analysis

This report examines the SO101 and Trossen WidowX AI setup wizards, identifies
common patterns and duplication, evaluates whether existing abstractions
(`RobotWorker`, `RobotClient`, `commands.py`) can be leveraged, and recommends
concrete refactoring steps. It incorporates a key architectural consideration:
future SO101, LeKiwi, and Aloha Mini robots may connect via **websocket** (like
Trossen does today) rather than serial bus, making motor setup and calibration
**optional** rather than required.

---

## Table of Contents

1. [Current Architecture](#1-current-architecture)
2. [Common Features Between SO101 and Trossen Wizards](#2-common-features)
3. [Detailed Duplication Analysis](#3-duplication-analysis)
   - 3.1 Backend (setup workers)
   - 3.2 Frontend (websocket hooks)
   - 3.3 Frontend (wizard providers)
   - 3.4 Frontend (verification steps)
   - 3.5 Frontend (diagnostics steps)
   - 3.6 Frontend (layout / setup-wizard.tsx)
   - 3.7 Frontend (route component)
4. [Can `RobotWorker` / `RobotClient` Be Leveraged?](#4-robot-worker-robot-client)
5. [Should the Messaging Protocol Be Stricter?](#5-messaging-protocol)
6. [Upcoming Robots and the Websocket Future](#6-upcoming-robots)
7. [Pros and Cons of Adding Abstractions Now](#7-pros-and-cons)
8. [Recommended Refactoring Steps](#8-recommendations)

---

## 1. Current Architecture <a id="1-current-architecture"></a>

### Directory Structure

All setup wizard code is colocated under `features/robots/setup-wizard/`:

```
features/robots/setup-wizard/
  shared/                                 4 files, ~666 lines
    stepper.tsx                            70 lines — Generic step-navigation bar
    setup-robot-viewer.tsx                203 lines — 3D URDF viewer with highlight support
    use-joint-highlight.ts                168 lines — Emissive glow on URDF meshes
    setup-wizard.module.scss              225 lines — All wizard CSS classes

  so101/                                  8 files, ~2,040 lines
    wizard-provider.tsx                   265 lines — 4-step provider with skip logic
    use-setup-websocket.ts                308 lines — 11 state fields, 10 commands
    setup-wizard.tsx                      140 lines — ViewerEffects + StepBody
    diagnostics-step.tsx                  302 lines — Voltage, probe, calibration status
    motor-setup-step.tsx                  254 lines — Per-motor ID assignment
    calibration-step.tsx                  317 lines — Homing + range-of-motion recording
    verification-step.tsx                 233 lines — 3D preview + save flow
    use-calibration-animations.ts         221 lines — Centering + ROM animations

  trossen/                                6 files, ~1,194 lines
    wizard-provider.tsx                   186 lines — 2-step provider, no skip logic
    use-trossen-setup-websocket.ts        171 lines — 6 state fields, 3 commands
    setup-wizard.tsx                       22 lines — StepBody only (no ViewerEffects)
    diagnostics-step.tsx                  183 lines — IP reachability + driver config
    verification-step.tsx                 192 lines — 3D preview + save flow
    debug-panel.tsx                       440 lines — Floating debug panel with mock state

routes/robots/new.tsx                     366 lines — Unified /robots/new page
```

**Total: 19 files, ~4,266 lines** (including route component).

### Backend Workers

```
workers/robots/
  transport_worker.py                      91 lines — Base class (transport, state, run_concurrent)
  so101_setup_worker.py                   810 lines — 9 phases, 10 commands, 23 methods
  trossen_setup_worker.py                 424 lines — 3 phases, 3 commands, 13 methods

api/robot_setup.py                        101 lines — WebSocket endpoint dispatcher
```

No `BaseSetupWorker` exists. Both workers extend `TransportWorker` directly.

### Route Architecture

A single `/robots/new` route hosts all setup wizards. The `New` component renders
three providers (`RobotModelsProvider` > `RobotFormProvider` > `TrossenDebugProvider`)
and switches between Robot Info (step 1) and the robot-specific wizard. The
`wizardStarted` boolean defers mounting the wizard provider (and its websocket)
until the user clicks "Begin Setup". The 3D viewer is always mounted; it never
remounts when switching between wizard steps.

---

## 2. Common Features Between SO101 and Trossen Wizards <a id="2-common-features"></a>

Both setup wizards share the same high-level flow:

| Concern                           | SO101                                                  | Trossen                    | Shared?                                                |
|-----------------------------------|--------------------------------------------------------|----------------------------|--------------------------------------------------------|
| WebSocket lifecycle               | `TransportWorker` subclass                             | `TransportWorker` subclass | Identical base class                                   |
| Auto-run diagnostics on connect   | Voltage + motor probe                                  | IP ping + driver configure | Pattern identical, domain differs                      |
| Command loop                      | `while not _stop_requested: receive -> dispatch`       | Same                       | Character-for-character identical (except log message) |
| `_send_phase_status()`            | `{event, state, phase, message}`                       | Same                       | Identical implementation                               |
| `_send_event()`                   | `{event, **kwargs}`                                    | Same                       | Identical implementation                               |
| Position streaming for 3D preview | Phase-driven `_broadcast_loop()` at fixed FPS          | Same                       | Same structure, different FPS and tick dispatch         |
| `ping` / `pong`                   | Yes                                                    | Yes                        | Identical                                              |
| `re_probe` command                | Re-runs voltage + probe                                | Re-runs diagnostics        | Same pattern (resets phase to `CONNECTING`)             |
| Per-motor ID assignment           | Yes (6 commands)                                       | No                         | SO101-only                                             |
| Calibration (homing + recording)  | Yes (4 commands)                                       | No                         | SO101-only                                             |
| PID / motor configuration         | Yes                                                    | No                         | SO101-only                                             |
| Raw position streaming            | `CALIBRATION_*` phases -> `_broadcast_raw_positions()` | No                         | SO101-only                                             |

On the frontend:

| Concern                                     | SO101                               | Trossen                  | Shared?                  |
|---------------------------------------------|--------------------------------------|--------------------------|--------------------------|
| Stepper component                           | Props-based, generic                 | Same component           | Already extracted        |
| SCSS styles                                 | Shared module                        | Same                     | Already extracted        |
| 3D viewer (`SetupRobotViewer`)              | With highlights + animations         | Without highlights       | Already extracted        |
| Joint highlight hook                        | Yes                                  | No (passes `[]`)         | Already extracted        |
| Wizard provider (dual-context)              | State + Actions contexts             | Same pattern             | Structurally identical   |
| Navigation (`goToStep`, `goNext`, `goBack`) | With skippable steps                 | Without skip             | ~70% identical           |
| `useSyncJointState` hook                    | Hardcoded SO101 URDF path            | Uses `urdfPathForType()` | Loop body ~90% identical |
| Save flow                                   | Create robot -> save cal -> update   | Create robot only        | ~50% identical           |
| Robot details card (JSX)                    | Name, type, serial, calibration      | Name, type, IP           | ~80% identical           |

---

## 3. Detailed Duplication Analysis <a id="3-duplication-analysis"></a>

### 3.1 Backend -- Setup Workers

**Files:**
- `so101_setup_worker.py` (810 lines, 9 phases, 10 commands, 23 methods)
- `trossen_setup_worker.py` (424 lines, 3 phases, 3 commands, 13 methods)

**Identical code (extractable immediately):**

```python
# _send_phase_status -- both workers, identical implementation
async def _send_phase_status(self, message: str) -> None:
    await self.transport.send_json({
        "event": "status",
        "state": self.state.value,
        "phase": self.phase.value,
        "message": message,
    })

# _send_event -- both workers, identical implementation
async def _send_event(self, event: str, **kwargs: Any) -> None:
    await self.transport.send_json({"event": event, **kwargs})
```

**Structurally identical (same shape, different details):**

```python
# run() -- both workers follow the same lifecycle
async def run(self) -> None:
    try:
        await self.transport.connect()
        self.state = WorkerState.RUNNING
        await self._run_diagnostics()      # robot-specific
        await self.run_concurrent(          # broadcast_loop + command_loop
            self._broadcast_loop(),
            self._command_loop(),
        )
    except Exception as e:
        self.state = WorkerState.ERROR
        self.phase = <Phase>.ERROR
        self.error_message = str(e)
        await self._send_event("error", message=str(e))
    finally:
        await self._cleanup()
        await self.shutdown()

# _command_loop -- identical except for log message string
async def _command_loop(self) -> None:
    while not self._stop_requested:
        data = await self.transport.receive_command()
        if data is None:
            continue
        command = data.get("command", "")
        try:
            await self._dispatch_command(command, data)
        except Exception as e:
            await self._send_event("error", message=str(e))

# _broadcast_loop -- same structure, different FPS and tick dispatch
async def _broadcast_loop(self) -> None:
    interval = 1.0 / FPS
    while not self._stop_requested:
        t0 = time.perf_counter()
        await self._broadcast_tick()     # SO101 passes last_raw arg
        elapsed = time.perf_counter() - t0
        await asyncio.sleep(max(0.001, interval - elapsed))

# _broadcast_tick -- phase-driven (each worker defines which phases stream)
async def _broadcast_tick(self) -> None:
    if self.phase == <Phase>.VERIFICATION:
        await self._broadcast_normalized_positions()
```

**Estimation:** ~120 lines of duplicated code across the two workers (~15% of
SO101 + ~28% of Trossen). A `BaseSetupWorker` class could absorb this.

### 3.2 Frontend -- WebSocket Hooks

**Files:**
- `use-setup-websocket.ts` (308 lines, 11 state fields, 10 commands)
- `use-trossen-setup-websocket.ts` (171 lines, 6 state fields, 3 commands)

**Identical patterns:**
- `useState` + `useRef(state)` + `stateRef.current = state` -- boilerplate
- `handleMessage` callback structure: `JSON.parse -> switch(data.event) -> setState`
- Both handle `status`, `state_was_updated`, `error`, `pong` identically
- URL construction: same template, different query params
- `useWebSocket` config: identical `onOpen`, `onClose`, `onError`, `shouldReconnect`
- `reProbe`, `enterVerification`, `ping` -- same pattern in both

**Key differences:**
- SO101 has 5 additional state fields (`voltageResult`, `probeResult`,
  `motorSetupProgress`, `homingResult`, `positions`, `calibrationResult`)
- SO101 has 7 additional commands (`startMotorSetup`, `motorConnected`, etc.)
- Trossen has `diagnosticsResult` (not in SO101)
- URL query params: SO101 uses `serial_number`, Trossen uses `connection_string`
- Trossen's `onClose` sets an error if `diagnosticsResult` was still null

**Estimation:** ~100 lines of duplicated boilerplate. A generic
`useSetupWebSocket<TState, TCommands>()` hook could absorb the common
`useWebSocket` setup, message routing for shared events, and URL construction.

### 3.3 Frontend -- Wizard Providers

**Files:**
- `so101/wizard-provider.tsx` (265 lines)
- `trossen/wizard-provider.tsx` (186 lines)

**Identical patterns:**
- Dual-context pattern (`StateContext` + `ActionsContext`)
- `goToStep`, `goNext`, `goBack`, `markCompleted` -- identical logic
- `useState` for `wizardState` with `currentStep` + `completedSteps`
- `stepIndex`, `canGoNext`, `canGoBack` derivations
- Context provider nesting

**Key differences:**
- SO101 adds `skippedSteps`, `markSkipped`, `unmarkSkipped`, `visibleSteps` filtering
- SO101 adds `calibrationPhase`, `preVerifyProbeResult` domain state
- SO101 has auto-skip logic for MOTOR_SETUP step (when `probeResult.all_motors_ok`)
- Trossen has no skippable steps (`visibleSteps = TROSSEN_WIZARD_STEPS` constant)
- Trossen integrates with `TrossenDebugProvider` for mock state

**Estimation:** ~80 lines of duplicated navigation logic. A generic
`createSetupWizardProvider<Step, State>()` factory could absorb the common parts.

### 3.4 Frontend -- Verification Steps

**Files:**
- `so101/verification-step.tsx` (233 lines)
- `trossen/verification-step.tsx` (192 lines)

**Identical patterns:**
- `useSyncJointState` hook body: iterate `jointState` entries, strip `.pos`
  suffix, handle `gripper` -> `left_carriage_joint` mapping for `wxai`, convert
  `degToRad()` for all other joints. The loop is ~90% identical between files.
- `robotBody` construction -- same shape, only `connection_string` vs
  `serial_number` emphasis differs.
- "Save Robot" button + `isPending` + `isDisabled` -- identical.
- Robot Details card JSX -- ~80% identical.

**Key differences:**
- SO101 has a 3-step save chain (create robot -> save calibration -> update
  `active_calibration_id`). Trossen has a 1-step save (create robot only).
- SO101 shows calibration status in the details card.
- SO101 hardcodes `SO101_PATH` = `'/SO101/so101_new_calib.urdf'`;
  Trossen calls `urdfPathForType(robotType)`.
- Trossen uses `MOTOR_TO_URDF_JOINT` lookup table to remap motor names to URDF
  joint names (`shoulder_pan` -> `joint_0`, etc.); SO101 motor names match URDF
  joint names directly.
- Trossen has a stray `console.log` on line 61.

**Estimation:** ~50 lines of duplicated code. The `useSyncJointState` hook
(~25 lines) should be extracted into `shared/` with the URDF path and joint-name
mapping parameterized. Note: the Trossen version is more general (dynamic URDF
path + joint mapping table), so extraction should be based on that version.

### 3.5 Frontend -- Diagnostics Steps

**Files:**
- `so101/diagnostics-step.tsx` (302 lines)
- `trossen/diagnostics-step.tsx` (183 lines)

**Identical patterns:**
- Error state rendering (`classes.errorBox` with connection error)
- Loading state (spinner + status message)
- Header with refresh button (identical JSX)
- Disclosure-based section layout
- Action bar with Back + Continue buttons

**Key differences:**
- Domain content is fundamentally different:
  - SO101: voltage section, motor probe grid, calibration status
  - Trossen: IP reachability, driver configuration
- SO101 has 3 conditional action buttons (Setup Motors / Calibrate / Skip to
  Verification); Trossen has 1 (Continue to Verification)
- SO101 has `showMotorSection` conditional gating; Trossen has `ip_reachable`
  conditional gating

**Estimation:** ~40 lines of shared boilerplate (error/loading/header). The
domain content diverges enough that extracting shared JSX components would add
complexity without proportional benefit.

### 3.6 Frontend -- Layout (`setup-wizard.tsx`)

**Files:**
- `so101/setup-wizard.tsx` (140 lines)
- `trossen/setup-wizard.tsx` (22 lines)

**Identical patterns:**
- Step content switching via `currentStep ===` conditionals -- same pattern.

**Key differences:**
- SO101 exports `SO101ViewerEffects` (renderless component that drives calibration
  animations and syncs highlight state to the parent via callback) and
  `SO101StepBody`. Trossen exports only `TrossenStepBody`.
- SO101 has `useHighlights()` internal hook; Trossen has nothing.
- SO101 renders 4 steps; Trossen renders 2.
- The size difference (140 vs 22 lines) is almost entirely due to the
  `SO101ViewerEffects` component.

**Note:** The grid layout, stepper, divider, and viewer panel rendering have been
lifted to the unified `New` component in `routes/robots/new.tsx` (366 lines).
Both wizard step bodies plug into this shared layout. There is no duplicated
layout code between the two `setup-wizard.tsx` files.

### 3.7 Frontend -- Route Component

**File:** `routes/robots/new.tsx` (366 lines)

The two separate route files (`so101-setup.tsx`, `trossen-setup.tsx`) have been
eliminated. A single unified `New` component handles both robot families:

- `robotFamily(type)` derives `'so101' | 'trossen' | null` from the type string
- The stepper prepends a `'robot_info'` step to the wizard-specific steps
- `wizardStarted` state defers websocket connection until "Begin Setup"
- Highlight state is lifted to `NewRobotPage`, passed to `UnifiedViewerPanel`
- `SO101ViewerEffects` pushes highlights up via callback; Trossen has no equivalent

No refactoring needed for the route component.

---

## 4. Can `RobotWorker` / `RobotClient` Be Leveraged? <a id="4-robot-worker-robot-client"></a>

### The Verification Step -- A Strong Candidate

The verification step (streaming positions for 3D preview) does the same thing as
`RobotWorker._broadcast_loop()`:

| Aspect         | RobotWorker broadcast                                | Setup worker streaming                                |
|----------------|------------------------------------------------------|-------------------------------------------------------|
| Read positions | `client.read_state(normalize=True)`                  | `bus.sync_read(normalize=True)` / `driver.get_all_positions()` |
| Deduplication  | `if current_values != previous_values`               | SO101 does this for raw positions only                |
| Output event   | `state_was_updated` (with `timestamp`, `is_controlled`) | `state_was_updated` (without `timestamp` or `is_controlled`) |
| Frequency      | Configurable FPS (default 30)                        | Phase-driven at fixed FPS (SO101: 30Hz, Trossen: 20Hz) |

**Could we use `RobotClient` for verification?**

In theory, yes -- during verification, the setup worker only needs `read_state()`.
But there are practical blockers:

1. **SO101 `RobotClient` requires a saved robot.** `RobotClientFactory.build(robot)`
   needs a `Robot` schema object with an `id` and `active_calibration_id`. The
   setup wizard deliberately avoids persisting to the DB until "Save Robot".

2. **SO101 setup worker manages `bus.calibration` manually.** It reads calibration
   from EEPROM and assigns it to `bus.calibration` directly. The `SO101Follower`
   client calls `lerobot.robots.so101_follower.SO101Follower.connect()`, which
   triggers lerobot's interactive `input()` prompts -- the exact thing the setup
   worker was built to avoid.

3. **Trossen's `configure()` homes the robot.** The setup worker calls
   `driver.configure()` once during diagnostics. Creating a new
   `TrossenWidowXAIFollower` client would call `driver.configure()` again,
   re-homing the robot unexpectedly.

4. **Different lifecycle expectations.** `RobotClient.connect()` is designed for
   long-lived operation. Setup workers have a short lifecycle with multiple
   phases.

**Verdict:** Reusing `RobotClient` for the verification streaming step is not
practical without significant changes to how `RobotClient` implementations
manage their connection lifecycle. The setup workers' direct hardware access is
intentional and correct.

### Could We Use `commands.py`'s Pydantic Models?

The setup workers' `match/case` dispatch handles 3-10 commands. The normal
`commands.py` handles 8 commands with Pydantic validation and a discriminated
union. Could setup workers use a similar system?

**Pros:**
- Type-safe command parsing with automatic validation
- Consistent error messages for malformed commands
- Self-documenting command schema

**Cons:**
- Setup workers have a much wider and different command set (only `ping` overlaps)
- Setup commands carry different payloads (e.g., `motor_connected` has a `motor`
  field, `start_positions_stream` has `fps`)
- The current `match/case` is simple and clear -- adding Pydantic models would
  double the code for the same behavior
- Setup commands are not performance-critical (they're user-initiated, not in a
  tight loop)

**Verdict:** Not worth it today. If setup commands grow significantly (e.g., 20+
commands), Pydantic models would be justified. For 3-10 commands, `match/case`
is clearer and more maintainable.

---

## 5. Should the Messaging Protocol Be Stricter? <a id="5-messaging-protocol"></a>

### Current Inconsistencies

| Issue                                  | Normal operation                                       | Setup workers                                      |
|----------------------------------------|--------------------------------------------------------|----------------------------------------------------|
| `timestamp` field                      | Always present (via `_create_event()`)                 | Never present                                      |
| `is_controlled` in `state_was_updated` | Always present                                         | Never present                                      |
| Error event format                     | `{"event": "error", "timestamp": ..., "message": ...}` | `{"event": "error", "message": ...}`               |
| Status event shape                     | `WorkerStatus.to_json()` -> includes `config` key      | `{event, state, phase, message}` -- no `config` key |

### Assessment

These inconsistencies are **not currently causing bugs** because:
- The frontend setup hooks and the normal robot websocket hooks are completely
  separate code paths -- they never share event handlers.
- The frontend `useSyncJointState` doesn't read `timestamp` or `is_controlled`.

However, they create a **conceptual tax**: developers must remember which event
format they're dealing with depending on the context (setup vs. normal
operation).

### Recommendation

Add `timestamp` to setup worker events by using `RobotClient._create_event()`
(or extracting it as a standalone utility). This is a one-line change per event
and makes the protocol uniform. Do **not** add `is_controlled` to setup events --
it's meaningless during setup (the robot is never "controlled" in the teleoperation
sense).

---

## 6. Upcoming Robots and the Websocket Future <a id="6-upcoming-robots"></a>

### Current Connection Model

Today, the two robot families use different connection methods:

- **SO101**: USB serial -> `FeetechMotorsBus` -> direct register reads/writes.
  The setup wizard must perform per-motor ID assignment, calibration (homing +
  range recording), and PID configuration because it speaks directly to the
  motor bus.
- **Trossen**: IP network -> `TrossenArmDriver` (SDK). The driver handles
  calibration, homing, and motor configuration internally. The setup wizard
  only needs to verify IP reachability and call `driver.configure()`.

### The Websocket Future

In the near future, SO101, LeKiwi, and Aloha Mini robots may connect via a
**websocket interface** instead of serial bus. In this model:

- A companion process (running on a Pi or similar) manages the motor bus
  locally and exposes a websocket API.
- The setup wizard connects to the companion process via websocket instead of
  opening a serial port directly.
- The companion process handles motor initialization, calibration state, and
  position streaming.

This means:

| Capability              | Serial bus (today)        | Websocket (future)                      |
|-------------------------|---------------------------|-----------------------------------------|
| Motor ID assignment     | Required (setup wizard)   | **Not needed** (pre-configured)         |
| Calibration (homing)    | Required (setup wizard)   | **Optional** (may already be done)      |
| Calibration (recording) | Required (setup wizard)   | **Optional** (may already be done)      |
| PID configuration       | Required (setup wizard)   | **Not needed** (companion handles it)   |
| Diagnostics             | Voltage + motor probe     | Health check via websocket API          |
| Position streaming      | Direct bus read           | Proxied through websocket               |

**The SO101 worker's motor setup and calibration phases become opt-in capabilities,
not hard requirements.** A websocket-connected SO101 would follow a flow closer
to Trossen's: diagnostics -> optional calibration -> verification.

### Impact on Architecture

This future scenario has significant architectural implications:

1. **`BaseSetupWorker` becomes more important.** With 4+ robot types and 2
   connection methods, having a shared lifecycle base class is essential. Each
   subclass can define which phases are available based on the connection method.

2. **Phase sets must be flexible.** The wizard should support a variable number of
   steps -- not hardcoded to 4 (SO101 serial) or 2 (Trossen). The current
   skip-logic in SO101's wizard provider is a precursor to this: when
   `all_motors_ok` is true, `MOTOR_SETUP` is auto-skipped. The same pattern
   would skip motor setup and calibration for websocket-connected SO101 robots.

3. **The "complex" vs "simple" split may disappear.** Today, SO101 is "complex"
   (motor-level setup) and Trossen is "simple" (SDK-managed). With websocket-
   connected SO101, the distinction becomes connection-method-dependent rather
   than robot-type-dependent. A single robot type could have both setup paths.

4. **Motor setup and calibration become reusable modules.** Rather than being
   baked into the SO101 worker, they should be composable capabilities that any
   serial-bus-connected robot can opt into. This suggests a mixin or plugin
   architecture for the backend worker.

### LeKiwi

LeKiwi is a mobile robot base (differential drive) with an SO101 arm on top.

**Serial bus connection (today):**
- Feetech STS3215 motors (same as SO101) for the arm + wheel motors
- Same calibration flow as SO101 for arm joints; base motors may differ
- Would heavily reuse SO101's motor setup and calibration phases

**Websocket connection (future):**
- Companion Pi manages the motor bus
- Setup wizard connects via websocket
- Motor setup and calibration handled by the companion or already done
- Wizard flow: diagnostics -> optional calibration -> verification

### Aloha Mini

Aloha Mini is a bimanual robot with two SO101-like arms.

**Serial bus connection (today):**
- Feetech STS3215 motors, 12 across 2 arms
- Two USB serial connections (one per arm)
- Would run the SO101 setup flow twice (sequentially or in parallel)

**Websocket connection (future):**
- Single websocket endpoint manages both arms
- Motor setup and calibration pre-done or handled by companion
- Wizard flow: diagnostics -> optional calibration -> verification

### What This Tells Us About Abstraction

```
                BaseSetupWorker
                 /           \
        Serial path       Websocket/SDK path
    (motor-level setup)   (pre-configured)
         /    |    \            /     \
     SO101  LeKiwi  Aloha   Trossen  SO101-ws  LeKiwi-ws
```

The abstraction boundary is no longer robot-type-dependent but
**connection-method-dependent**. The key capabilities are:

| Capability         | Serial-bus robots   | Websocket/SDK robots |
|--------------------|---------------------|----------------------|
| Diagnostics        | Always              | Always               |
| Motor setup        | If motors not OK    | Never                |
| Calibration        | If not calibrated   | Optional             |
| PID configuration  | After calibration   | Never                |
| Verification       | Always              | Always               |

A `BaseSetupWorker` should provide the shared lifecycle (connect -> diagnostics
-> run_concurrent -> cleanup) and let subclasses compose which phases are active.
Motor setup and calibration should be implementable as opt-in capabilities that
any worker can include when the connection method requires them.

---

## 7. Pros and Cons of Adding Abstractions Now <a id="7-pros-and-cons"></a>

### Option A: Extract `BaseSetupWorker` + shared frontend hooks now

**Pros:**
- Eliminates ~120 lines of backend duplication, ~100 lines of frontend WS hook
  duplication, ~80 lines of provider duplication, ~50 lines of verification step
  duplication
- Establishes patterns before LeKiwi/Aloha Mini add more complexity
- Prepares for the websocket future -- the base class is connection-agnostic
- Makes it clear what's robot-specific vs. framework-level
- Reduces the surface area for bugs (one implementation of the command loop,
  one implementation of the broadcast loop, etc.)

**Cons:**
- Premature if the websocket architecture changes the lifecycle fundamentally
- Adds indirection -- developers must understand the base class to work on any
  setup worker
- The current duplication is manageable (2 robots, well-separated files)
- Risk of over-engineering: the "shared" parts may diverge as the websocket
  protocol evolves

### Option B: Wait until LeKiwi integration, then refactor

**Pros:**
- Three data points (SO101, Trossen, LeKiwi) give a much better understanding
  of what's truly common vs. accidental
- The websocket architecture may be clearer by then
- No wasted effort if LeKiwi's requirements invalidate the current pattern
- Current code works and is easy to understand in isolation

**Cons:**
- LeKiwi integration will require either: (a) more copy-paste from SO101, or
  (b) refactoring under time pressure
- The third copy will make the codebase harder to maintain
- Trossen's code already demonstrates that the patterns are stable

### Recommendation

**Option A, but scoped conservatively.** Extract only the things that are
character-for-character identical or near-identical. Do not try to abstract over
the differences (e.g., don't create a generic diagnostics step component, don't
build a phase-composition system). The specific extractions are listed in
Section 8.

The websocket future does not change the immediate recommendation -- the
lifecycle, command loop, broadcast loop, and helper methods are equally needed
for serial-bus and websocket-connected robots. What it does change is the
framing: the `BaseSetupWorker` should not assume a fixed phase set, and the
frontend wizard navigation should support variable step counts from day one.

---

## 8. Recommended Refactoring Steps <a id="8-recommendations"></a>

Ordered by impact and safety. Each step is independently committable.

### Step 1: Extract `BaseSetupWorker` (backend)

Create `workers/robots/base_setup_worker.py`:

```python
class BaseSetupWorker(TransportWorker):
    """Base class for robot setup workers.

    Provides:
    - run() lifecycle: connect -> diagnostics -> run_concurrent(broadcast, command) -> cleanup
    - _command_loop() with dispatch to subclass
    - _broadcast_loop() with phase-driven _broadcast_tick()
    - _send_phase_status(), _send_event()

    Subclasses must implement:
    - _run_diagnostics()
    - _dispatch_command(command, data)
    - _cleanup()
    - _broadcast_tick() (reads positions based on current phase)
    """
```

**Lines saved:** ~120 across both workers.
**Risk:** Low -- the extracted code is identical in both workers.
**Websocket-ready:** The base class is connection-agnostic. A websocket-connected
SO101 worker would be another subclass with a simpler phase set.

### Step 2: Extract `useSyncJointState` (frontend)

Move to `shared/use-sync-joint-state.ts`:

```ts
// Accepts jointState + robotType, resolves URDF path via urdfPathForType(),
// supports an optional jointNameMapping for robots where motor names don't
// match URDF joint names (e.g., Trossen's MOTOR_TO_URDF_JOINT table).
// SO101's hardcoded path becomes urdfPathForType('SO101_Follower').
```

The Trossen version is more general (dynamic URDF path, joint mapping table),
so extraction should be based on that version. SO101 would pass `undefined` for
the mapping (motor names match URDF joints directly).

**Lines saved:** ~25 per verification step (~50 total).
**Risk:** Very low -- the loop body is near-identical.

### Step 3: Add `timestamp` to setup worker events (backend)

Extract `_create_event()` from `RobotClient` into a standalone utility (or just
call `datetime.now().timestamp()` in `_send_event()`). Apply to both setup
workers (or to `BaseSetupWorker._send_event()` after Step 1).

**Lines saved:** 0 (this is about consistency, not deduplication).
**Risk:** Very low -- frontend ignores `timestamp` today.

### Step 4: Extract shared wizard navigation logic (frontend)

Create a `createWizardNavigation<S>()` utility or custom hook that provides
`goToStep`, `goNext`, `goBack`, `markCompleted`, `stepIndex`, `canGoNext`,
`canGoBack`. SO101's `markSkipped`/`unmarkSkipped`/`visibleSteps` filtering
would be opt-in.

This step becomes especially important for the websocket future: a websocket-
connected SO101 would need to dynamically show/hide motor setup and calibration
steps based on whether the companion process reports them as necessary. The
skip-logic in SO101's current provider is already a primitive version of this.

**Lines saved:** ~50 per provider (~100 total).
**Risk:** Medium -- the SO101 provider's skip logic adds complexity. This should
be done carefully to avoid making the simple case (Trossen) harder to understand.

### Step 5 (optional): Generic `SetupWizardLayout` component

A shared layout component that renders step content and accepts a
`renderStep(currentStep)` callback. Both wizards already share the outer
Grid/Stepper/Viewer layout via the unified `New` component, so this would only
cover the inner step switching.

**Lines saved:** ~20 per layout (~40 total).
**Risk:** Low -- now that the outer layout is unified, the inner switching is
trivial.

### What NOT to Extract

- **Diagnostics steps:** Domain content differs too much. Extracting shared JSX
  (error box, loading spinner, header) saves ~40 lines but adds indirection
  for components that are simple and readable as-is.
- **`commands.py`-style Pydantic models for setup commands:** Not worth the
  overhead for 3-10 commands.
- **A unified websocket hook:** The SO101 hook has 11 state fields and 10
  commands; the Trossen hook has 6 and 3. A generic hook would need extensive
  parameterization. Instead, extract only the boilerplate (WS connection setup,
  shared event handlers) and let each robot define its own state shape.
- **A phase-composition/mixin system for the backend:** Premature. Wait until
  the websocket architecture is finalized before building a plugin system for
  motor setup and calibration capabilities.

---

## Summary

| What                        | Duplication     | Recommendation                | Risk     |
|-----------------------------|-----------------|-------------------------------|----------|
| Backend lifecycle + helpers | ~120 lines      | Extract `BaseSetupWorker`     | Low      |
| `useSyncJointState`         | ~50 lines       | Extract to shared             | Very low |
| Event `timestamp`           | Consistency gap | Add to setup events           | Very low |
| Wizard navigation           | ~100 lines      | Extract `createWizardNavigation` | Medium |
| Layout inner switching      | ~40 lines       | Extract (optional)            | Low      |
| Diagnostics steps           | ~40 lines       | Leave as-is                   | --       |
| WS hook boilerplate         | ~100 lines      | Extract connection setup only | Medium   |
| Pydantic setup commands     | 0 duplication   | Leave as-is                   | --       |
| Phase composition system    | N/A             | Wait for websocket arch       | --       |

Total recoverable duplication: **~410 lines** across backend and frontend, of
which **~220 lines** (Steps 1-3) can be extracted with low risk. The remaining
**~190 lines** (Steps 4-5, WS boilerplate) require more careful design and
benefit from the third robot as a data point.

### Websocket Impact on Prioritization

The prospect of websocket-connected SO101/LeKiwi/Aloha changes the relative
priority of extractions:

| Step | Without websocket consideration | With websocket consideration |
|------|--------------------------------|------------------------------|
| `BaseSetupWorker` | Nice-to-have, deduplication | **Essential** -- new workers will multiply |
| `useSyncJointState` | Low-hanging fruit | Same -- connection-agnostic |
| Wizard navigation | Wait for 3rd robot | **Higher priority** -- dynamic step sets needed |
| Phase composition | Not considered | **Future work** -- after websocket arch is clear |

The `BaseSetupWorker` extraction (Step 1) is the highest-value change regardless
of whether the websocket future materializes. It should be done first.
