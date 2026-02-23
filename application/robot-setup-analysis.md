# Robot Setup Wizard — Refactoring Analysis

This report examines the SO101 and Trossen WidowX AI setup wizards, identifies
common patterns and duplication, evaluates whether existing abstractions
(`RobotWorker`, `RobotClient`, `commands.py`) can be leveraged, and recommends
concrete refactoring steps. It also considers how upcoming LeKiwi and Aloha Mini
integrations would fit into the current and proposed architecture.

---

## Table of Contents

1. [Common Features Between SO101 and Trossen Wizards](#1-common-features)
2. [Detailed Duplication Analysis](#2-duplication-analysis)
   - 2.1 Backend (setup workers)
   - 2.2 Frontend (websocket hooks)
   - 2.3 Frontend (wizard providers)
   - 2.4 Frontend (verification steps)
   - 2.5 Frontend (diagnostics steps)
   - 2.6 Frontend (layout / setup-wizard.tsx)
   - 2.7 Frontend (route components)
3. [Can `RobotWorker` / `RobotClient` Be Leveraged?](#3-robot-worker-robot-client)
4. [Should the Messaging Protocol Be Stricter?](#4-messaging-protocol)
5. [Upcoming Robots: LeKiwi and Aloha Mini](#5-upcoming-robots)
6. [Pros and Cons of Adding Abstractions Now](#6-pros-and-cons)
7. [Recommended Refactoring Steps](#7-recommendations)

---

## 1. Common Features Between SO101 and Trossen Wizards <a id="1-common-features"></a>

Both setup wizards share the same high-level flow:

| Concern                           | SO101                                                               | Trossen                    | Shared?                                                                |
|-----------------------------------|---------------------------------------------------------------------|----------------------------|------------------------------------------------------------------------|
| WebSocket lifecycle               | `TransportWorker` subclass                                          | `TransportWorker` subclass | Identical base class                                                   |
| Auto-run diagnostics on connect   | Voltage + motor probe                                               | IP ping + driver configure | Pattern identical, domain differs                                      |
| Command loop                      | `while not _stop_requested: receive → dispatch`                     | Same                       | Character-for-character identical                                      |
| `_send_phase_status()`            | `{event, state, phase, message}`                                    | Same                       | Identical implementation                                               |
| `_send_event()`                   | `{event, **kwargs}`                                                 | Same                       | Identical implementation                                               |
| `_spawn_task()`                   | Background task set                                                 | Same                       | ~~Identical implementation~~ Removed — replaced by `_broadcast_loop()` |
| Position streaming for 3D preview | Phase-driven `_broadcast_loop()` → `state_was_updated`              | Same                       | Same event format                                                      |
| Streaming guard                   | ~~`if self._streaming: return`~~ Phase enum controls broadcast tick | Same                       | Identical                                                              |
| `ping` / `pong`                   | Yes                                                                 | Yes                        | Identical                                                              |
| `re_probe` command                | Re-runs voltage + probe                                             | Re-runs diagnostics        | Same pattern (resets phase to `CONNECTING`)                            |
| Per-motor ID assignment           | Yes (6 commands)                                                    | No                         | SO101-only                                                             |
| Calibration (homing + recording)  | Yes (4 commands)                                                    | No                         | SO101-only                                                             |
| PID / motor configuration         | Yes                                                                 | No                         | SO101-only                                                             |
| Raw position streaming            | `start_positions_stream` / `stop_positions_stream`                  | No                         | SO101-only                                                             |

On the frontend:

| Concern                                     | SO101                                                       | Trossen                  | Shared?                       |
|---------------------------------------------|-------------------------------------------------------------|--------------------------|-------------------------------|
| Stepper component                           | Props-based, generic                                        | Same component           | Already extracted             |
| SCSS styles                                 | Shared module                                               | Same                     | Already extracted             |
| 3D viewer (`SetupRobotViewer`)              | With highlights + animations                                | Without highlights       | Already extracted             |
| Joint highlight hook                        | Yes                                                         | No (passes `[]`)         | Already extracted             |
| Wizard provider (dual-context)              | State + Actions contexts                                    | Same pattern             | Structurally identical        |
| Navigation (`goToStep`, `goNext`, `goBack`) | With skippable steps                                        | Without skip             | ~70% identical                |
| `useSyncJointState` hook                    | Hardcoded SO101 URDF path                                   | Uses `urdfPathForType()` | Loop body 100% identical      |
| Stream on mount / stop on unmount           | ~~`useEffect` + `mountRef`~~ Removed — backend phase-driven | Same                     | ~~100% identical~~ Eliminated |
| Save flow                                   | Create robot → save calibration → update robot              | Create robot only        | ~50% identical                |
| Robot details card (JSX)                    | Name, type, serial, calibration status                      | Name, type, IP           | ~80% identical                |

---

## 2. Detailed Duplication Analysis <a id="2-duplication-analysis"></a>

### 2.1 Backend — Setup Workers

**Files:**
- `so101_setup_worker.py` (813 lines)
- `trossen_setup_worker.py` (368 lines)

**Identical code (extractable immediately):**

```python
# _send_phase_status — both workers, identical implementation
async def _send_phase_status(self, message: str) -> None:
    await self.transport.send_json({
        "event": "status",
        "state": self.state.value,
        "phase": self.phase.value,
        "message": message,
    })

# _send_event — both workers, identical implementation
async def _send_event(self, event: str, **kwargs: Any) -> None:
    await self.transport.send_json({"event": event, **kwargs})

# _broadcast_loop — both workers, identical implementation
async def _broadcast_loop(self) -> None:
    interval = 1.0 / self.FPS
    while not self._stop_requested:
        t0 = time.perf_counter()
        await self._broadcast_tick()
        elapsed = time.perf_counter() - t0
        await asyncio.sleep(max(0.0, interval - elapsed))
```

**Structurally identical (same shape, different details):**

```python
# run() — both workers follow the same lifecycle
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

# _command_loop — identical in both
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

# _broadcast_tick — phase-driven (each worker defines which phases stream)
async def _broadcast_tick(self) -> None:
    if self.phase == <Phase>.VERIFICATION:
        await self._broadcast_normalized_positions()
```

**Estimation:** ~120 lines of duplicated code across the two workers (~15% of
SO101 + ~33% of Trossen). A `BaseSetupWorker` class could absorb this.

### 2.2 Frontend — WebSocket Hooks

**Files:**
- `use-setup-websocket.ts` (317 lines, 11 state fields, 13 commands)
- `use-trossen-setup-websocket.ts` (161 lines, 6 state fields, 3 commands)

**Identical patterns:**
- `useState` + `useRef(state)` + `stateRef.current = state` — boilerplate
- `handleMessage` callback structure: `JSON.parse → switch(data.event) → setState`
- Both handle `status`, `state_was_updated`, `error`, `pong` identically
- URL construction: same template, different query params
- `useWebSocket` config: identical `onOpen`, `onClose`, `onError`, `shouldReconnect`
- `reProbe`, `enterVerification`, `ping` — same pattern in both

**Key differences:**
- SO101 has 7 additional state fields (`voltageResult`, `probeResult`, etc.)
- SO101 has 9 additional commands (`startMotorSetup`, `motorConnected`, etc.)
- Trossen has `diagnosticsResult` (not in SO101)
- URL query params: SO101 uses `serial_number`, Trossen uses `connection_string`

**Estimation:** ~100 lines of duplicated boilerplate. A generic
`useSetupWebSocket<TState, TCommands>()` hook could absorb the common
`useWebSocket` setup, message routing for shared events, and URL construction.

### 2.3 Frontend — Wizard Providers

**Files:**
- `so101/.../wizard-provider.tsx` (~200 lines)
- `trossen/.../wizard-provider.tsx` (~150 lines)

**Identical patterns:**
- Dual-context pattern (`StateContext` + `ActionsContext`)
- `goToStep`, `goNext`, `goBack`, `markCompleted` — identical logic
- `useState` for `wizardState` with `currentStep` + `completedSteps`
- `stepIndex`, `canGoNext`, `canGoBack` derivations
- Context provider nesting

**Key differences:**
- SO101 adds `skippedSteps`, `markSkipped`, `unmarkSkipped`, `visibleSteps` filtering
- SO101 adds `calibrationPhase`, `preVerifyProbeResult` domain state
- SO101 has auto-skip logic for MOTOR_SETUP step
- Trossen has no skippable steps (`visibleSteps = TROSSEN_WIZARD_STEPS` constant)

**Estimation:** ~80 lines of duplicated navigation logic. A generic
`createSetupWizardProvider<Step, State>()` factory could absorb the common parts.

### 2.4 Frontend — Verification Steps

**Files:**
- `so101/.../verification-step.tsx` (~180 lines)
- `trossen/.../verification-step.tsx` (~171 lines)

**Identical patterns:**
- `useSyncJointState` hook body: iterate `jointState` entries, strip `.pos`
  suffix, handle `gripper` → `left_carriage_joint` mapping for `wxai`, convert
  `degToRad()` for all other joints. The loop is **character-for-character
  identical** between the two files (only the URDF path resolution differs).
- `robotBody` construction — same shape, only `connection_string` vs
  `serial_number` emphasis differs.
- "Save Robot" button + `isPending` + `isDisabled` — identical.
- Robot Details card JSX — ~80% identical.

**No longer duplicated (eliminated by phase-driven refactoring):**
- ~~Mount effect: `if (isConnected) streamPositions(); return () => stopStream()`
  using `mountRef` pattern~~ — streaming is now controlled by the backend's
  `_broadcast_loop()`, triggered by the `enterVerification` command sent from
  the diagnostics step's "Continue" button.

**Key differences:**
- SO101 has a 3-step save chain (create robot → save calibration → update
  `active_calibration_id`). Trossen has a 1-step save (create robot only).
- SO101 shows calibration status in the details card.
- SO101 hardcodes `SO101_PATH`; Trossen calls `urdfPathForType()`.

**Estimation:** ~50 lines of duplicated code. The `useSyncJointState` hook
(~20 lines) should be extracted into `shared/setup-wizard/`. The mount/unmount
streaming pattern has been eliminated by the phase-driven broadcast refactoring
— no extraction needed.

### 2.5 Frontend — Diagnostics Steps

**Files:**
- `so101/.../diagnostics-step.tsx` (~250 lines)
- `trossen/.../diagnostics-step.tsx` (~170 lines)

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
complexity without proportional benefit. The header with refresh button could be
a small shared component.

### 2.6 Frontend — Layout (`setup-wizard.tsx`)

**Files:**
- `so101/.../setup-wizard.tsx` (~180 lines)
- `trossen/.../setup-wizard.tsx` (~100 lines)

**Identical patterns:**
- Grid structure: `areas={['stepper stepper', 'form viewer']}`,
  `columns={['size-6000', '1fr']}`, `rows={['auto', '1fr']}` — 100% identical.
- Stepper rendering with `<Divider>` — identical.
- `ViewerPanel` null-type fallback (dashed border + "Select a robot type")
  — 100% identical.
- `ViewerPanel` model display (background + overflow hidden + `<SetupRobotViewer>`)
  — ~90% identical.
- Step content switching via `currentStep ===` conditionals — same pattern.

**Key differences:**
- SO101 passes `highlights` to `SetupRobotViewer`; Trossen passes `[]`.
- SO101 has `useHighlights()` hook and animation hooks; Trossen has neither.
- SO101 renders 4 steps; Trossen renders 2.

**Estimation:** ~50 lines of duplicated JSX. A generic `SetupWizardLayout`
component could render the grid, stepper, viewer panel, and accept a
`renderStep(currentStep)` callback. However, the SO101-specific animations
and highlights complicate this — they'd need to be injected via props.

### 2.7 Frontend — Route Components

**Files:**
- `routes/robots/so101-setup.tsx` (~15 lines)
- `routes/robots/trossen-setup.tsx` (~15 lines)

These are trivial wrappers: `<Provider><View><Content /></View></Provider>`.
They are intentionally separate for route isolation. No refactoring needed.

---

## 3. Can `RobotWorker` / `RobotClient` Be Leveraged? <a id="3-robot-worker-robot-client"></a>

### The Verification Step — A Strong Candidate

The verification step (streaming positions for 3D preview) does the same thing as
`RobotWorker._broadcast_loop()`:

| Aspect         | RobotWorker broadcast                                   | Setup worker streaming                                                     |
|----------------|---------------------------------------------------------|----------------------------------------------------------------------------|
| Read positions | `client.read_state(normalize=True)`                     | `bus.sync_read(..., normalize=True)` or `driver.get_all_positions()`       |
| Deduplication  | `if current_values != previous_values`                  | SO101 does this for raw positions only                                     |
| Output event   | `state_was_updated` (with `timestamp`, `is_controlled`) | `state_was_updated` (without `timestamp`, without `is_controlled`)         |
| Frequency      | Configurable FPS (default 30)                           | Phase-driven `_broadcast_loop()` at fixed FPS (SO101: 30Hz, Trossen: 20Hz) |

**Could we use `RobotClient` for verification?**

In theory, yes — during verification, the setup worker only needs `read_state()`.
But there are practical blockers:

1. **SO101 `RobotClient` requires a saved robot.** `RobotClientFactory.build(robot)`
   needs a `Robot` schema object with an `id` and `active_calibration_id`. The
   setup wizard deliberately avoids persisting to the DB until "Save Robot".

2. **SO101 setup worker manages `bus.calibration` manually.** It reads calibration
   from EEPROM and assigns it to `bus.calibration` directly. The `SO101Follower`
   client calls `lerobot.robots.so101_follower.SO101Follower.connect()`, which
   triggers lerobot's interactive `input()` prompts — the exact thing the setup
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

The setup workers' `match/case` dispatch handles 4–12 commands. The normal
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
- The current `match/case` is simple and clear — adding Pydantic models would
  double the code for the same behavior
- Setup commands are not performance-critical (they're user-initiated, not in a
  tight loop)

**Verdict:** Not worth it today. If setup commands grow significantly (e.g., 20+
commands), Pydantic models would be justified. For 4–12 commands, `match/case`
is clearer and more maintainable.

---

## 4. Should the Messaging Protocol Be Stricter? <a id="4-messaging-protocol"></a>

### Current Inconsistencies

| Issue                                  | Normal operation                                       | Setup workers                                      |
|----------------------------------------|--------------------------------------------------------|----------------------------------------------------|
| `timestamp` field                      | Always present (via `_create_event()`)                 | Never present                                      |
| `is_controlled` in `state_was_updated` | Always present                                         | Never present                                      |
| Error event format                     | `{"event": "error", "timestamp": ..., "message": ...}` | `{"event": "error", "message": ...}`               |
| Status event shape                     | `WorkerStatus.to_json()` → includes `config` key       | `{event, state, phase, message}` — no `config` key |

### Assessment

These inconsistencies are **not currently causing bugs** because:
- The frontend setup hooks and the normal robot websocket hooks are completely
  separate code paths — they never share event handlers.
- The frontend `useSyncJointState` doesn't read `timestamp` or `is_controlled`.

However, they create a **conceptual tax**: developers must remember which event
format they're dealing with depending on the context (setup vs. normal
operation).

### Recommendation

Add `timestamp` to setup worker events by using `RobotClient._create_event()`
(or extracting it as a standalone utility). This is a one-line change per event
and makes the protocol uniform. Do **not** add `is_controlled` to setup events —
it's meaningless during setup (the robot is never "controlled" in the teleoperation
sense).

---

## 5. Upcoming Robots: LeKiwi and Aloha Mini <a id="5-upcoming-robots"></a>

### LeKiwi

LeKiwi is a mobile robot base (differential drive) with an SO101 arm mounted on
top. Expected characteristics:
- **Motor bus:** Feetech STS3215 (same as SO101) for the arm joints, plus
  additional motors for the wheel base
- **Connection:** Likely serial (USB) for the arm, possibly additional
  connection for the base
- **Calibration:** Same homing + range recording flow as SO101 for the arm
  joints; base motors may not need calibration
- **Setup wizard needs:** Very similar to SO101 — voltage check, motor probe,
  per-motor ID assignment, calibration. May need additional steps for base
  motor configuration.

**Implication:** LeKiwi would benefit heavily from an SO101-derived setup
worker. If we extract a `BaseSetupWorker`, LeKiwi could extend the SO101
worker (or share most of its code) and add base-motor-specific steps.

### Aloha Mini

Aloha Mini is a bimanual robot with two SO101-like arms. Expected characteristics:
- **Motor bus:** Likely Feetech STS3215 for both arms
- **Connection:** Two USB serial connections (one per arm)
- **Calibration:** Same as SO101, but for 12 motors across 2 arms
- **Setup wizard needs:** Two SO101 setup flows in sequence (or parallel?), plus
  a bimanual coordination verification step.

**Implication:** Aloha Mini would reuse the SO101 setup flow almost entirely,
just running it twice. A `BaseSetupWorker` with a configurable motor list and
connection method would make this straightforward.

### What This Tells Us About Abstraction

The upcoming robots reinforce that:
1. The SO101 pattern (motor probe → ID assignment → calibration → verification)
   is the "complex" path that will be reused by LeKiwi and Aloha Mini.
2. The Trossen pattern (IP ping → configure → verification) is the "simple" path
   for SDK-managed robots.
3. Both paths share the same WebSocket lifecycle, command loop, and phase-driven
   broadcast loop.

A `BaseSetupWorker` that provides the lifecycle, command loop, and broadcast loop
infrastructure — with subclasses providing the robot-specific diagnostics,
`_broadcast_tick()`, and calibration steps — is the right level of abstraction.

---

## 6. Pros and Cons of Adding Abstractions Now <a id="6-pros-and-cons"></a>

### Option A: Extract `BaseSetupWorker` + shared frontend hooks now

**Pros:**
- Eliminates ~120 lines of backend duplication, ~100 lines of frontend WS hook
  duplication, ~80 lines of provider duplication, ~50 lines of verification step
  duplication
- Establishes patterns before LeKiwi/Aloha Mini add more complexity
- Makes it clear what's robot-specific vs. framework-level
- Reduces the surface area for bugs (one implementation of the command loop,
  one implementation of the broadcast loop, etc.)

**Cons:**
- Premature if LeKiwi/Aloha Mini have unexpected requirements that break the
  abstraction
- Adds indirection — developers must understand the base class to work on any
  setup worker
- The current duplication is manageable (2 robots, well-separated files)
- Risk of over-engineering: the "shared" parts may diverge as robots get more
  complex

### Option B: Wait until LeKiwi integration, then refactor

**Pros:**
- Three data points (SO101, Trossen, LeKiwi) give a much better understanding
  of what's truly common vs. accidental
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
the differences (e.g., don't create a generic diagnostics step component). The
specific extractions are listed in Section 7.

---

## 7. Recommended Refactoring Steps <a id="7-recommendations"></a>

Ordered by impact and safety. Each step is independently committable.

### Step 1: Extract `BaseSetupWorker` (backend)

Create `workers/robots/base_setup_worker.py`:

```python
class BaseSetupWorker(TransportWorker):
    """Base class for robot setup workers.

    Provides:
    - run() lifecycle: connect → diagnostics → run_concurrent(broadcast_loop, command_loop) → cleanup
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
**Risk:** Low — the extracted code is identical in both workers.

### Step 2: Extract `useSyncJointState` (frontend)

Move to `shared/setup-wizard/use-sync-joint-state.ts`:

```ts
// Accepts jointState + robotType, resolves URDF path via urdfPathForType(),
// handles gripper → left_carriage_joint mapping for wxai.
// SO101's hardcoded path becomes urdfPathForType('SO101_Follower').
```

**Lines saved:** ~20 per verification step (40 total).
**Risk:** Very low — the loop body is identical.

### ~~Step 3: Extract `useStreamOnMount`~~ (No Longer Needed)

The phase-driven broadcast loop refactoring eliminated the need for frontend
streaming management. The `_broadcast_loop()` auto-streams positions when the
worker enters the `VERIFICATION` phase, triggered by an `enterVerification`
command sent from the diagnostics step's "Continue" button. No mount/unmount
`useEffect` is needed.

**Lines eliminated by refactoring:** ~30 (removed from both verification steps).
**Risk:** N/A — already done.

### Step 4: Add `timestamp` to setup worker events (backend)

Extract `_create_event()` from `RobotClient` into a standalone utility (or just
call `datetime.now().timestamp()` in `_send_event()`). Apply to both setup
workers.

**Lines saved:** 0 (this is about consistency, not deduplication).
**Risk:** Very low — frontend ignores `timestamp` today.

### Step 5: Extract shared wizard navigation logic (frontend)

Create a `createWizardNavigation<S>()` utility or custom hook that provides
`goToStep`, `goNext`, `goBack`, `markCompleted`, `stepIndex`, `canGoNext`,
`canGoBack`. SO101's `markSkipped`/`unmarkSkipped`/`visibleSteps` filtering
would be opt-in.

**Lines saved:** ~50 per provider (100 total).
**Risk:** Medium — the SO101 provider's skip logic adds complexity. This should
be done carefully to avoid making the simple case (Trossen) harder to understand.

### Step 6 (optional): Generic `SetupWizardLayout` component

A shared layout component that renders the Grid + Stepper + Divider + ViewerPanel
+ step content area. SO101 and Trossen would pass:
- `steps`, `labels`, `currentStep`, `completedSteps`, `onGoToStep`
- `renderStep(step)` callback
- `viewerHighlights` (optional, defaults to `[]`)

**Lines saved:** ~40 per layout (80 total).
**Risk:** Medium — SO101's animations are driven by step-specific state, making
the viewer panel harder to generalize.

### What NOT to Extract

- **Diagnostics steps:** Domain content differs too much. Extracting shared JSX
  (error box, loading spinner, header) saves ~40 lines but adds indirection
  for components that are simple and readable as-is.
- **`commands.py`-style Pydantic models for setup commands:** Not worth the
  overhead for 4–12 commands.
- **A unified websocket hook:** The SO101 hook has 11 state fields and 13
  commands; the Trossen hook has 6 and 3. A generic hook would need extensive
  parameterization. Instead, extract only the boilerplate (WS connection setup,
  shared event handlers) and let each robot define its own state shape.

---

## Summary

| What                        | Duplication     | Recommendation                                               | Risk     |
|-----------------------------|-----------------|--------------------------------------------------------------|----------|
| Backend lifecycle + helpers | ~120 lines      | Extract `BaseSetupWorker`                                    | Low      |
| `useSyncJointState`         | ~40 lines       | Extract to shared                                            | Very low |
| ~~`useStreamOnMount`~~      | ~~~30 lines~~   | ~~Extract to shared~~ Eliminated by phase-driven refactoring | —        |
| Event `timestamp`           | Consistency gap | Add to setup events                                          | Very low |
| Wizard navigation           | ~100 lines      | Extract `createWizardNavigation`                             | Medium   |
| Layout component            | ~80 lines       | Extract (optional)                                           | Medium   |
| Diagnostics steps           | ~40 lines       | Leave as-is                                                  | —        |
| WS hook boilerplate         | ~100 lines      | Extract connection setup only                                | Medium   |
| Pydantic setup commands     | 0 duplication   | Leave as-is                                                  | —        |
| Route components            | 0 duplication   | Leave as-is                                                  | —        |

Total recoverable duplication: **~440 lines** across backend and frontend, of
which **~280 lines** (Steps 1–2, 4) can be extracted with low risk. An additional
**~30 lines** (Step 3) have already been eliminated by the phase-driven broadcast
refactoring. The remaining **~160 lines** (Steps 5–6) require more careful design
and can wait until LeKiwi integration provides a third data point.
