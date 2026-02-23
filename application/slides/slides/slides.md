# Robot Setup Wizard

## Refactoring Analysis

*SO101 & Trossen WidowX AI*

Notes: This presentation summarizes the refactoring analysis for the robot setup wizard feature. We cover common patterns, duplication, the websocket future, and concrete recommendations.

---

## Agenda

1. **Current Architecture** -- What we have today
2. **Common Features** -- What both wizards share
3. **Duplication Analysis** -- Where code is repeated
4. **Can We Reuse RobotWorker/RobotClient?**
5. **Messaging Protocol** -- Consistency gaps
6. **The Websocket Future** -- SO101/LeKiwi/Aloha via websocket
7. **Recommendations** -- What to extract and when

Notes: The analysis covers backend (Python) and frontend (React/TypeScript). A key new consideration: future robots may connect via websocket, making motor setup and calibration optional.

---

## Current Architecture

-v-

### Directory Structure

```
setup-wizard/
  shared/       4 files,   ~666 lines
  so101/        8 files, ~2,040 lines
  trossen/      6 files, ~1,194 lines

routes/robots/new.tsx     366 lines (unified page)

workers/robots/
  transport_worker.py      91 lines (base class)
  so101_setup_worker.py   810 lines (9 phases, 10 commands)
  trossen_setup_worker.py 424 lines (3 phases,  3 commands)

api/robot_setup.py        101 lines (WS dispatcher)
```

**Total: 19 frontend files (~4,266 lines), 4 backend files (~1,426 lines)**

Notes: All setup wizard code is colocated under `features/robots/setup-wizard/`. No BaseSetupWorker exists -- both workers extend TransportWorker directly. The route is unified: a single `/robots/new` page hosts both wizards.

-v-

### The Two Wizards

| | SO101 | Trossen WidowX AI |
|---|---|---|
| **Connection** | USB Serial | IP Network |
| **Motors** | 6 (Feetech STS3215) | 7 (Dynamixel-based) |
| **Calibration** | Homing + range recording | None (SDK handles it) |
| **Setup steps** | 4 (Diagnostics, Motor Setup, Calibration, Verification) | 2 (Diagnostics, Verification) |
| **Backend** | `SO101SetupWorker` (810 lines, 23 methods) | `TrossenSetupWorker` (424 lines, 13 methods) |

Notes: SO101 is the "complex" path with per-motor setup. Trossen is the "simple" path -- the SDK handles most configuration internally.

---

## Common Features

-v-

### Backend -- Identical Code

Both workers extend `TransportWorker` and share:

- **`run()` lifecycle**: connect &rarr; diagnostics &rarr; `run_concurrent(broadcast_loop, command_loop)` &rarr; cleanup
- **`_command_loop()`**: `while not _stop_requested: receive &rarr; dispatch` (identical except log message)
- **`_broadcast_loop()`**: Phase-driven streaming at fixed FPS with `time.perf_counter()` timing
- **`_send_phase_status()`**: `{event, state, phase, message}`
- **`_send_event()`**: `{event, **kwargs}`
- **`ping`/`pong`** heartbeat

Notes: These 6 items are character-for-character identical (or near-identical) in both workers. They are prime candidates for extraction into a BaseSetupWorker.

-v-

### Frontend -- Already Shared

These components live in `setup-wizard/shared/`:

| Component | File | Lines |
|---|---|---|
| `Stepper` | `stepper.tsx` | 70 |
| `SetupRobotViewer` | `setup-robot-viewer.tsx` | 203 |
| `useJointHighlight` | `use-joint-highlight.ts` | 168 |
| SCSS styles | `setup-wizard.module.scss` | 225 |

The outer layout (Grid + Stepper + Divider + Viewer Panel) is unified in `routes/robots/new.tsx`.

Notes: The stepper is fully generic (props-based). The 3D viewer supports optional joint highlights. The outer layout is shared -- no duplication between wizards.

-v-

### Frontend -- Still Duplicated

| Pattern | SO101 | Trossen | Duplication |
|---|---|---|---|
| WebSocket hooks | 308 lines, 11 state fields, 10 cmds | 171 lines, 6 state fields, 3 cmds | ~100 lines |
| Wizard providers | 265 lines, skip logic | 186 lines, no skip | ~80 lines |
| `useSyncJointState` | Hardcoded path, no joint mapping | Dynamic path, joint mapping table | ~50 lines |
| Diagnostics steps | 302 lines | 183 lines | ~40 lines (but domain differs) |

Notes: These represent ~270 lines of recoverable duplication. The diagnostics steps share boilerplate but domain content differs too much to extract.

---

## Duplication by the Numbers

-v-

### Overview

| Area | Duplicated Lines | Risk to Extract |
|---|---|---|
| Backend lifecycle + helpers | ~120 | Low |
| Frontend WS hooks | ~100 | Medium |
| Wizard providers | ~80 | Medium |
| `useSyncJointState` | ~50 | Very Low |
| Diagnostics boilerplate | ~40 | **Leave as-is** |
| Layout switching | ~40 | Low |
| **Total recoverable** | **~410** | |

Notes: ~220 lines can be extracted with low risk (Steps 1-3). The remaining ~190 lines need more careful design or benefit from a third data point.

-v-

### Backend -- Setup Workers

```
so101_setup_worker.py    ████████████████████████  810 lines
                          ████ ~120 shared (~15%)

trossen_setup_worker.py  ████████████  424 lines
                          ████ ~120 shared (~28%)
```

**Identical:** `_send_phase_status`, `_send_event`, `_command_loop`, `run()` lifecycle

**Structurally similar:** `_broadcast_loop` (different FPS, tick args), `_broadcast_tick` (different phase dispatch)

**Robot-specific:** Diagnostics, calibration (SO101), motor setup (SO101), `_dispatch_command`, `_cleanup`

Notes: A BaseSetupWorker class could absorb ~120 lines. The command loop is the biggest win -- it's identical except for one log message.

-v-

### Frontend -- WebSocket Hooks

```
use-setup-websocket.ts          308 lines (11 state fields, 10 commands)
use-trossen-setup-websocket.ts  171 lines ( 6 state fields,  3 commands)
```

**Shared event handlers:** `status`, `state_was_updated`, `error`, `pong` -- identical

**Shared commands:** `reProbe`, `enterVerification`, `ping`

**Shared boilerplate:** `useWebSocket` config, URL construction, `useState`+`useRef` pattern

Notes: A generic hook could absorb ~100 lines of boilerplate. SO101 adds voltage/probe/motor/calibration state; Trossen adds diagnosticsResult. The state shapes are fundamentally different.

-v-

### Frontend -- Verification Steps

**`useSyncJointState` -- ~90% identical in both files:**

- Iterate `jointState` entries, strip `.pos` suffix
- Map `gripper` &rarr; `left_carriage_joint` for wxai robots
- Convert all other joints with `degToRad()`

**Key differences:**

| | SO101 | Trossen |
|---|---|---|
| URDF path | Hardcoded `SO101_PATH` | Dynamic `urdfPathForType()` |
| Joint mapping | None (motor names == URDF) | `MOTOR_TO_URDF_JOINT` table |
| Save flow | 3-step (create &rarr; save cal &rarr; update) | 1-step (create only) |

Notes: useSyncJointState is the lowest-hanging fruit. Extraction should be based on the Trossen version (more general). SO101 would pass `undefined` for the joint mapping.

---

## Can We Reuse RobotWorker / RobotClient?

-v-

### The Temptation

Verification streaming looks like `RobotWorker._broadcast_loop()`:

| | RobotWorker | Setup Worker |
|---|---|---|
| Read positions | `client.read_state()` | `bus.sync_read()` / `driver.get_all_positions()` |
| Dedup | `current != previous` | SO101: raw only |
| Output | `state_was_updated` + timestamp | `state_was_updated` (no timestamp) |
| Frequency | Configurable FPS (30Hz) | Fixed FPS (SO101: 30Hz, Trossen: 20Hz) |

Notes: On the surface, the verification step could delegate to RobotClient. But there are practical blockers.

-v-

### Why Not

1. **SO101 `RobotClient` requires a saved robot** -- needs DB `id` + `active_calibration_id`, but setup avoids DB until "Save"

2. **SO101 setup manages `bus.calibration` manually** -- `SO101Follower` client would trigger lerobot's interactive `input()` prompts

3. **Trossen `configure()` homes the robot** -- creating a new client would re-home unexpectedly

4. **Different lifecycle** -- `RobotClient.connect()` is long-lived; setup workers are short-lived with phases

**Verdict: Not practical** without significant changes to `RobotClient`

Notes: The setup workers' direct hardware access is intentional. The workarounds needed to use RobotClient would be worse than the duplication.

---

## Messaging Protocol

-v-

### Inconsistencies

| Field | Normal Operation | Setup Workers |
|---|---|---|
| `timestamp` | Always present | **Never present** |
| `is_controlled` | Always present | **Never present** |
| Error format | `{event, timestamp, message}` | `{event, message}` |
| Status shape | Includes `config` key | No `config` key |

Notes: Not causing bugs today (separate code paths), but creates conceptual overhead.

-v-

### Recommendation

**Add `timestamp` to setup worker events.**

- One-line change per event (or in `BaseSetupWorker._send_event()`)
- Makes protocol uniform
- Frontend already ignores it (no breakage)

**Do NOT add `is_controlled`** -- meaningless during setup

Notes: Low risk, low effort, good hygiene.

---

## The Websocket Future

-v-

### Today's Connection Model

| | SO101 | Trossen |
|---|---|---|
| **Connection** | USB Serial &rarr; `FeetechMotorsBus` | IP Network &rarr; `TrossenArmDriver` |
| **Motor setup** | Required (per-motor ID assignment) | Not needed (SDK) |
| **Calibration** | Required (homing + recording) | Not needed (SDK) |
| **PID config** | Required (after calibration) | Not needed (SDK) |

Setup complexity is determined by the **connection method**, not the robot type.

Notes: SO101 talks directly to the motor bus, so the wizard must handle everything. Trossen delegates to the SDK.

-v-

### What's Changing

Future SO101, LeKiwi, and Aloha Mini robots may connect via **websocket**:

- A **companion process** (on Pi or similar) manages the motor bus locally
- Exposes a websocket API for position streaming, health checks, calibration
- The setup wizard connects via websocket instead of serial port

Notes: This fundamentally changes which setup steps are needed for these robots.

-v-

### Impact on Setup Steps

| Capability | Serial Bus (today) | Websocket (future) |
|---|---|---|
| Motor ID assignment | **Required** | Not needed |
| Calibration (homing) | **Required** | **Optional** |
| Calibration (recording) | **Required** | **Optional** |
| PID configuration | **Required** | Not needed |
| Diagnostics | Voltage + probe | Health check via WS |
| Position streaming | Direct bus read | Proxied through WS |

**Motor setup and calibration become opt-in, not hard requirements.**

Notes: A websocket-connected SO101 would follow a flow closer to Trossen's: diagnostics &rarr; optional calibration &rarr; verification.

-v-

### The "Complex" vs "Simple" Split Dissolves

Today:
```
     "Complex" (serial)          "Simple" (SDK)
      SO101, LeKiwi, Aloha        Trossen
```

Future:
```
     Serial-connected             Websocket/SDK-connected
      SO101-serial                 SO101-ws
      LeKiwi-serial                LeKiwi-ws
      Aloha-serial                 Aloha-ws
                                   Trossen
```

The distinction becomes **connection-method-dependent**, not robot-type-dependent. A single robot type may have both paths.

Notes: This is the key architectural insight. The BaseSetupWorker should not assume a fixed phase set.

-v-

### Architecture Implications

1. **`BaseSetupWorker` becomes essential** -- with 4+ robot types and 2 connection methods, shared lifecycle is critical

2. **Phase sets must be flexible** -- variable steps, not hardcoded to 4 or 2. SO101's auto-skip logic is a precursor.

3. **Motor setup & calibration become reusable modules** -- composable capabilities, not baked into one worker

4. **Frontend navigation must support dynamic steps** -- show/hide based on what the companion reports as necessary

Notes: The websocket future doesn't change the immediate extractions, but it changes the framing and raises the priority of flexible step management.

---

## Recommendations

-v-

### Step 1: `BaseSetupWorker` (Backend)

```python
class BaseSetupWorker(TransportWorker):
    """Provides:
    - run() lifecycle
    - _command_loop() with dispatch
    - _broadcast_loop() with _broadcast_tick()
    - _send_phase_status(), _send_event()

    Subclasses implement:
    - _run_diagnostics()
    - _dispatch_command(command, data)
    - _cleanup()
    - _broadcast_tick()
    """
```

**~120 lines saved** | Risk: **Low** | Websocket-ready: base class is connection-agnostic

Notes: Highest-impact extraction. The command loop alone is ~17 lines of identical code. Future websocket workers would be another subclass.

-v-

### Step 2: `useSyncJointState` (Frontend)

Move to `shared/use-sync-joint-state.ts`

- Accepts `jointState` + `robotType`
- Resolves URDF path via `urdfPathForType()`
- Supports optional `jointNameMapping` for robots where motor names differ from URDF joints
- Based on Trossen version (more general)

**~50 lines saved** | Risk: **Very Low**

Notes: SO101 passes no mapping (motor names == URDF joints). Trossen passes `MOTOR_TO_URDF_JOINT`. Clean parameterization.

-v-

### Step 3: Add `timestamp` to Setup Events

Add `datetime.now().timestamp()` to `BaseSetupWorker._send_event()` (or standalone utility).

**0 lines saved** (consistency improvement) | Risk: **Very Low**

-v-

### Step 4: Shared Wizard Navigation (Frontend)

`createWizardNavigation<S>()` utility providing:

- `goToStep`, `goNext`, `goBack`, `markCompleted`
- `stepIndex`, `canGoNext`, `canGoBack`
- SO101's `markSkipped`/`unmarkSkipped` as opt-in

**~100 lines saved** | Risk: **Medium**

**Higher priority with websocket future** -- dynamic step sets needed for WS-connected robots

Notes: The skip logic complexity is manageable if designed as opt-in. The simple case (Trossen) should remain simple.

-v-

### Step 5 (Optional): Inner Step Switching

A shared `renderStep(currentStep)` pattern for the step body components.

**~40 lines saved** | Risk: **Low**

*Outer layout already unified in `new.tsx`.*

-v-

### What NOT to Extract

- **Diagnostics steps** -- domain content differs too much
- **Pydantic models for setup commands** -- not worth it for 3-10 commands
- **Unified websocket hook** -- state shapes differ too much; extract only boilerplate
- **Phase composition/mixin system** -- premature; wait for websocket architecture

Notes: Knowing what NOT to abstract is as important as knowing what to abstract.

---

## Summary

-v-

### Extraction Priority

| Step | Lines | Risk | Priority |
|---|---|---|---|
| 1. `BaseSetupWorker` | ~120 | Low | **Do now** |
| 2. `useSyncJointState` | ~50 | Very Low | **Do now** |
| 3. Event timestamps | 0 | Very Low | **Do now** |
| 4. Wizard navigation | ~100 | Medium | Do now or wait |
| 5. Step switching | ~40 | Low | Optional |

**Low-risk extractions: ~220 lines (Steps 1-3)**

**Deferred extractions: ~190 lines (Steps 4-5, WS boilerplate)**

-v-

### Websocket Impact

| Step | Without WS consideration | With WS consideration |
|---|---|---|
| `BaseSetupWorker` | Nice deduplication | **Essential** -- new workers will multiply |
| `useSyncJointState` | Low-hanging fruit | Same -- connection-agnostic |
| Wizard navigation | Wait for 3rd robot | **Higher priority** -- dynamic steps needed |
| Phase composition | Not considered | **Future work** -- after WS arch is clear |

Notes: The BaseSetupWorker extraction is the highest-value change regardless of whether the websocket future materializes.

---

## Thank You

### Key Takeaways

> 1. Extract what's identical. Leave what's different.
> 2. The websocket future makes `BaseSetupWorker` essential, not optional.
> 3. Motor setup and calibration should be opt-in capabilities.
> 4. Let the third robot validate the higher-level abstractions.

*Full report: `robot-setup-analysis.md`*
