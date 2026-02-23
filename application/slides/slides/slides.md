# Robot Setup Wizard

## Refactoring Analysis

*SO101 & Trossen WidowX AI*

Notes: This presentation summarizes the refactoring analysis for the robot setup wizard feature. We'll cover common patterns, duplication, and concrete recommendations.

---

## Agenda

1. **Common Features** — What both wizards share
2. **Duplication Analysis** — Where code is repeated
3. **Can We Reuse RobotWorker/RobotClient?**
4. **Messaging Protocol** — Consistency gaps
5. **Upcoming Robots** — LeKiwi & Aloha Mini
6. **Recommendations** — What to extract and when

Notes: The analysis covers both backend (Python) and frontend (React/TypeScript) code.

---

## The Two Wizards

| | SO101 | Trossen WidowX AI |
|---|---|---|
| **Connection** | USB Serial | IP Network |
| **Motors** | 6 (Feetech STS3215) | 7 (Dynamixel-based) |
| **Calibration** | Homing + range recording | None (hardcoded) |
| **Setup steps** | 4 (Diagnostics, Motor Setup, Calibration, Verification) | 2 (Diagnostics, Verification) |
| **Backend** | `SO101SetupWorker` (746 lines) | `TrossenSetupWorker` (371 lines) |

Notes: SO101 is the "complex" path with per-motor setup. Trossen is the "simple" path — the SDK handles most configuration internally.

---

## Common Features

-v-

### Backend — Identical Code

Both workers extend `TransportWorker` and share:

- **`run()` lifecycle**: connect &rarr; diagnostics &rarr; command loop &rarr; cleanup
- **`_command_loop()`**: `while not _stop_requested: receive &rarr; dispatch`
- **`_send_phase_status()`**: `{event, state, phase, message}`
- **`_send_event()`**: `{event, **kwargs}`
- **`_spawn_task()`**: Background task set management
- **Streaming guard**: `if self._streaming: return`
- **`ping`/`pong`** heartbeat

Notes: These 7 items are character-for-character identical in both workers. They are prime candidates for extraction.

-v-

### Frontend — Already Shared

These components were extracted during development:

| Component | Location |
|---|---|
| `Stepper` | `shared/setup-wizard/stepper.tsx` |
| `SetupRobotViewer` | `shared/setup-wizard/setup-robot-viewer.tsx` |
| `useJointHighlight` | `shared/setup-wizard/use-joint-highlight.ts` |
| SCSS styles | `shared/setup-wizard/setup-wizard.module.scss` |

Notes: The stepper was refactored from context-dependent to props-based so both wizards can use it.

-v-

### Frontend — Duplicated Patterns

Still duplicated across both wizards:

- **WebSocket hooks** — `useState` + `useRef` boilerplate, `handleMessage` switch, URL construction
- **Wizard providers** — dual-context pattern, navigation logic (`goToStep`, `goNext`, `goBack`)
- **Verification steps** — `useSyncJointState` loop, mount/unmount streaming, save flow
- **Layout components** — Grid structure, stepper rendering, viewer panel

Notes: These represent ~310 lines of recoverable duplication across the frontend.

---

## Duplication by the Numbers

-v-

### Overview

| Area | Duplicated Lines | Risk to Extract |
|---|---|---|
| Backend lifecycle + helpers | ~120 | Low |
| Frontend WS hooks | ~100 | Medium |
| Wizard providers | ~80 | Medium |
| Verification steps | ~80 | Very Low |
| Layout components | ~80 | Medium |
| Diagnostics steps | ~40 | **Leave as-is** |
| **Total recoverable** | **~470** | |

Notes: ~310 lines can be extracted with low risk (Steps 1-4 of recommendations). The remaining ~160 lines need more careful design.

-v-

### Backend — Setup Workers

```
so101_setup_worker.py    ████████████████████████  746 lines
                          ████ ~120 shared (~16%)

trossen_setup_worker.py  ████████████  371 lines
                          ████ ~120 shared (~32%)
```

**Identical:** `_send_phase_status`, `_send_event`, `_spawn_task`, `_command_loop`, streaming guard, `run()` lifecycle

**Robot-specific:** Diagnostics, calibration (SO101 only), motor setup (SO101 only), `_dispatch_command`

Notes: A BaseSetupWorker class could absorb 120 lines. The command loop is the biggest win — it's identical in both workers.

-v-

### Frontend — WebSocket Hooks

```
use-setup-websocket.ts          317 lines (11 state fields, 13 commands)
use-trossen-setup-websocket.ts  166 lines ( 6 state fields,  4 commands)
```

**Shared:** `useState`+`useRef` boilerplate, `status`/`state_was_updated`/`error`/`pong` handlers, URL construction, `useWebSocket` config

**Different:** SO101 has 7 extra state fields, 9 extra commands, uses `serial_number`; Trossen uses `connection_string`

Notes: A generic hook could absorb ~100 lines of boilerplate while letting each robot define its own state shape and commands.

-v-

### Frontend — Verification Steps

```
so101  verification-step.tsx   ~180 lines
trossen verification-step.tsx  ~140 lines
```

**Character-for-character identical:**

- `useSyncJointState` — iterate joints, strip `.pos`, map gripper, convert `degToRad()`
- Mount effect — `streamPositions` on connect, `stopStream` on unmount
- Save button + `isPending` + `isDisabled`

**Different:** SO101 has 3-step save chain (create &rarr; save calibration &rarr; update); Trossen is 1-step

Notes: useSyncJointState is the lowest-hanging fruit. It's ~20 lines, identical in both files, and trivial to extract.

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
| Frequency | Configurable FPS (30Hz) | Hardcoded ~20Hz |

Notes: On the surface, the verification streaming step looks like it could delegate to RobotClient. But there are practical blockers.

-v-

### Why Not

1. **SO101 `RobotClient` requires a saved robot** — needs DB `id` + `active_calibration_id`, but setup avoids DB until "Save"

2. **SO101 setup manages `bus.calibration` manually** — `SO101Follower` client would trigger lerobot's interactive `input()` prompts

3. **Trossen `configure()` homes the robot** — creating a new client would re-home unexpectedly

4. **Different lifecycle** — `RobotClient.connect()` is long-lived; setup workers are short-lived with multiple phases

**Verdict: Not practical** without significant changes to `RobotClient`

Notes: The setup workers' direct hardware access is intentional and correct. The workarounds needed to use RobotClient would be worse than the duplication.

-v-

### What About `commands.py` Pydantic Models?

Setup workers use `match/case` for 4-12 commands.
Normal operation uses Pydantic discriminated unions for 8 commands.

**Not worth it today:**

- Only `ping` overlaps between the two command sets
- Setup commands carry different payloads
- `match/case` is clear for 4-12 commands
- Setup commands aren't performance-critical

*Revisit if setup commands grow to 20+.*

Notes: The Pydantic models add type safety but also double the code for the same behavior at this scale.

---

## Messaging Protocol

-v-

### Current Inconsistencies

| Field | Normal Operation | Setup Workers |
|---|---|---|
| `timestamp` | Always present | **Never present** |
| `is_controlled` | Always present | **Never present** |
| Error format | `{event, timestamp, message}` | `{event, message}` |
| Status shape | Includes `config` key | No `config` key |

Notes: These inconsistencies don't cause bugs today because the frontend hooks are completely separate code paths. But they create conceptual overhead.

-v-

### Recommendation

**Add `timestamp` to setup worker events.**

- One-line change per event
- Makes protocol uniform
- Frontend already ignores it (no breakage)

**Do NOT add `is_controlled`** — meaningless during setup (robot is never teleoperated)

Notes: This is a consistency improvement, not a bug fix. Low risk, low effort, good hygiene.

---

## Upcoming Robots

-v-

### LeKiwi

- **What:** Mobile base (differential drive) + SO101 arm on top
- **Motors:** Feetech STS3215 (same as SO101) + wheel motors
- **Calibration:** Same homing + range recording for arm joints
- **Implication:** Would heavily reuse SO101 setup worker, possibly extend it with base motor steps

Notes: LeKiwi is the strongest argument for extracting a BaseSetupWorker. It would share ~90% of the SO101 setup flow.

-v-

### Aloha Mini

- **What:** Bimanual robot with two SO101-like arms
- **Motors:** Feetech STS3215 (12 motors across 2 arms)
- **Connection:** Two USB serial connections
- **Implication:** Would run the SO101 setup flow twice (sequentially or parallel)

Notes: Aloha Mini further reinforces that the SO101 motor-level setup flow is the pattern that will be most reused.

-v-

### What This Tells Us

```
                    BaseSetupWorker
                    /              \
           "Complex" path      "Simple" path
         (motor-level setup)   (SDK-managed)
              /     |    \              \
          SO101  LeKiwi  Aloha Mini   Trossen
```

- SO101 pattern = motor probe &rarr; ID assignment &rarr; calibration &rarr; verification
- Trossen pattern = IP ping &rarr; configure &rarr; verification
- **Both** share: WebSocket lifecycle, command loop, verification streaming

Notes: The two setup "families" are clear. A base class that handles the shared infrastructure lets each family focus on its domain logic.

---

## Pros & Cons of Abstracting Now

-v-

### Option A: Extract Now

| Pros | Cons |
|---|---|
| Eliminates ~470 lines of duplication | Premature if LeKiwi breaks assumptions |
| Establishes patterns before 3rd robot | Adds indirection to understand |
| Reduces bug surface area | Current duplication is manageable |
| Clear what's framework vs. robot-specific | Risk of over-engineering |

-v-

### Option B: Wait for LeKiwi

| Pros | Cons |
|---|---|
| 3 data points &rarr; better abstractions | More copy-paste during LeKiwi integration |
| No wasted effort if patterns change | 3rd copy makes codebase harder to maintain |
| Current code works and is clear | Refactoring under time pressure |

-v-

### Verdict: Option A, Scoped Conservatively

Extract **only** what is character-for-character identical.

**Do not** try to abstract over the differences.

Notes: The key insight is that we don't need to create a grand unified abstraction. We just need to stop copying the same boilerplate.

---

## Recommended Refactoring Steps

-v-

### Step 1: `BaseSetupWorker` (Backend)

```python
class BaseSetupWorker(TransportWorker):
    """Provides:
    - run() lifecycle
    - _command_loop() with dispatch
    - _send_phase_status(), _send_event(), _spawn_task()
    - _streaming flag + _handle_stop_stream()

    Subclasses implement:
    - _run_diagnostics()
    - _dispatch_command(command, data)
    - _cleanup()
    - _handle_stream_positions()
    """
```

**~120 lines saved** | Risk: **Low**

Notes: This is the highest-impact extraction. The command loop alone is ~20 lines of identical code.

-v-

### Step 2: `useSyncJointState` (Frontend)

Move to `shared/setup-wizard/use-sync-joint-state.ts`

- Accepts `jointState` + `robotType`
- Resolves URDF path via `urdfPathForType()`
- Handles `gripper` &rarr; `left_carriage_joint` mapping for wxai

**~40 lines saved** | Risk: **Very Low**

Notes: The loop body is identical in both verification steps. SO101's hardcoded path becomes urdfPathForType('SO101_Follower').

-v-

### Step 3: `useStreamOnMount` (Frontend)

Move to `shared/setup-wizard/use-stream-on-mount.ts`

- Accepts `{ streamPositions, stopStream }` + `isConnected`
- Calls `streamPositions` on mount if connected
- Calls `stopStream` on unmount
- Uses `mountRef` pattern to avoid re-triggering

**~30 lines saved** | Risk: **Very Low**

-v-

### Step 4: Add `timestamp` to Setup Events

Extract `_create_event()` as standalone utility, or inline `datetime.now().timestamp()`.

**0 lines saved** (consistency improvement) | Risk: **Very Low**

-v-

### Step 5: Shared Wizard Navigation (Frontend)

`createWizardNavigation<S>()` utility providing:

- `goToStep`, `goNext`, `goBack`, `markCompleted`
- `stepIndex`, `canGoNext`, `canGoBack`
- SO101's `markSkipped`/`unmarkSkipped` as opt-in

**~100 lines saved** | Risk: **Medium**

Notes: The skip logic adds complexity. Needs careful design to keep the simple case (Trossen) simple.

-v-

### Step 6 (Optional): `SetupWizardLayout`

Shared Grid + Stepper + Divider + ViewerPanel with a `renderStep(step)` callback.

**~80 lines saved** | Risk: **Medium**

*SO101's step-specific animations complicate the viewer panel generalization.*

-v-

### What NOT to Extract

- **Diagnostics steps** — domain content differs too much (~40 lines, high indirection cost)
- **Pydantic models for setup commands** — not worth it for 4-12 commands
- **Unified websocket hook** — state shapes differ too much; extract only the boilerplate

Notes: Knowing what NOT to abstract is as important as knowing what to abstract. The diagnostics steps are simple and readable as-is.

---

## Summary

| Step | Lines Saved | Risk | Priority |
|---|---|---|---|
| 1. `BaseSetupWorker` | ~120 | Low | Do now |
| 2. `useSyncJointState` | ~40 | Very Low | Do now |
| 3. `useStreamOnMount` | ~30 | Very Low | Do now |
| 4. Event timestamps | 0 | Very Low | Do now |
| 5. Wizard navigation | ~100 | Medium | Wait for LeKiwi |
| 6. Layout component | ~80 | Medium | Wait for LeKiwi |

**Low-risk extractions: ~310 lines (Steps 1-4)**

**Deferred extractions: ~160 lines (Steps 5-6)**

Notes: Steps 1-4 are independently committable and can be done incrementally. Steps 5-6 benefit from a third data point.

---

## Thank You

### Key Takeaway

> Extract what's identical. Leave what's different.
> Let the third robot validate the abstraction.

*Full report: `robot-setup-analysis.md`*
