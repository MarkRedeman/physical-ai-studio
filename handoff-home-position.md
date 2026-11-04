# Home position and inference takeovers

## Implemented behavior

**Currently hidden:** `SHOW_HOME_POSITION = false` in `session-workspace.tsx` hides the Environment entry point pending a second iteration. The backend, storage/provider code, and persistent Stop banner remain intact. Home entry-point browser flows are skipped while hidden; teleoperation takeover remains enabled.

- **Home position** lives in Environment, outside the device-details chevron.
- Set home from fresh, stationary follower measurements while control is paused. No default/zero pose is invented.
- Save the validated pose in browser local storage, scoped to project/environment/follower. Storage failure falls back to session memory with a warning; unsaved environments are session-only.
- **Return to home** opens a confirmation dialog requiring an explicit clear-path checkbox. It stops existing control, performs a supervised joint-space return, and selects Recording only after an accepted takeover. No automatic return or automatic resumption of model/teleop control.
- **Stop return** appears above the live views and remains available independently of normal command locks or sidebar scrolling.
- Enabling **Teleoperate** during inference is also a takeover: stop model control, enable teleop, then select Recording after acknowledgement.
- Studio UI control changes reject an open episode rather than implicitly discarding it. Mode changes with no running control use a recorder-idle check without commanding motion. Passive selection retains a fallback for older backends.

## Scope and limitations

This is an **experimental operator-assisted return**, not a safety-rated function or collision planner. Clear the path before each return and have the physical emergency stop available. A feasible saved pose does not prove that the intermediate joint-space path is collision-free.

The initial adapter gate accepts only a single calibrated, serial-identified `physicalai_so101_plugin.SO101` follower in normalized or degree units, with torque held on disconnect. Raw-tick, bimanual, unidentified, and generic velocity/torque adapters are not enabled. Limits come from calibration; the prototype speed/acceleration limits are conservative software limits, not hardware certification.

- The gripper is excluded from saved targets and its measured position is preserved during return. This does **not** guarantee grip force or payload retention; commission with an empty gripper first.
- Normalized boundary readings are clipped by SO-101 and cannot identify the true position. Home capture/return rejects them; invalid hold observations stop the worker and use the adapter's native/raw-position disconnect hold.
- Homes are fingerprinted against environment, robot/serial identity, calibration, units, driver version, and home-profile version. Recalibration, replacement, or driver upgrades require recapture.
- There is no database migration or cross-browser synchronization.

## Motion/transport safeguards

- Backend-authoritative capture uses a coherent worker-loop sample, not the browser's last observation or a mixed shared-store read.
- Freshness: 250 ms; capture settling: 300 ms. Return uses a quintic trajectory with bounded velocity/acceleration, at most 60 s including settling allowance.
- A delayed worker tick (>100 ms), excessive per-joint tracking error, stale/invalid feedback, or timeout stops the operation. Completion requires measured arrival and settling, not merely a command acknowledgement.
- The UI renews a backend lease every 250 ms; the worker stops after 1.5 s without renewal. Hiding the page or unmounting cancels/stops renewals. Home commands are not buffered for reconnect/replay.
- WebSocket normal commands stay ordered; cancellation/renewal are processed separately so Stop is not queued behind a starting return.
- Operation IDs and sequence numbers prevent old progress/cancel/retry messages from disturbing a subsequent control. Cancellation during startup prevents a later trajectory from starting; session close is rechecked after awaited work.
- Stopped model/teleop threads cannot publish delayed actions. A previous control that remains alive blocks new motion rather than allowing competing controllers.

## Code map

Backend:

- `application/backend/src/schemas/home_position.py`: poses, capabilities, operation states and commands.
- `application/backend/src/runtime_session/home_position.py`: adapter gate, compatibility, trajectory and supervision.
- `application/backend/src/workers/robot_worker.py`: coherent capture, home RPCs and the only hardware writer.
- `application/backend/src/runtime_session/session.py`: takeover, episode checks, cancellation and state relay.
- `application/backend/src/api/runtime_ws.py`: priority cancellation transport.

UI, under `application/ui/src/features/runtime-session/workspace/`:

- `home-position-storage.ts`: strict storage validation and scoped keys.
- `home-position-provider.tsx`: persistence, commands, operation state and browser lease.
- `home-position.tsx` / `.module.css`: Environment block, confirmation and persistent Stop banner.
- `session-mode.tsx`: home-aware locks, teleop takeover and Recording selection.

OpenAPI/types were regenerated **offline**, without restarting the active backend or entering its lifespan. Generated files are ignored by `application/ui/.gitignore`; regenerate locally after changing schemas. End active robot sessions before restarting the backend and reconnecting the environment to expose the new capabilities.

## Verification / follow-up

Focused backend tests cover trajectory bounds, capture/identity checks, watchdog and tracking failures, normalized clipping, startup cancellation, priority Stop, late actions and preserving open episodes. UI tests cover storage validation, capture, confirmation, cancellation and no automatic movement on restoration. Browser tests in `application/ui/tests/home-position.spec.ts` use intercepted sockets only.

**No physical motion or hardware commissioning was performed.** Before hardware use, verify small cleared-path moves, actual stop latency, driver I/O timeouts, false-positive tracking thresholds, gripper/payload behavior, browser/network/backend loss, and synchronous/asynchronous inference takeover. Do not increase profile limits or add adapters without corresponding tests and operator-approved commissioning.

Final focused verification: 28 home/backend tests, 15 home/storage UI tests, 3 mocked browser flows, and the UI production build passed. The broader backend/session checks previously passed (49 tests). Full UI type-check still reports pre-existing issues in `session-workspace.test.tsx` (RTL `exact` option) and `mock-settings.ts` (obsolete `rerun` field). Broader regression cleanup remains for the next iteration.
