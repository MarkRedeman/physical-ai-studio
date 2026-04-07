# Improving Server Startup Time

## Problem

`main.py` transitively imports the entire ML stack (torch, lightning, physicalai, lerobot, cv2) at module level, even though none of it is needed until a training job runs or a WebSocket connects. This adds several seconds to server startup.

## Import Chain Analysis

### P0 (Critical): `core/scheduler.py` eagerly imports `TrainingWorker`

**Chain:**

```
main.py (line 28: from core import lifespan)
  → core/__init__.py (line 1: from core.lifecycle import lifespan)
    → core/lifecycle.py (line 14: from .scheduler import Scheduler)
      → core/scheduler.py (line 8: from workers.training_worker import TrainingWorker)
        → workers/training_worker.py (lines 11-40):
            ├─ lightning.pytorch.callbacks (ModelCheckpoint)
            ├─ lightning.pytorch.loggers (CSVLogger)
            ├─ physicalai.data (LeRobotDataModule)
            ├─ physicalai.export (ExportablePolicyMixin)
            ├─ physicalai.train (Trainer)
            ├─ models.utils (load_policy, setup_policy)
            │    ├─ physicalai.inference (InferenceModel)
            │    └─ physicalai.policies (ACT, Pi0, Pi05, SmolVLA)
            ├─ services.training_service
            │    ├─ lightning.pytorch.callbacks (Callback, ProgressBar)
            │    └─ lightning.pytorch.utilities.types (STEP_OUTPUT)
            └─ utils.device (get_lightning_strategy, get_torch_device)
```

**Libraries loaded:** torch (~2-5s alone), lightning, physicalai.\*, lerobot, all policy classes.

**Why it's unnecessary:** `Scheduler` only needs `TrainingWorker` when `start_workers()` is called during the lifespan startup, not at module import time.

**Estimated savings:** 3-8 seconds.

**Fix:** Move the import inside `start_workers()`:

```python
# core/scheduler.py — before
from workers.training_worker import TrainingWorker

class Scheduler:
    def start_workers(self) -> None:
        training_proc = TrainingWorker(...)

# core/scheduler.py — after
class Scheduler:
    def start_workers(self) -> None:
        from workers.training_worker import TrainingWorker
        training_proc = TrainingWorker(...)
```

---

### P1 (High): `api/dependencies.py` re-triggers the same chain

**Chain:**

```
main.py (line 13: from api.dependencies import CameraRegistryDep, RobotRegistryDep)
  → api/dependencies.py (line 9: from core.scheduler import Scheduler)
    → core/scheduler.py (line 8: from workers.training_worker import TrainingWorker)
      → (same ML stack as P0)
```

**Why it's unnecessary:** `Scheduler` is only used as a type annotation for FastAPI dependency injection. The functions just read from `request.app.state` — they never construct a `Scheduler`.

**Fix:** Use a `TYPE_CHECKING` guard:

```python
# api/dependencies.py — before
from core.scheduler import Scheduler

# api/dependencies.py — after
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.scheduler import Scheduler
```

Note: If P0 is fixed (scheduler no longer eagerly imports TrainingWorker), this chain becomes cheap anyway. But the `TYPE_CHECKING` guard is still the correct pattern since the import is only used for type annotations.

---

### P2 (High): API routers → `InternalLeRobotDataset` → torch

Multiple API routers eagerly import `InternalLeRobotDataset`, which does `import torch` at the top of its module.

**Chains:**

```
main.py → api/dataset.py (line 17)
  → internal_datasets.lerobot.lerobot_dataset (InternalLeRobotDataset)
    ├─ import torch
    ├─ lerobot.datasets.*
    └─ lerobot.processor

main.py → api/models.py (line 14)
  → internal_datasets.utils (get_internal_dataset)
    → internal_datasets.lerobot.lerobot_dataset
      → torch, lerobot.*

main.py → api/project.py
  → internal_datasets.utils (get_internal_dataset)
    → (same as above)
```

**Why it's unnecessary:** `InternalLeRobotDataset` and `get_internal_dataset` are only used inside request handler functions, not at module scope.

**Fix:** Move imports inside the handler functions, or use lazy imports in `internal_datasets/utils.py`:

```python
# internal_datasets/utils.py — before
from internal_datasets.lerobot.lerobot_dataset import InternalLeRobotDataset

def get_internal_dataset(...):
    return InternalLeRobotDataset(...)

# internal_datasets/utils.py — after
def get_internal_dataset(...):
    from internal_datasets.lerobot.lerobot_dataset import InternalLeRobotDataset
    return InternalLeRobotDataset(...)
```

---

### P3 (High): `api/record.py` → robot control → cv2, lerobot, physicalai, trossen_arm

**Chain:**

```
main.py → api/record.py (lines 13-15)
  ├─ robots.robot_client_factory (RobotClientFactory)
  │    ├─ robots.so101 → lerobot.motors.feetech (FeetechMotorsBus)
  │    ├─ robots.widowxai → trossen_arm
  │    └─ ...
  ├─ core.scheduler → (ML stack, same as P0)
  └─ workers.robot_control_worker (RobotControlWorker)
       ├─ control.environment_integration
       │    ├─ cv2
       │    ├─ lerobot.datasets.*
       │    └─ physicalai.data (Observation)
       ├─ control.sync_mixed_model_integration
       │    ├─ physicalai.data (Observation)
       │    └─ workers.model_worker → physicalai.inference
       └─ internal_datasets.lerobot.lerobot_dataset → torch
```

**Libraries loaded:** torch, lerobot, physicalai, cv2, trossen_arm.

**Why it's unnecessary:** These are only needed when a user connects to the robot control WebSocket, not at router registration time.

**Fix:** Move heavy imports inside the WebSocket handler functions.

---

### P4 (Moderate): `api/camera.py` → `CameraWorker` → cv2

**Chain:**

```
main.py → api/camera.py (lines 7, 14)
  ├─ frame_source (FrameSourceFactory)
  └─ workers.camera_worker (CameraWorker)
       ├─ import cv2
       └─ import numpy as np
```

**Estimated savings:** ~0.3-0.5 seconds.

**Fix:** Lazy-import `CameraWorker` inside the endpoint handlers that use it.

---

## What's Already Done Right

`utils/device.py` correctly defers `import torch` inside the function body rather than at module level. This is the pattern to follow everywhere.

## Recommended Order of Implementation

| Step | Fix | Estimated Savings |
|------|-----|-------------------|
| 1 | P0: Lazy-import `TrainingWorker` in `scheduler.py` | 3-8s |
| 2 | P1: `TYPE_CHECKING` guard for `Scheduler` in `api/dependencies.py` | Prevents redundant trigger of P0 chain |
| 3 | P2: Lazy-import `InternalLeRobotDataset` in `internal_datasets/utils.py` | 2-5s (independent torch path) |
| 4 | P3: Lazy-import robot/control code in `api/record.py` | 1-3s (cv2, trossen_arm, lerobot) |
| 5 | P4: Lazy-import `CameraWorker` in `api/camera.py` | 0.3-0.5s |

P0 alone would have the largest impact. P1-P4 clean up independent import paths to the same heavy libraries.

## Background Warmup: Fast Startup Without Cold-Start Latency

Deferring heavy imports (P0-P4) makes the server start fast, but introduces a cold-start penalty: the first training job or camera feed request pays the full import cost. We can eliminate this by eagerly loading the heavy modules in a background thread *after* the server is already accepting requests.

### Approach: `asyncio.create_task` + `asyncio.to_thread` in lifespan

Schedule a fire-and-forget task right before yielding in the lifespan context manager. The server starts accepting requests immediately; the warmup runs concurrently in a thread pool worker.

```python
# core/lifecycle.py

import asyncio

_WARMUP_MODULES = [
    "torch",
    "lightning.pytorch",
    "physicalai.data",
    "physicalai.train",
    "physicalai.export",
    "physicalai.inference",
    "physicalai.policies",
    "cv2",
    "lerobot",
]

def _warmup_imports() -> None:
    """Import heavy modules so they are cached in sys.modules."""
    import importlib
    for mod in _WARMUP_MODULES:
        try:
            importlib.import_module(mod)
        except ImportError:
            pass  # optional dependency not installed

@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    setup_uvicorn_logging()
    # ... fast startup (scheduler, registries, etc.) ...

    # Schedule background warmup — server is about to start accepting
    # requests, so this runs concurrently with early traffic.
    asyncio.create_task(asyncio.to_thread(_warmup_imports))

    yield

    # ... shutdown ...
```

### Why this works

1. **Python's import lock is thread-safe.** If a request handler needs `torch` while the warmup thread is already importing it, the handler blocks on the import lock until the warmup finishes — no double-import, no race condition.

2. **No GIL contention in practice.** Module imports are mostly I/O (reading `.pyc` files from disk, loading shared libraries via `dlopen`). The GIL is released during these operations, so the event loop remains responsive.

3. **Graceful degradation.** If a user triggers training before warmup completes, the import just completes synchronously in whichever thread needs it first. There is no error path.

### Alternative approaches considered

| Approach | Pros | Cons |
|----------|------|------|
| `threading.Thread(daemon=True)` | Simple, no async needed | Not integrated with lifespan; harder to track completion |
| `loop.run_in_executor(None, ...)` | Uses the default thread pool | Returns a Future that nobody awaits; slightly less idiomatic |
| **`asyncio.create_task(to_thread(...))`** | **Clean, idiomatic, integrated with lifespan** | **Recommended** |
| `importlib.util.LazyLoader` proxies | Transparent to consuming code | Fragile with complex packages (torch, lightning); hard to debug; can break `isinstance` checks |
| Separate warmup process | True parallelism (no GIL) | Doesn't help — each process has its own `sys.modules`; the import must happen in the server process |

### Optional: Warmup completion signal

If you want to log when warmup finishes or gate certain behavior on it:

```python
_warmup_done = asyncio.Event()

async def _warmup_task():
    await asyncio.to_thread(_warmup_imports)
    _warmup_done.set()
    logger.info("Background import warmup complete")

# In lifespan:
asyncio.create_task(_warmup_task())
```

Endpoint handlers can then optionally `await _warmup_done.wait()` if they want to guarantee the imports are ready before proceeding — though in practice the import lock already handles this correctly.

## Dev Hot-Reload and Multiprocessing Workers

FastAPI/uvicorn hot-reload only restarts the **API server process**. It does **not** hot-patch running child processes created with `multiprocessing`.

In this codebase, `TrainingWorker` runs as a separate process, so changes to files like `workers/training_worker.py` are not reflected until:

1. the API process reloads,
2. lifespan shutdown stops child workers,
3. lifespan startup re-imports heavy ML modules,
4. a new worker process is spawned.

This explains why hot-reload works great for API routes but feels slow for training code edits.

### Why it feels especially slow here

Training startup imports `torch`, `lightning`, `torchmetrics`, and `scipy`, which can take multiple seconds per fresh process. Even with API startup optimized, worker refresh still pays this cost.

### Recommended dev workflow improvement: split API and worker in development

Use two independent processes in dev:

- **API process** (`uvicorn --reload`) for endpoints/UI integration
- **Worker process** (`training-worker` entrypoint with file watcher) for training loop logic

Both processes communicate through the same DB/event queue abstractions already used by the app. In dev mode, the API should **not** auto-spawn the training worker; instead, the worker is launched by a separate command.

#### Suggested design

1. Add a setting flag:
   - `AUTO_START_TRAINING_WORKER=true` (default for prod)
   - In dev, set `AUTO_START_TRAINING_WORKER=false`
2. In `lifespan`, only call `app_scheduler.start_workers()` when the flag is true.
3. Add a dedicated CLI command, e.g.:

```bash
uv run --no-sync src/cli.py run-training-worker
```

4. Wrap that command with a watcher in dev (`watchfiles`/`watchexec`), scoped to worker-related paths:

```bash
watchfiles \
  --filter python \
  'uv run --no-sync src/cli.py run-training-worker' \
  src/workers src/services src/models src/control src/utils
```

5. Keep API reload separate:

```bash
./run.sh   # starts uvicorn with reload in dev
```

Now API edits and worker edits restart independently, which shortens iteration loops.

### Optional high-leverage refinements

- **Narrow worker watch scope** to avoid unnecessary restarts on unrelated API-only edits.
- **Debounce restart events** (e.g. 200-500 ms) so formatter save bursts trigger one restart.
- **Delay heavy imports in worker process entrypoint** exactly like API code paths, keeping worker boot minimal.
- **Mark readiness in logs** (e.g. `Training worker ready in Xs`) so startup regressions are obvious.

### Production guidance

Do not use file watchers in production. Keep the current integrated lifecycle behavior:

- API process starts
- background warmup runs
- scheduler starts worker(s)

This keeps deployment simple and deterministic while preserving fast API availability.

### Expected impact

- **API endpoint edit loop:** remains fast (~1-2s reload)
- **Worker code edit loop:** improves from full app restart + worker spawn to isolated worker restart
- **Overall dev UX:** less waiting, fewer unrelated restarts, clearer ownership of failures (API vs worker)
