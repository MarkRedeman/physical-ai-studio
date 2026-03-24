# Proposal: Split Training and Inference Workers into Runtime-Isolated Services

## Executive summary

This proposal introduces a **coordinator + worker** runtime model to decouple ML execution from the main backend process.

### Primary outcomes

1. Run training and inference on **different hardware backends simultaneously** (CPU/CUDA/XPU).
2. Keep the core backend usable even when ML workers are unavailable.
3. Enable registration of multiple workers (including remote workers) and route jobs to selected workers.
4. Preserve most existing domain logic by replacing process-local queues with a networked transport.

---

## Current architecture (baseline)

Today, the backend process starts application workers and event handling in-process:

- `Scheduler` creates multiprocessing workers (notably `TrainingWorker`) and shares an `mp.Queue`.
- `EventProcessor` consumes that queue and notifies API/websocket handlers.
- Inference currently depends on process-local queues/events (`ModelWorker`).

This design is efficient for single-environment execution but tightly couples worker runtime to the backend process and environment.

---

## Problem statement

The application must support deployment across CPU, CUDA, and XPU backends. A single runtime environment cannot reliably serve mixed backend roles at once (e.g., CUDA training + XPU inference) while remaining operationally simple.

Additionally, backend startup and user workflows (robot setup, camera setup, recording) should not depend on ML worker readiness.

---

## Proposed target architecture

## 1) Service roles

### Coordinator service

- Runs FastAPI/API routes, orchestration, registries, health/reporting.
- Tracks worker availability and dispatches commands.
- Processes worker events and updates DB/state.
- Must start and operate in degraded mode without ML workers.

### Training worker service(s)

- Executes training jobs only.
- Runs in dedicated environment/image (`cpu` or `cuda` or `xpu`).
- Registers with coordinator and sends heartbeats.

### Inference worker service(s)

- Executes inference requests only.
- Runs in dedicated environment/image (`cpu` or `cuda` or `xpu`).
- Registers with coordinator and sends heartbeats.

### Message bus

- Use Redis Streams for command/event transport across services.

---

## 2) Data/control flow

```text
Client/UI
   |
   v
Coordinator API
   |\
   | \-- DB (jobs/models/datasets)
   |
   \---- Redis Streams -------------------------------.
             |                                         |
             v                                         v
      Training Worker(s)                        Inference Worker(s)
```

---

## 3) Degraded-mode behavior (required)

The coordinator must remain functional when training/inference workers are down.

### Must remain available

- robot setup/calibration
- camera setup
- project/environment configuration
- recording/dataset operations

### Must be gated with clear errors when workers unavailable

- train job start/dispatch
- runtime inference execution

### Health response should expose capability state

```json
{
  "status": "degraded",
  "coordinator": "ready",
  "workers": {
    "training": "down",
    "inference": "down"
  },
  "capabilities": {
    "robot_setup": true,
    "camera_setup": true,
    "recording": true,
    "training": false,
    "inference": false
  }
}
```

---

## Worker registration and selection model

## Registration metadata

Each worker registers with coordinator using:

- `worker_id` (UUID)
- `role` (`training` | `inference`)
- `backend` (`cpu` | `cuda` | `xpu`)
- `host`
- `version`
- `capacity`
- optional `labels`

Suggested persistent model:

```text
workers(
  id TEXT PRIMARY KEY,
  role TEXT NOT NULL,
  backend TEXT NOT NULL,
  status TEXT NOT NULL,             -- online|offline|draining|busy
  host TEXT,
  version TEXT,
  labels JSON,
  capacity INTEGER NOT NULL DEFAULT 1,
  last_heartbeat_at DATETIME NOT NULL,
  created_at DATETIME NOT NULL,
  updated_at DATETIME NOT NULL
)
```

## Lifecycle

1. Worker starts, registers.
2. Coordinator acknowledges and marks `online`.
3. Worker heartbeats every 5–10s.
4. Missed heartbeat timeout marks worker `offline`.
5. `draining` status prevents new assignments during maintenance.

## Scheduling and user choice

Training job payload supports explicit or constrained placement:

```json
{
  "preferred_worker_id": "optional-uuid",
  "required_backend": "cuda",
  "required_labels": {"site": "lab-a"}
}
```

Scheduler policy:

1. filter by role + online + constraints
2. select worker by least-loaded (or round-robin)
3. enqueue command to selected worker stream
4. persist assignment in job record

This supports multiple training workers concurrently (e.g. one CUDA, one XPU) and user selection at job submission time.

---

## Remote workers (VPN/Tailscale)

This architecture supports remote workers on other machines.

### Supported pattern

- Coordinator + Redis on one host/network segment
- Workers connect over private VPN (e.g. Tailscale)
- Workers register/heartbeat and receive commands remotely

### Minimum security requirements

- private network ACLs
- per-worker auth token
- TLS for coordinator endpoints across hosts
- per-worker command authorization boundaries

### Reliability requirements

- idempotent command handling
- command ack + retry
- worker lease/heartbeat timeout recovery
- requeue behavior for unstarted work

---

## Transport design (Redis Streams)

Use role/worker scoped streams:

- `physicalai:workers:register`
- `physicalai:worker:{worker_id}:commands`
- `physicalai:worker:{worker_id}:events`
- optionally role-level streams for broadcast/control

Event envelope:

```json
{
  "event_id": "uuid",
  "event_type": "JOB_UPDATE",
  "timestamp": "...",
  "payload": {"...": "..."}
}
```

---

## Concrete implementation plan

## Phase 1 — Introduce transport abstraction

### Add

- `services/event_bus.py` (protocol/interface)
- `services/event_bus_inmemory.py` (compat mode using `mp.Queue`)
- `services/event_bus_redis.py` (Redis Streams implementation)

### Refactor

- `services/event_processor.py` to consume from `EventBus` abstraction.
- Keep existing subscription API unchanged.

## Phase 2 — Role-aware runtime startup

### Add settings

In `settings.py`:

```python
worker_mode: Literal["all", "coordinator", "training", "inference"] = "all"
training_backend: Literal["auto", "cpu", "cuda", "xpu"] = "auto"
inference_backend: Literal["auto", "cpu", "cuda", "xpu"] = "auto"
redis_url: str = "redis://redis:6379/0"
```

### Add entrypoints

- `entrypoints/coordinator.py`
- `entrypoints/training_worker.py`
- `entrypoints/inference_worker.py`

`main.py` remains for current dev path (`worker_mode=all`).

## Phase 3 — Worker registration + heartbeat

### Add

- `WorkerRegistryService` (DB-backed)
- registration endpoint/consumer
- heartbeat loop in worker entrypoints

### Update scheduler

- Scheduler dispatches to registered workers instead of local process references when in distributed mode.

## Phase 4 — Training worker transport migration

Refactor `TrainingWorker` event emission from direct `mp.Queue.put` to `event_bus.publish`.

Keep training core logic unchanged.

## Phase 5 — Inference command channel

Replace process-local observation/output queue coupling with command/result stream for cross-service inference execution.

## Phase 6 — Compose deployment profiles

Deploy coordinator + redis + independent worker services.

Example assignments:

- coordinator: cpu image
- training-worker: cuda image
- inference-worker: xpu image

---

## Redis installation and packaging strategy

## Container deployment

Use Redis container in compose:

```yaml
redis:
  image: redis:7-alpine
  restart: unless-stopped
```

No public port required unless debugging.

## Standalone desktop deployment (Tauri + PyInstaller)

Do not require manual Redis installation.

### Recommended

Bundle `redis-server` binary and run it as a managed sidecar process:

1. desktop app starts broker process
2. waits for readiness (`PING`)
3. starts coordinator
4. starts optional worker processes

If broker startup fails, coordinator still starts in degraded mode.

### Abstraction

Implement a broker manager interface:

```python
class BrokerManager(Protocol):
    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    async def connection_url(self) -> str: ...
```

Implementations:

- `ExternalRedisManager` (compose/server install)
- `BundledRedisManager` (desktop bundled runtime)

---

## Operational impacts

## Benefits

- mixed backend operation (training vs inference)
- independent scaling of worker roles
- better fault isolation (worker crash does not kill API)
- readiness/capability-aware UX

## Costs

- distributed runtime complexity
- transport and retry semantics to manage
- additional monitoring requirements

## Mitigations

- phase rollout with compatibility mode (`worker_mode=all`)
- strict event schema + idempotency keys
- heartbeat-driven worker liveness + draining mode

---

## Recommended MVP scope

Deliver first:

1. EventBus abstraction + Redis transport.
2. Coordinator degraded-mode startup behavior.
3. Worker registration/heartbeat with one training worker and one inference worker.
4. Compose deployment with `coordinator(cpu) + training(cuda) + inference(xpu)`.
5. End-to-end validation of:
   - setup/recording while workers are offline
   - training dispatch to selected worker
   - inference request routed to selected worker

This gives immediate mixed-hardware support with minimal disruption to core application logic.
