# Proposal: Split Training and Inference Workers into Independently Deployable Runtime Environments

## Goal

Support running **training** and **inference** with different hardware backends at the same time on the same host:

- Training: `cpu | cuda | xpu`
- Inference: `cpu | cuda | xpu`

while keeping the existing FastAPI application and worker model largely intact.

---

## Why this change is needed

Today, the backend process and its child workers share one Python environment and one resolved torch stack (`cpu` or `cuda` or `xpu`).

That makes combinations like:

- training on CUDA + inference on XPU
- training on XPU + inference on CPU

impossible in one runtime environment.

This is not just a packaging inconvenience; it is an environment-level runtime constraint.

---

## Proposed architecture (incremental)

Keep the monolith API, but split execution responsibilities:

1. **Coordinator API service**
   - Runs FastAPI, DB migrations, API routes, scheduler/orchestration logic.
   - Does not perform heavy torch work directly.

2. **Training worker service**
   - Runs `TrainingWorker` loop in its own image/env (`cpu|cuda|xpu`).

3. **Inference worker service**
   - Runs model inference loop(s) in its own image/env (`cpu|cuda|xpu`).

4. **Message bus service** (Redis Streams)
   - Replaces in-process `multiprocessing.Queue` event transport across process/container boundaries.

## What this unlocks for user onboarding and day-0 usage

Yes: this split also decouples the main server from ML runtime startup.

With this architecture, users can start using core product flows immediately, even if training/inference workers are not running yet:

- robot setup/calibration
- camera setup
- project/environment configuration
- data recording and dataset management

This works because the coordinator/API service no longer depends on loading torch backends or ML workers during startup.

### Required behavior contract

To make this explicit and reliable, implement the coordinator as **degraded-mode capable**:

1. API starts successfully when training/inference workers are absent.
2. Endpoints that require ML execution (train/start inference) return clear `409/503` responses with actionable messages.
3. Non-ML endpoints (robots, cameras, recordings, datasets, projects) continue to work fully.
4. Health endpoint exposes per-role readiness:
   - `coordinator: ready`
   - `training_worker: up|down`
   - `inference_worker: up|down`

Example health payload shape:

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

### Concrete implementation notes

- In `core/lifecycle.py`, do not hard-fail app startup when worker channels are unavailable.
- In coordinator mode, initialize worker clients lazily and reconnect in background.
- Gate ML actions in API handlers with a worker-availability check.
- Keep local dev default (`WORKER_MODE=all`) unchanged to preserve current developer UX.

### High-level data flow

```text
UI / API clients
      |
      v
 FastAPI Coordinator (api)
      |
      +--> DB (jobs, models, datasets)
      |
      +--> Redis Streams (commands/events)
              |                     |
              v                     v
        Training Worker       Inference Worker
      (cuda/xpu/cpu env)     (cuda/xpu/cpu env)
```

---

## Why not full microservices + Celery immediately?

Celery/RQ are valid tools, but immediate full migration is high cost and not required to unlock mixed-hardware operation.

You already have worker lifecycle primitives (`BaseProcessWorker`, `TrainingWorker`, `ModelWorker`). The minimal missing piece is **cross-container transport**.

Recommendation:

- Phase 1/2: Use **Redis Streams directly**
- Re-evaluate Celery only if you later need advanced distributed task orchestration/retries/scheduling.

---

## Concrete implementation plan

## Phase 0 — Configuration model

Add explicit worker role/backend settings.

### File: `application/backend/src/settings.py`

Add fields:

```python
worker_mode: Literal["coordinator", "training", "inference", "all"] = "all"
training_backend: Literal["cpu", "cuda", "xpu", "auto"] = "auto"
inference_backend: Literal["cpu", "cuda", "xpu", "auto"] = "auto"

redis_url: str = "redis://redis:6379/0"
event_stream_name: str = "physicalai:events"
command_stream_name: str = "physicalai:commands"
```

Behavior:

- `all` = current behavior (single-process dev mode)
- `coordinator` = API + event processor only
- `training` = training worker only
- `inference` = inference worker only

---

## Phase 1 — Event transport abstraction

Introduce an abstraction to decouple worker signaling from `mp.Queue`.

### New module: `application/backend/src/services/event_bus.py`

```python
from typing import Protocol, Any

class EventBus(Protocol):
    async def publish(self, event_type: str, payload: dict[str, Any]) -> None: ...
    async def consume_forever(self) -> None: ...
```

Implementations:

- `InMemoryEventBus` (wraps existing `mp.Queue`) for local/dev compatibility
- `RedisStreamEventBus` for distributed mode

### New module: `application/backend/src/services/event_bus_redis.py`

Use Redis Streams:

- stream: `physicalai:events`
- entry fields: `{"type": "JOB_UPDATE", "payload": "<json>"}`

### Update: `application/backend/src/services/event_processor.py`

Refactor constructor from:

```python
EventProcessor(event_queue: mp.Queue)
```

to:

```python
EventProcessor(event_bus: EventBus)
```

Keep same subscription API and handler dispatch behavior.

---

## Phase 2 — Worker entrypoints (standalone process mode)

Add standalone entrypoints for each worker role.

### New file: `application/backend/src/entrypoints/coordinator.py`

- Starts FastAPI app (current behavior)
- Creates scheduler in coordinator mode
- Starts `EventProcessor` backed by Redis bus

### New file: `application/backend/src/entrypoints/training_worker.py`

- Initializes logging
- Runs only `TrainingWorker` loop
- Publishes updates via Redis event bus

### New file: `application/backend/src/entrypoints/inference_worker.py`

- Initializes logging
- Runs only `ModelWorker` manager loop
- Consumes inference commands from Redis stream

---

## Phase 3 — Scheduler split by role

### File: `application/backend/src/core/scheduler.py`

Current scheduler always starts training worker. Change to role-based startup:

```python
if settings.worker_mode in ("all", "training"):
    start_training_worker()

if settings.worker_mode in ("all", "inference"):
    start_inference_supervisor()
```

For distributed mode:

- coordinator should **not** start local worker processes
- coordinator should only manage API-side orchestration and event consumption.

---

## Phase 4 — Training worker transport changes

### File: `application/backend/src/workers/training_worker.py`

Replace direct `self.queue.put((EventType.X, payload))` usage with event bus publisher calls.

From:

```python
self.queue.put((EventType.MODEL_UPDATE, model))
...
self.queue.put((EventType.JOB_UPDATE, job))
```

To:

```python
await self.event_bus.publish("MODEL_UPDATE", model.model_dump())
await self.event_bus.publish("JOB_UPDATE", job.model_dump())
```

Also move interrupt signaling from `mp.Event` to a command key/stream in distributed mode.

---

## Phase 5 — Inference worker command API

Inference currently uses process-local `observation_queue` and `output_queue`.

For cross-env workers, add command/result channels:

- `physicalai:inference:commands`
- `physicalai:inference:results`

Command payload example:

```json
{
  "command": "RUN_INFERENCE",
  "model_id": "...",
  "backend": "xpu",
  "request_id": "...",
  "observation": {"...": "..."}
}
```

Result payload example:

```json
{
  "request_id": "...",
  "elapsed_ms": 7.4,
  "action": {"...": "..."}
}
```

---

## Phase 6 — Docker/Compose deployment matrix

Leverage existing images:

- `physical-ai-studio-cpu`
- `physical-ai-studio-cuda`
- `physical-ai-studio-xpu`

and compose them by role.

### Proposed compose services

Add in `application/docker/docker-compose.yaml`:

```yaml
services:
  redis:
    image: redis:7-alpine
    container_name: physical-ai-redis

  coordinator:
    image: ...physical-ai-studio-cpu:${IMAGE_TAG:-main}
    environment:
      WORKER_MODE: coordinator
      REDIS_URL: redis://redis:6379/0
    depends_on: [redis]

  training-worker:
    image: ${TRAINING_IMAGE:-...physical-ai-studio-cuda:${IMAGE_TAG:-main}}
    environment:
      WORKER_MODE: training
      TRAINING_BACKEND: ${TRAINING_BACKEND:-cuda}
      REDIS_URL: redis://redis:6379/0
    depends_on: [redis]

  inference-worker:
    image: ${INFERENCE_IMAGE:-...physical-ai-studio-xpu:${IMAGE_TAG:-main}}
    environment:
      WORKER_MODE: inference
      INFERENCE_BACKEND: ${INFERENCE_BACKEND:-xpu}
      REDIS_URL: redis://redis:6379/0
    depends_on: [redis]
```

### Example deployment combinations

| Coordinator | Training | Inference |
|---|---|---|
| cpu | cuda | xpu |
| cpu | xpu | cpu |
| cpu | cuda | cuda |
| cpu | cpu | xpu |

This explicitly enables mixed backend runtime assignment.

---

## Redis installation and packaging strategy (Docker + standalone binary)

Because this proposal uses Redis Streams for cross-process transport, we need a clear installation/runtime story for two targets:

1. containerized deployment (current Docker/Compose)
2. future standalone desktop packaging (Tauri + PyInstaller backend)

### A. Docker/Compose deployment

Use a standard Redis container in compose (already proposed above):

```yaml
redis:
  image: redis:7-alpine
  container_name: physical-ai-redis
  restart: unless-stopped
```

Guidelines:

- Keep Redis on the internal compose network (no public port required unless debugging).
- Use AOF persistence if you want recovery across restarts for queued events/commands.
- For local single-node use, one Redis instance is sufficient.

### B. Standalone executable (Tauri + PyInstaller)

For standalone app distribution, avoid requiring users to pre-install Redis manually.

Recommended approach: **embed Redis as a managed sidecar process** started/stopped by the desktop app.

#### Option B1 (recommended): bundle redis-server binary with app

- Ship platform-specific `redis-server` binaries as part of the installer bundle.
- On app startup, launch `redis-server` as a child process bound to localhost only.
- On app exit, terminate Redis gracefully.

Runtime settings:

- bind: `127.0.0.1`
- port: configurable (default `6379`, fallback to ephemeral free port)
- data dir: app-owned writable directory (e.g. under `DATA_DIR`)
- auth: optional for local-only, but preferable to set a generated password and use `redis://:password@127.0.0.1:port/0`

#### Option B2: embedded message broker alternative (if Redis packaging becomes problematic)

If bundling Redis is too heavy for some platforms, define a transport interface with a second implementation (e.g. SQLite-backed queue) for standalone mode.

Trade-off:

- lower packaging complexity
- reduced throughput/features compared to Redis Streams

### C. Startup orchestration for standalone mode

For Tauri + PyInstaller, startup order should be:

1. start Redis sidecar process
2. wait for Redis readiness (`PING` with timeout/retry)
3. start coordinator backend
4. optionally start training/inference worker processes

If Redis fails to start:

- coordinator should still launch in limited mode for robot/camera/setup flows
- training/inference endpoints should return clear unavailable status

### D. Distribution considerations by platform

- **Linux**: easiest path for bundling `redis-server`; package in AppImage/deb/rpm payload.
- **macOS**: bundle signed Redis binary in app resources; ensure notarization/signing covers child executable.
- **Windows**: use a supported Redis build (or valkey equivalent) and run as child process, not as a system service.

### E. Minimal implementation contract in code

Add a broker launcher abstraction in backend/desktop integration layer:

```python
class BrokerManager(Protocol):
    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    async def connection_url(self) -> str: ...
```

Implementations:

- `ExternalRedisManager` (Docker / existing Redis)
- `BundledRedisManager` (standalone packaged app)

This keeps deployment differences out of core worker/business logic.

---

## API-level implementation details

## Worker registration, discovery, and worker selection

Yes — this architecture supports registering multiple workers per role (for example, one `training` worker on `cuda` and one on `xpu`) and letting users select which worker executes a job.

It also unlocks running workers on remote machines (for example on the same Tailscale/VPN network), as long as connectivity and auth are configured.

### Registration model

Add a worker registry concept in the coordinator:

- `worker_id` (UUID)
- `role` (`training` or `inference`)
- `backend` (`cpu|cuda|xpu`)
- `host` / `version` / `labels` (optional metadata)
- `status` (`online|offline|draining|busy`)
- `last_heartbeat_at`
- `capacity` (e.g. max concurrent jobs)

Suggested DB table:

```text
workers(
  id TEXT PRIMARY KEY,
  role TEXT NOT NULL,
  backend TEXT NOT NULL,
  status TEXT NOT NULL,
  host TEXT,
  version TEXT,
  labels JSON,
  capacity INTEGER NOT NULL DEFAULT 1,
  last_heartbeat_at DATETIME NOT NULL,
  created_at DATETIME NOT NULL,
  updated_at DATETIME NOT NULL
)
```

### Registration protocol

Worker startup flow:

1. Worker starts and connects to coordinator (or publishes to `physicalai:workers:register`).
2. Worker sends registration payload:

```json
{
  "worker_id": "uuid",
  "role": "training",
  "backend": "cuda",
  "host": "trainer-01",
  "version": "1.2.0",
  "capacity": 1,
  "labels": {"gpu_mem_gb": 24}
}
```

3. Coordinator persists/updates worker record.
4. Worker sends heartbeat every N seconds (e.g. 5-10s).
5. Coordinator marks worker offline when heartbeat timeout is exceeded.

### Command channels per worker

Use one command stream per worker (or consumer group per worker):

- `physicalai:worker:{worker_id}:commands`
- `physicalai:worker:{worker_id}:events`

This makes targeted scheduling straightforward.

### Worker selection for training

When user creates a training job, support:

1. `worker_id` explicitly selected by user, or
2. selection constraints (`backend=cuda`, label constraints), or
3. auto scheduling (least loaded online worker with matching role/backend).

Add fields to training job payload:

```json
{
  "preferred_worker_id": "optional-uuid",
  "required_backend": "cuda",
  "required_labels": {"region": "lab-a"}
}
```

Coordinator scheduler logic:

- filter workers by `role=training` + `status=online`
- apply backend/label constraints
- pick candidate by policy (least busy / round-robin / weighted)
- write command to selected worker queue
- set job status to queued-assigned

---

## Remote workers over VPN (e.g. Tailscale)

Yes — this design naturally supports remote workers.

### What it unlocks

- keep coordinator/UI on one machine
- register remote training workers on GPU hosts
- register remote inference workers on edge devices
- choose worker per job from UI/API

### Network patterns

Two valid options:

1. **Central Redis + coordinator**, remote workers connect over VPN.
2. **Coordinator API for registration + command polling**, workers use HTTPS/WebSocket to coordinator (Redis stays private).

For first implementation, central Redis is simplest.

### Security requirements for remote execution

Minimum requirements:

- private network only (Tailscale subnet/tailnet ACLs)
- worker auth token per worker (rotatable)
- TLS for coordinator endpoints if crossing hosts
- authorization policy: worker can only consume its own command stream

Recommended hardening:

- mTLS between workers and coordinator
- signed worker registration payloads
- `draining` mode for safe maintenance

### Reliability considerations for remote workers

- command acknowledgement and retry with idempotent job execution keys
- job lease/lock with timeout (to recover from worker crash)
- heartbeat timeout → requeue unstarted jobs
- explicit "in-progress" checkpointing for long training tasks

---

## Job lifecycle

- Keep DB (`JobService`) as source of truth.
- Training worker polls pending train jobs (already implemented).
- Worker emits updates via event bus.
- Coordinator consumes and fans out to websocket/API subscribers.

## Model lifecycle

- Coordinator records model metadata on completion event.
- Inference worker subscribes to model update events and lazily reloads model cache.

## Cancellation

Replace `training_interrupt_event` with command channel:

- coordinator writes `TRAINING_CANCEL` command with `job_id`
- training worker checks cancellation token each callback interval.

---

## Reliability and observability requirements

1. **Idempotent event handling**
   - Include `event_id`/`job_id` to deduplicate.

2. **Consumer groups** (Redis streams)
   - Prevent event loss during coordinator restarts.

3. **Structured logs**
   - Always include `worker_role`, `backend`, `job_id`, `request_id`.

4. **Health endpoints per service**
   - `/api/health` for coordinator
   - lightweight health command for workers (or periodic heartbeat key in Redis)

---

## Backward compatibility strategy

Keep `WORKER_MODE=all` as default initially, so current local runs (`./run.sh`) still function.

Migration path:

1. Introduce transport abstraction + redis bus.
2. Keep in-memory transport default.
3. Add standalone worker entrypoints.
4. Enable compose split mode behind env flags.
5. Switch production compose to split mode after validation.

---

## Risks and mitigations

1. **Increased operational complexity**
   - Mitigation: keep single `docker-compose.yaml` and env-driven image/backend selection.

2. **Inference latency increase due to transport**
   - Mitigation: keep Redis local to host/network namespace; benchmark P95 latency.

3. **SQLite multi-process contention**
   - Mitigation: short-term serialize writes; medium-term move to Postgres for multi-service reliability.

4. **Event ordering assumptions**
   - Mitigation: include monotonic sequence/version and apply optimistic update logic.

---

## Minimum viable milestone (recommended)

Deliver this first:

1. Redis event bus abstraction + processor integration.
2. `WORKER_MODE` split and standalone entrypoints.
3. Compose with `coordinator(cpu) + training(cuda) + inference(xpu)`.
4. Basic end-to-end scenario:
   - submit training job
   - receive training updates in coordinator
   - perform inference request through inference worker

Once this works, iterate on cancellation semantics and richer worker orchestration.

---

## Decision

Proceed with **worker/environment split** (not full microservice rewrite) as the next architecture step.

This provides mixed-hardware capability now, preserves most of the existing code, and leaves room for future evolution to Celery or full service decomposition if needed.
