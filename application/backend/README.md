# Physical AI Studio Backend

FastAPI server for demonstration data management and VLA model training orchestration.

## Overview

The backend provides RESTful APIs and services for:

- **Camera Management** - Configure and stream from multiple camera sources (RealSense, USB, GenICam)
- **Dataset Management** - Store and organize demonstration recordings
- **Training Orchestration** - Launch and monitor policy training jobs
- **Model Management** - Track trained models and export configurations
- **WebRTC Streaming** - Real-time video streaming for data collection

## Architecture

```
backend/src/
├── api/          # FastAPI route handlers
├── core/         # Business logic and domain models
├── db/           # Database models and migrations (SQLAlchemy + Alembic)
├── repositories/ # Data access layer
├── schemas/      # Pydantic request/response schemas
├── services/     # Business logic services
├── utils/        # Shared utilities
├── webrtc/       # WebRTC signaling and streaming
└── workers/      # Background task workers
```

`core/scheduler.py` polls SQLite in a backend thread and starts a short-lived
process for each runnable training, model-optimization, or dataset-import job.
Idle Studio instances do not run job workers. Local training and model
optimization share one exclusive local slot; remote training reserves each
trainer/GPU target independently. Runnable jobs are considered in submission
order across job types, so older optimization jobs are not overtaken by newer
local training jobs. One import can run alongside them, and no import worker
stays alive while waiting for user review. The scheduler reaps completed processes, reconciles
abandoned jobs, and stops active workers during application shutdown. Per-job
multiprocessing events handle cancellation without a manager process. Shutdown
keeps the API event loop free to drain worker events while joining processes.

`services/job_handlers.py` registers one typed lifecycle handler per job type.
Handlers define runnable eligibility, atomic claim conditions, resource keys,
lazy worker construction, interruption, and orphan recovery. The scheduler owns
processes and reservations without branching on job types; the repository owns
atomic database transitions without choosing domain policy. Workers retain
execution and transactional result publication. Training target handlers remain
responsible for local/remote target selection within the training job handler.

Every worker claims its assigned job before execution. Dataset imports can claim
only queued detection/import steps, never upload or review waits. Generic
`/api/jobs/{job_id}:interrupt` rejects dataset imports because their running
workers cannot honor interruption; use the import-specific review-stage cancel
endpoint instead. A worker exit does not imply completion: a review wait or
remote reattachment can legitimately leave a job pending.

## Setup

### Prerequisites

- Python 3.13
- [uv](https://github.com/astral-sh/uv) package manager

### Install Dependencies

Choose the torch variant that matches your hardware:

```bash
cd application/backend

# Choose one matching your hardware:
uv sync --extra cpu     # CPU only
# uv sync --extra cuda  # NVIDIA GPU (CUDA)
# uv sync --extra xpu   # Intel GPU (XPU)
```

This installs all backend dependencies including FastAPI, SQLAlchemy, aiortc, and the physicalai library.

### (Optional) Enable hardware acceleration for video encoding
Using hardware acceleration for video encoding can improve the speed of recording significantly.
Please check out [this document](docs/video_hardware_acceleration_intel.md) for more information.

## Usage

### Start Server

```bash
# Activate virtual environment
source .venv/bin/activate

# Run server (local training by default)
uv run physicalai-studio serve

# Equivalent thin wrapper
./run.sh
```

Server starts at `http://localhost:7860` by default.

Pressing **Play** in runtime inference mode starts a fresh Rerun recording and
starts (or reuses) a background Rerun server on
`127.0.0.1:9876`, without opening a window. Connect a native Rerun viewer manually
with `rerun rerun+http://127.0.0.1:9876/proxy` while the session is running.
Opening the runtime page alone does not start Rerun. Pressing **Stop**, leaving
the page, or unloading or failing the model/control stops the recording.
Studio stops its background server when the last recording using it stops;
servers started outside Studio are left running. The server buffers up to 256 MiB.
Each model run has a separate recording of measured joints,
target actions, latest camera frames, model predictions, queue depth, inference
latency, and host/Studio CPU and memory usage. Visualization failures do not stop
the session. Actions are targets rather than the robot's interpolated commands;
camera frames are not guaranteed to be the inputs to a particular inference.
Visualization uses Runtime's `RerunCallback`, including its joint colors,
prediction markers, action/queue timeline windows, and CPU/GPU monitoring sidebar.
Studio dispatches control events on a bounded observer thread and caps latest
camera frames at 10 Hz. Each run has a thread-local recording so simultaneous
sessions cannot send visualization data into one another's streams.

Under **Settings → General → Rerun visualization**, enable or disable logging,
opt into opening the native viewer, change the local port, or select connect mode
with a `host:port` or `rerun+http(s)://host:port/proxy` address. Settings are persisted
in the user settings JSON and apply to new model runs, not ones already running.
The viewer and connection address are relative to the backend host, not the browser.

To change the host/port:

```bash
# Option A: CLI flags
uv run physicalai-studio serve --host 127.0.0.1 --port 8000

# Option B: environment variables
HOST=127.0.0.1
PORT=8000
```

### Remote Training

The `serve` process supports local and remote training at the same time. Configure
remote trainer URLs in the Studio UI, then choose the execution target when you
submit a training job. `run.sh` is a thin wrapper around the CLI.

To run training on a separate, GPU-enabled machine, deploy and configure a
Physical AI Trainer service from
[`docs/remote-trainer.md`](docs/remote-trainer.md), then register its URL
as a remote trainer in the Studio UI. The backend sends dataset snapshots to
the service, monitors the training job, and imports the resulting model.

Alternatively, register an SSH training target. Studio manages a persistent
trainer container on the GPU server and reaches it through a standing SSH
tunnel. SSH targets are available when Studio is bound to localhost; they are
unavailable on network-exposed deployments because the Studio API has no
authentication. See
[`docs/explanation/ssh-remote-trainer.md`](docs/explanation/ssh-remote-trainer.md)
before registering one.

### Export Variants and Optimization

In a model's **Model formats** tab, choose the **Add variant** card, select an
OpenVINO source export, and choose **Optimize to INT8**. Studio queues an
`optimize_model` job and compresses the policy's weights
with NNCF's symmetric INT8 mode. The model needs a complete OpenVINO export;
optimization does not export a missing format or require a calibration dataset.
NNCF is included in the backend dependencies.

The original model and its training history stay unchanged. New artifacts live
under `exports/<backend>/variants/<variant>/` on the **same model**, with a
complete `manifest.json` and all weights, tokenizers, and other dependencies in
each package. Each published backend/variant has its own compact card with
package-specific download actions. Choose a variant in the inference dialog or
runtime workspace. Downloads contain only the selected package; a full model
download includes all published variants. Unused named variants can be deleted.
Each card's actions menu includes download, runtime export (where supported),
and deletion for named variants. **Optimize** opens the same creation dialog
with that card's export preselected, when further optimization is available.
**Run** opens inference directly for that card's backend and variant, using the
preferred compatible device. Robot motion still requires **Play** in the inference
workspace; opening the screen does not start control automatically.

Supported operations are:

- **Optimize to INT8**: symmetric weight compression of an uncompressed OpenVINO
  package, including RTC and SnapFlow exports. No checkpoint is required.
- **Export RTC**: re-export SmolVLA or Pi05 from its matching checkpoint with RTC
  enabled before tracing. Studio uses Runtime's RTC executor and action queue,
  carries RTC controls through the inference worker, and measures RPC latency.
- **Export SnapFlow**: export an existing `snapflow-model.ckpt`. This does not
  distill ordinary weights: run SnapFlow training first when that checkpoint is
  missing. New two-phase training runs export the ordinary checkpoint at the
  backend root and the distilled checkpoint under `variants/snapflow/`.

Canonical combinations are `int8`, `rtc`, `rtc-int8`, `snapflow`,
`snapflow-int8`, `snapflow-rtc`, and `snapflow-rtc-int8`. Graph-changing operations
reload a checkpoint before optionally applying INT8; they do not edit an INT8 IR
to add RTC. Operations with missing prerequisites are disabled with a reason.

Graph-changing variant exports memory-map modern checkpoints and discard
optimizer/scheduler state before building a disposable CPU policy. LoRA/DoRA
weights are merged without restoration snapshots because the export worker does
not reuse that policy. ONNX intermediate files and external weight sidecars stay
inside the job's staging directory and are removed after export or by failed-job
cleanup, rather than consuming a RAM-backed `/tmp`. Pi05 conversion still needs
substantial memory for model construction and the ONNX/OpenVINO graphs.

Manifests record portable metadata under `studio.variant`; no database migration
or additional model record is needed. Complete inference-only packages, full
model folders, and `exports/` directories can be imported. Existing separately
optimized models and their `studio.optimization` metadata remain supported;
Studio does not merge or delete them automatically.

Runtime exports flatten the chosen package under `exports/<backend>/` and include
`runtime.yaml`. RTC downloads also include `run_rtc.py`: use
`python run_rtc.py --duration-s=60` for shared model/executor latency tracking.
The plain `physicalai run` recipe supports RTC with a zero-delay estimate because
JSON/YAML recipes cannot share the same tracker object between components.

Optimization runs locally, including for models trained on remote trainers.
Its status and cancellation control appear on the source model's OpenVINO card.
Exclusive filesystem reservations prevent duplicate active target variants. Queued
cancellations publish job updates immediately, without waiting for a worker.

NNCF cannot be interrupted during compression. Running cancellation requests are
persisted in the job payload; publication checks that request atomically before
publishing the variant. Cancellation that wins discards unpublished output;
if publication wins first, a late cancellation receives a finished-job conflict.
Failed or interrupted jobs discard unpublished files
and can be retried; Studio reconciles abandoned jobs after a restart.
Hidden publication markers prevent incomplete packages from appearing in the
catalog. Completed publications whose worker died before removing the marker
are reconciled when the model is next read. Private staging/reservation files
are never included in downloads.

This is **weight-only** compression, not full INT8 activation quantization.
Model size, inference speed, and policy accuracy depend on the model and device.
Validate the optimized policy before deploying it to a robot.

### Inference execution modes

Execution modes are export-specific: ordinary exports offer synchronous and
asynchronous execution; RTC-enabled OpenVINO exports use **Real-time chunking
(RTC)**. Select an RTC variant to use RTC, or an ordinary export for sync/async.
An ordinary compiled graph cannot gain RTC inputs by changing execution mode.
RTC export uses the existing checkpoint and does not require RTC-specific training.
Studio uses the existing Runtime executor, queue, and inference APIs unchanged.

### Database Migrations

```bash
# Create new migration
uv run alembic revision --autogenerate -m "description"

# Apply migrations
uv run alembic upgrade head

# Rollback migration
uv run alembic downgrade -1
```

### CLI Commands

```bash
# Initialize database
uv run physicalai-studio db init

# Run migrations
uv run physicalai-studio db migrate
```

## API Documentation

Once the server is running:

- **Interactive API Docs** - http://localhost:7860/docs (Swagger UI)
- **Alternative Docs** - http://localhost:7860/redoc (ReDoc)
- **OpenAPI Schema** - http://localhost:7860/api/openapi.json

## Configuration

Configuration via environment variables (see `src/settings.py`):

| Variable      | Description                                                                                                  | Default                                                                                                 |
|---------------|--------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|
| `STORAGE_DIR` | Root directory for persistent artifacts (`datasets/`, `models/`, `snapshots/`, `robots/`, `cache/`, `logs/`) | Linux: `${XDG_DATA_HOME:-~/.local/share}/physicalai`; macOS: `~/Library/Application Support/physicalai` |

Create `.env` file in backend directory for local overrides.

## Development

### Code Quality

```bash
# Format code
uv run ruff format .

# Lint code
uv run ruff check .

# Type check
uv run mypy src/

# Type check (Pyrefly)
uv run pyrefly check -c pyproject.toml
```

### Project Structure

- **API Layer** (`api/`) - HTTP endpoints, request validation
- **Service Layer** (`services/`) - Business logic, orchestration
- **Repository Layer** (`repositories/`) - Database queries
- **Core** (`core/`) - Domain models and pure business logic
- **Schemas** (`schemas/`) - Input/output data validation

### Adding New Endpoints

1. Define Pydantic schemas in `schemas/`
2. Create repository methods in `repositories/`
3. Implement service logic in `services/`
4. Add route handlers in `api/`
5. Register routes in `main.py`

## Troubleshooting

### Data/Storage Migration Behavior

On startup (`./run.sh`), the backend runs migration checks before Alembic:

- Storage migration: old `~/.cache/physicalai` -> `STORAGE_DIR`
- Database migration: old `data/physicalai.db`, Docker legacy `/app/data/physicalai.db`, or a legacy `$DATA_DIR/physicalai.db` -> `$STORAGE_DIR/data/physicalai.db`

In interactive terminals, users are prompted for confirmation when a migration is needed.

### Camera Not Detected

- **RealSense**: Install [librealsense](https://github.com/IntelRealSense/librealsense)
- **GenICam**: Install vendor-specific SDKs
- **USB**: Check permissions (`sudo usermod -a -G video $USER`)

## See Also

- **[Application Overview](../README.md)** - Full application architecture
- **[UI](../ui/README.md)** - React frontend
- **[Library](../../library/README.md)** - Python SDK for training
