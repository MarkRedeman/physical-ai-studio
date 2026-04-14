# Async Import Framework Design (Dataset + Model)

## Goal

Define a concrete, incremental design for **asynchronous import jobs** that supports:

1. Dataset import (initial scope: app-export ZIP),
2. Future dataset import sources (LeRobot v2/v3, Trossen SDK),
3. Future model import using the same framework.

This design focuses first on:

- API contracts,
- job schema/payload/steps,
- canonical manifest schemas,
- incremental rollout plan.

---

## Architectural Decisions (Locked)

1. **Imports are asynchronous jobs** (no synchronous long-running import endpoints).
2. **Scheduler-driven execution**: import workers are started via `application/backend/src/core/scheduler.py` and auto-pick pending import jobs from DB.
3. **Separate workers by resource type**:
   - `DatasetImportWorker`
   - `ModelImportWorker`
3. **Staged import UX**:
   - Step A: upload + detect + generate draft manifest,
   - Step B: user finalizes missing choices,
   - Step C: worker performs extract/register commit.
4. **Typed/discriminated job payloads** (not untyped dict-only payload).
5. **One generic import framework**, specialized by resource type (`dataset` / `model`) and source adapter.
6. **Adapter-driven import execution is required** for each worker (no source-specific logic inline in workers).

---

## Existing Backend Patterns to Reuse

- Job API under `/api/jobs` (currently `:train`) and websocket job updates.
- Worker polling pattern (e.g., `TrainingWorker`) via scheduler-managed process(es).
- DB-backed job queue using `jobs` table (`type`, `status`, `payload`, `extra_info`, timestamps).
- Event processor queue for `JOB_UPDATE` notifications.

---

## Incremental Rollout Plan

## Phase 1 — Dataset import (app exports only)

Deliver:

- New `dataset_import` job type and payload.
- Upload endpoint to create draft job.
- Detect + draft-manifest generation for app-export ZIP.
- Finalize endpoint (name/environment/user overrides).
- Import worker execution + dataset registration.

No LeRobot direct and no Trossen yet.

## Phase 2 — Model import (app exports only)

Deliver:

- New `model_import` job payload variant.
- Same staged flow with a dedicated model import worker.
- Finalize endpoint for model naming/link options.

## Phase 3 — Source expansion

- Dataset adapters for LeRobot v2/v3.
- Dataset adapter for Trossen SDK.
- Model source adapters as needed.

---

## API Proposal

All imports use project-scoped endpoints and return a `Job` resource.

## 1) Dataset import APIs

### 1.1 Create draft dataset import (upload)

`POST /api/projects/{project_id}/imports/datasets`

**Request**: `multipart/form-data`

- `archive` (required): ZIP file
- `source_hint` (optional): enum hint (`studio`, `lerobot_v2`, `lerobot_v3`, `trossen_sdk`, `auto` default)

Note: v1 supports multipart upload only. API shape should remain extensible so a future phase can support object-store/presigned references.

**Behavior**:

1. Persist uploaded file to temp/staging location.
2. Create `Job(type="dataset_import", status="pending")` with payload in `UPLOADED` step.
3. Worker automatically starts (scheduler polling pending import jobs).
4. Worker runs detection + draft manifest generation and moves job to `WAITING_FOR_USER_INPUT`.

**Response**: `202 Accepted`

```json
{
  "id": "<job_id>",
  "type": "dataset_import",
  "status": "pending",
  "progress": 0,
  "payload": { "type": "dataset_import", "step": "uploaded", "project_id": "..." }
}
```

### 1.2 Get dataset import draft/manifests

`GET /api/jobs/{job_id}`

Reuses existing job endpoint. When step reaches `WAITING_FOR_USER_INPUT`, payload contains:

- `detected_source`,
- `dataset_manifest_draft`,
- `validation_report`,
- `required_user_inputs`.

### 1.3 Finalize dataset import

`POST /api/jobs/{job_id}:finalize-dataset-import`

**Request**:

```json
{
  "dataset_name": "pick_and_place_session_01",
  "environment_id": "<uuid>",
  "default_task": "pick_cube",
  "user_overrides": {
    "fps": 30,
    "notes": "Imported from old instance"
  }
}
```

**Behavior**:

1. Valid only when job step is `WAITING_FOR_USER_INPUT`.
2. Persist finalize inputs into payload.
3. Transition step to `READY_TO_COMMIT` and status to `pending`/`running`.
4. Worker performs extraction, validation, dataset DB record creation, and completion.

**Response**: `202 Accepted` with updated job.

## 2) Model import APIs

### 2.1 Create draft model import (upload)

`POST /api/projects/{project_id}/imports/models`

Same upload semantics as dataset import, but for model packages.

### 2.2 Finalize model import

`POST /api/jobs/{job_id}:finalize-model-import`

**Model finalize request example**:

```json
{
  "model_name": "cube_grasp_policy_v5",
  "policy": "act",
  "linking": {
    "dataset_id": null,
    "snapshot_id": null,
    "parent_model_id": null
  },
  "user_overrides": {
    "set_as_latest": true
  }
}
```

### 2.3 Optional validation preview (shared)

`POST /api/jobs/{job_id}:revalidate-import`

Runs validation against updated overrides without final commit. Optional for v1; useful for UI preview loops.

---

## Job Schema & Step Design

## 1) Job type extension

In `application/backend/src/schemas/job.py`:

- Extend `JobType`:
  - `TRAINING = "training"`
  - `DATASET_IMPORT = "dataset_import"`
  - `MODEL_IMPORT = "model_import"`

## 2) Discriminated payload design (required)

`Job.payload` should be modeled as a discriminated union (same design style as camera schemas using `Field(discriminator=...)`).

Conceptual shape:

- `TrainingJobPayload(type="training", ...)`
- `DatasetImportJobPayload(type="dataset_import", ...)`
- `ModelImportJobPayload(type="model_import", ...)`

This replaces relying on untyped `dict` payload semantics.

## 3) Shared import step enum

Use a shared step enum for import orchestration:

- `UPLOADED`
- `DETECTING_SOURCE`
- `GENERATING_DRAFT_MANIFEST`
- `WAITING_FOR_USER_INPUT`
- `VALIDATING_FINAL_INPUT`
- `READY_TO_COMMIT`
- `EXTRACTING_ARCHIVE`
- `REGISTERING_RESOURCE`
- `IMPORTING_RESOURCE`
- `COMPLETED`
- `FAILED`
- `CANCELED`

Cancellation policy (v1):

- cancel/interrupt is allowed only **before** `IMPORTING_RESOURCE` starts.
- once full import has started (`IMPORTING_RESOURCE` and later), cancel request is rejected with conflict.

## 4) Dataset import payload proposal

`DatasetImportJobPayload` fields:

- `type: "dataset_import"`
- `project_id: UUID`
- `step: ImportStep`
- `uploaded_archive_path: str`
- `archive_sha256: str | None`
- `source_hint: str | None`
- `detected_source: DatasetImportSource | None`
- `detected_adapter: str | None` (e.g., `StudioAdapter`, `LeRobotV2Adapter`)
- `dataset_manifest_draft: DatasetManifestV1 | None`
- `validation_report: ImportValidationReport | None`
- `required_user_inputs: list[str]`
- `finalize_input: DatasetImportFinalizeInput | None`
- `source_original_dataset_uuid: UUID | None` (if provided by source manifest)
- `result_dataset_id: UUID | None`

## 5) Model import payload proposal

`ModelImportJobPayload` fields:

- `type: "model_import"`
- `project_id: UUID`
- `step: ImportStep`
- `uploaded_archive_path: str`
- `archive_sha256: str | None`
- `source_hint: str | None`
- `detected_source: ModelImportSource | None`
- `detected_adapter: str | None` (e.g., `StudioAdapter`)
- `model_manifest_draft: ModelManifestV1 | None`
- `validation_report: ImportValidationReport | None`
- `required_user_inputs: list[str]`
- `finalize_input: ModelImportFinalizeInput | None`
- `source_original_model_uuid: UUID | None` (if provided by source manifest)
- `result_model_id: UUID | None`

## 6) Job progress/message conventions

- `progress` reflects coarse stage progression (0..100).
- `message` reflects current high-level stage.
- `extra_info` remains optional for transient telemetry only (bytes processed, file counts); orchestration state should live in typed payload by default.

---

## Manifest Proposals

Manifest objects exist in two forms:

1. **Draft canonical manifest** generated from source adapter (may have unknowns).
2. **Final canonical manifest** after user finalize input.

## 1) Dataset manifest (`DatasetManifestV1`)

```json
{
  "manifest_version": "1.0",
  "resource_type": "dataset",
  "source": {
    "source_type": "studio",
    "adapter": "StudioAdapter",
    "source_format_version": "1.0",
    "source_identifier": "optional external ref",
    "original_dataset_uuid": "optional-uuid-from-source"
  },
  "identity": {
    "suggested_name": "cube_grabber",
    "default_task": "pick_cube"
  },
  "capture": {
    "fps": 30,
    "episode_count": 124,
    "frame_count": 89321
  },
  "robots": [
    {
      "robot_type": "so100",
      "joint_schema": ["joint_1", "joint_2", "joint_3"]
    }
  ],
  "cameras": [
    {
      "camera_name": "front",
      "resolution": { "width": 1280, "height": 720 },
      "fps": 30
    }
  ],
  "schema": {
    "feature_schema_hash": "sha256:...",
    "lerobot_codebase_version": "v3.0"
  },
  "integrity": {
    "archive_sha256": "...",
    "archive_size_bytes": 123456789
  },
  "warnings": [],
  "missing_fields": ["environment_id"]
}
```

## 2) Model manifest (`ModelManifestV1`)

```json
{
  "manifest_version": "1.0",
  "resource_type": "model",
  "source": {
    "source_type": "studio",
    "adapter": "StudioAdapter",
    "source_format_version": "1.0",
    "source_identifier": "optional external ref",
    "original_model_uuid": "optional-uuid-from-source"
  },
  "identity": {
    "suggested_name": "cube_grasp_policy",
    "policy": "act"
  },
  "artifact": {
    "export_format": "onnx",
    "entrypoint": "policy.onnx",
    "files": ["policy.onnx", "manifest.json"]
  },
  "compatibility": {
    "input_schema_hash": "sha256:...",
    "output_schema_hash": "sha256:...",
    "expected_robot_type": "so100"
  },
  "lineage": {
    "dataset_signature": null,
    "snapshot_signature": null,
    "parent_model_signature": null
  },
  "integrity": {
    "archive_sha256": "...",
    "archive_size_bytes": 45678912
  },
  "warnings": [],
  "missing_fields": []
}
```

---

## Source Adapter Design

## 1) Dataset source enum

- `studio`
- `lerobot_v2`
- `lerobot_v3`
- `trossen_sdk`
- `unknown`

## 2) Model source enum

- `studio`
- `unknown` (expand later)

## 3) Adapter interface

Each adapter implements:

1. `detect(staging_path) -> confidence/source_type`
2. `parse_to_draft_manifest(staging_path) -> DatasetManifestV1 | ModelManifestV1`
3. `validate(manifest, finalize_input) -> ImportValidationReport`
4. `commit(...) -> created_resource_id`

In Phase 1/2, implement only `geti_export` adapters.

### Required adapter classes

Dataset import adapters:

- `StudioAdapter`
- `LeRobotV2Adapter`
- `LeRobotV3Adapter`
- `TrossenSDKAdapter`

Model import adapters (initially):

- `StudioAdapter`

Execution rule:

- `DatasetImportWorker` must detect source and dispatch to one of dataset adapters.
- `ModelImportWorker` must detect source and dispatch to one of model adapters.
- workers orchestrate steps; adapters own source-specific parsing/validation/commit behavior.

---

## Scheduler / Worker Integration Proposal

`application/backend/src/core/scheduler.py` should start import worker processes in addition to training worker(s).

Proposed incremental change:

1. Add `DatasetImportWorker` process class.
2. Add `ModelImportWorker` process class.
2. In `Scheduler.start_workers()`, start:
   - `TrainingWorker` (existing)
   - `DatasetImportWorker` (new)
   - `ModelImportWorker` (new)
3. `DatasetImportWorker` polling loop:
   - fetch oldest pending `dataset_import` job,
   - detect source,
   - dispatch adapter (`StudioAdapter` / `LeRobotV2Adapter` / `LeRobotV3Adapter` / `TrossenSDKAdapter`),
   - execute step machine,
   - write job updates,
   - emit `JOB_UPDATE` events.
4. `ModelImportWorker` polling loop:
   - fetch oldest pending `model_import` job,
   - detect source,
   - dispatch adapter (`StudioAdapter` initially),
   - execute step machine,
   - write job updates,
   - emit `JOB_UPDATE` events.

No API call should directly execute extraction/registration logic; APIs only enqueue/update jobs.

---

## Detailed Endpoint Contracts

## 1) Dataset upload endpoint

**Endpoint**

- `POST /api/projects/{project_id}/imports/datasets`

**Success**

- `202 Accepted` + created job

**Errors**

- `400` invalid/missing file
- `413` upload too large
- `415` unsupported archive type
- `409` duplicate active import for same archive checksum + project (optional dedupe)

Additional duplicate guard:

- if source manifest provides `original_dataset_uuid` and that source UUID was already imported, reject with `409`.

## 2) Model upload endpoint

**Endpoint**

- `POST /api/projects/{project_id}/imports/models`

Same response/error contract as dataset upload.

Additional duplicate guard:

- if source manifest provides `original_model_uuid` and that source UUID was already imported, reject with `409`.

## 3) Dataset finalize endpoint

**Endpoint**

- `POST /api/jobs/{job_id}:finalize-dataset-import`

**Rules**

- only valid for `dataset_import` jobs,
- only valid from `WAITING_FOR_USER_INPUT` step,
- validates payload shape as `DatasetImportFinalizeInput`.

**Errors**

- `404` job not found
- `409` job in invalid step/state
- `422` invalid finalize data

## 4) Model finalize endpoint

**Endpoint**

- `POST /api/jobs/{job_id}:finalize-model-import`

**Rules**

- only valid for `model_import` jobs,
- only valid from `WAITING_FOR_USER_INPUT` step,
- validates payload shape as `ModelImportFinalizeInput`.

**Errors**

- `404` job not found
- `409` job in invalid step/state
- `422` invalid finalize data

## 5) Job status endpoint

Reuse existing:

- `GET /api/jobs/{job_id}`
- `GET /api/jobs`
- WebSocket `/api/jobs/ws`

No new status endpoint required for v1.

---

## Validation Report Shape (shared)

`ImportValidationReport` proposal:

- `is_valid: bool`
- `blocking_errors: list[{code, message, path?}]`
- `warnings: list[{code, message, path?}]`
- `required_user_inputs: list[str]`
- `compatibility_summary: dict`

This object is stored in import payload and returned through job fetch/websocket updates.

---

## Data and Safety Rules

1. Never extract archives outside staging/final storage roots.
2. Path traversal protection required for ZIP members.
3. Enforce both compressed and uncompressed size limits.
4. Use atomic move from staging to final location.
5. Write DB record only after successful extraction/validation.
6. On failure, leave clear `message`, `validation_report`, and cleanup status in payload.

Implementation notes for next hardening increment:

- **Stricter ZIP extraction policy**:
  - reject absolute paths, `..` traversal, symlink entries, and unsupported file types,
  - enforce max file count + max single-file size + total uncompressed size,
  - extract into isolated temp dir, then validate expected structure before move.
- **Improve manifest extraction from archive context**:
  - parse source manifest from known paths (`studio-dataset-manifest.json`, `geti-dataset-manifest.json`) and root folder variants,
  - merge manifest metadata with inferred archive structure details,
  - prefer canonicalized fields (`original_dataset_uuid`, source/version, robot/camera/fps hints) and surface provenance warnings when partial/ambiguous.

---

## Incremental Implementation Checklist (API + Job first)

1. Extend `JobType` and add discriminated import payload models in `schemas/job.py`.
2. Add job service methods:
   - create dataset/model import draft jobs,
   - finalize dataset import jobs,
   - finalize model import jobs,
   - fetch pending import jobs by type.
3. Add import API routes (project-scoped upload + separate job finalize endpoints).
4. Add `DatasetImportWorker` + `ModelImportWorker` and register both in `Scheduler.start_workers()`.
5. Implement phase-1 adapter (`StudioAdapter` dataset) with draft manifest + validation.
6. Emit `JOB_UPDATE` events at each step transition.
7. Keep model import APIs/payloads in place (phase-2 execution logic can initially return not implemented step if needed).

---

## Open Questions (API + Job Design)

| ID | Question | Status |
|---|---|---|
| OQ-1 | Split finalize APIs into `:finalize-dataset-import` and `:finalize-model-import`? | DECIDED (split) |
| OQ-2 | Upload transport strategy? | DECIDED (v1 multipart only; keep extensible for future presigned/object-store) |
| OQ-3 | Worker topology for imports? | DECIDED (separate `DatasetImportWorker` and `ModelImportWorker`) |
| OQ-4 | Duplicate handling strategy? | DECIDED (no active-job checksum dedupe; enforce duplicate check by source original UUID when present) |
| OQ-5 | Payload vs `extra_info`? | DECIDED (prefer typed payloads; `extra_info` telemetry only) |
| OQ-6 | Import cancellation policy? | DECIDED (allowed only before full import starts) |
