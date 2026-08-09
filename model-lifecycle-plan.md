# Model Lifecycle Plan

How Physical AI Studio should manage the full lifecycle of trained models: from
a single trained checkpoint to a package that holds multiple training artifacts
(checkpoints), algorithm variants (e.g. SnapFlow), and inference artifacts
(optimized exports per backend and precision), with import, download, retrain,
and prune operations on top.

This document is a design plan. It describes the current state, the problem
space, the proposed data model, and a phased implementation roadmap.

---

## Current state

Today a **Model** is a single DB row pointing at a directory. Training
produces one final checkpoint and a set of exports; everything else is derived
from the filesystem.

### On-disk layout

```
models/{id}/
  model.ckpt                # final Lightning checkpoint (physicalai engine)
  lerobot/                  # raw LeRobot checkpoint (lerobot engine)
  exports/{backend}/        # one dir per export backend: torch, openvino, onnx, executorch
  version_0/                # training logs (metrics.csv, hparams.yaml)
  snapshot_*/               # optional dataset snapshot
```

### Behavior today

- **Training** (`application/backend/src/training/job.py`): the physicalai
  engine runs Lightning and keeps exactly one final checkpoint (`model.ckpt`);
  the LeRobot engine
  (`application/backend/src/training/lerobot.py`) trains with LeRobot's own
  loop, saves one final checkpoint, and collapses it into `lerobot/` at
  publish time.
- **Exports** are written once at the end of training, flat per backend, at a
  single implicit precision (the UI hardcodes "FP16").
- **Retrain / continue training**: a row-menu action that reopens the train
  dialog with the base model preselected. It always resumes from the base
  model's *latest* checkpoint, offers **no checkpoint picker** and **no
  optimizer/LR-reset option**, and is forced local-only (the remote trainer
  protocol can receive a dataset but not a base checkpoint).
- **Download**: a full-package zip (`GET /api/models/{id}/download`) or a
  per-backend export zip (`GET /api/models/{id}/exports/{backend}/download`).
- **Import**: CLI-only (`models import-dir`), and it *requires* a torch export
  plus `version_0/` logs. There is no UI import flow.
- **Re-export / compress**: `POST /api/models/{id}:export` creates a **new
  Model row** (a child) with fresh exports, rather than adding an artifact to
  the existing model.
- **Remote training**: the trainer runs the same `run_training_job`, zips the
  whole model directory, and the studio downloads and extracts it into
  `models/{id}`. Chunked, resumable transfers both ways.

### The core problem

A Model conflates three things that are really separate:

1. **A training run**, which can produce *multiple* training artifacts
   (intermediate checkpoints at chosen epochs).
2. **A trainable artifact** — a PyTorch checkpoint that can be resumed, have
   algorithms applied on top of it (SnapFlow, RTC), and be exported.
3. **Inference artifacts** — optimized exports keyed by
   `(backend × precision)`.

Users need to train, keep several checkpoints, apply custom algorithms,
export from any of those checkpoints to multiple backends and precisions,
download any of them, import models (trainable, inference-only, or both),
continue training from a chosen checkpoint (with or without resetting the
optimizer), prune to save disk, and do all of this on local or remote
trainers.

---

## Proposed model

**Keep one Model row = a "model package".** Artifacts are files indexed by a
`manifest.json`; capabilities (`trainable`, `inference`, `algorithms`) are
computed from content. This mirrors how `available_backends` already works and
keeps remote transfer trivial, because the model directory *is* the package.
No DB table migration is required.

Two content-derived artifact concepts:

- **TrainingArtifact** — a PyTorch checkpoint:
  - `final` — the last epoch's checkpoint (the package's "latest").
  - `intermediate` — a saved checkpoint at a chosen epoch.
  - `variant` — a checkpoint produced by applying an algorithm (e.g.
    SnapFlow) to a source checkpoint; still PyTorch, still exportable and
    resumable.
  - Each has an epoch-first ID, and a `has_training_state` flag describing
    whether it carries optimizer + LR-scheduler state (the property that makes
    it *exactly* resumable; "gradient information" is not persisted — the
    optimizer/LR state is what enables exact resume).
- **InferenceArtifact** — an export keyed by
  `(source_checkpoint × backend × precision)`, produced by a background job.

Capabilities fall out of content:
- `trainable` = at least one checkpoint exists.
- `inference` = at least one export exists.
- `algorithms` = the set of variant checkpoints.

An imported OpenVINO-only package is therefore inference-only: the UI disables
Retrain and hides training tabs, while Run and formats still work.

---

## Package layout

```
models/{id}/
  manifest.json                          # artifact index (source of truth)
  model.ckpt                             # physicalai: symlink → checkpoints/epoch=10.ckpt
                                         # lerobot: converted final (real file, back-compat)
  checkpoints/
    epoch=02.ckpt                        # physicalai intermediate
    epoch=10.ckpt                        # final epoch; manifest.latest
    epoch=10-snapflow.ckpt               # algorithm variant (applied: [snapflow]); chainable
    epoch=NN/                            # lerobot raw: pretrained_model/ + training_state/
  lerobot/                               # symlink → checkpoints/epoch=NN/ (back-compat resume)
  exports/
    epoch=02/openvino/fp16/              # namespaced exports (non-latest sources)
    epoch=10/torch/fp16/
    torch/fp16/                          # symlink → epoch=10/torch/fp16/  (canonical "latest")
    openvino/fp16/                       # (same, latest source)
  version_0/                             # training logs
```

Design notes:

- **Epoch-first checkpoint names.** The epoch is always present, even for
  variants (`epoch=10-snapflow.ckpt`); chained algorithms extend the name
  (`epoch=10-snapflow-rtc.ckpt`). For the LeRobot engine, which saves raw
  checkpoint *directories* (`pretrained_model/` + `training_state/`), the dir
  is `checkpoints/epoch=NN/`; the step is recorded in the manifest.
- **`model.ckpt` and `lerobot/` stay as back-compat aliases** so existing
  readers (import, re-export, Runtime) keep working.
- **Canonical "latest" export via symlink.** All exports are physically
  stored under `exports/{checkpoint_id}/{backend}/{precision}/`. The canonical
  `exports/{backend}/{precision}/` is a symlink to the latest checkpoint's
  export dir, so the run-model flow and existing consumers keep working and
  the pointer never requires moving files. Legacy flat `exports/{backend}/`
  dirs are read as `source: latest` (or `unknown`).
- **Precision is first-class.** Exports are namespaced by precision (`fp16`,
  `int8`, `fp32`, …); NNCF INT8 compression slots in as `precision: int8`.

---

## Manifest schema

`manifest.json` at the model root is the source of truth for artifacts.

```json
{
  "format": "physicalai_model_package",
  "version": 1,
  "policy": "act",
  "training_engine": "physicalai",
  "training": {
    "max_steps": 5000,
    "batch_size": 8,
    "precision": "bf16-mixed"
  },
  "checkpoints": {
    "latest": "epoch=10",
    "items": [
      {
        "id": "epoch=02",
        "path": "checkpoints/epoch=02.ckpt",
        "kind": "intermediate",
        "epoch": 2,
        "step": 1000,
        "has_training_state": true,
        "applied": [],
        "size_bytes": 123
      },
      {
        "id": "epoch=10",
        "path": "checkpoints/epoch=10.ckpt",
        "kind": "final",
        "epoch": 10,
        "step": 5000,
        "has_training_state": true,
        "applied": [],
        "size_bytes": 123
      },
      {
        "id": "epoch=10-snapflow",
        "path": "checkpoints/epoch=10-snapflow.ckpt",
        "kind": "variant",
        "epoch": 10,
        "step": 5000,
        "has_training_state": false,
        "applied": ["snapflow"],
        "source": "epoch=10",
        "size_bytes": 123
      }
    ]
  },
  "exports": {
    "torch/fp16": {
      "source_checkpoint": "epoch=10",
      "path": "exports/torch/fp16",
      "precision": "fp16",
      "size_bytes": 456,
      "file_count": 2,
      "io_spec": {}
    }
  }
}
```

A reader falls back to the filesystem for legacy models that predate the
manifest, synthesizing entries from the existing layout. No migration is
needed.

---

## Operations & backend surface

| Operation | Endpoint | Notes |
| --- | --- | --- |
| List artifacts | `GET /api/models/{id}/artifacts` | manifest-driven: checkpoints + exports + algorithms |
| Export | `POST /api/models/{id}:export` | extend existing: add `source_checkpoint` (default: latest) and `precision`. **Behavior change:** adds an artifact to the *same* model instead of creating a child Model row (an immutable artifact operation). |
| Apply algorithm | `POST /api/models/{id}:apply-algorithm` `{algorithm, source_checkpoint, name}` | background job → variant checkpoint (`applied: [snapflow]`); the variant is exportable and resumable. |
| Prune | `POST /api/models/{id}:prune` `{scope}` | three scopes, see below |
| Download | `…/checkpoints/{id}/download`, `…/exports/{backend}/{precision}/download`, full-package zip | |
| Import | `POST /api/projects/{id}/imports/models:prepare` + `:upload` (dataset-import pattern) | relax the mandatory-torch requirement so inference-only packages import |
| Retrain | `POST /api/jobs:train` payload + `checkpoint_id`, `reset_training_state`, `checkpoint_save` | new Model row (tree), `version++`, as today |

### Prune (three scopes, destructive)

Pruning runs as a background job with a confirm dialog and updates
`manifest.json`:

1. **Drop intermediates** — delete all non-latest checkpoints plus their
   namespaced exports. Fast (pure file deletion); exact-resume is preserved
   for the final checkpoint.
2. **Compact** — rewrite a checkpoint weights-only, stripping optimizer/LR
   state so `has_training_state=false`. Keeps the model trainable and
   resumable with a fresh optimizer, but it is no longer *exactly* resumable.
   Slower, because it requires a torch load/save.
3. **Prune to inference-only** — remove all training artifacts so
   `trainable=false`; only Run and exports remain.

### Remote continue-training

The remote trainer protocol currently accepts a dataset upload but not a base
checkpoint, so "continue training on a GPU box" is impossible today. This work
adds a symmetric checkpoint upload:

- Studio zips the selected checkpoint (the `.ckpt`, or a LeRobot `epoch=NN/`
  directory including `training_state/`).
- New trainer endpoint `PUT /jobs/{id}/checkpoint` (mirror of the existing
  dataset upload) streams it in; the trainer extracts it and passes
  `resume_from` into `run_training_job`.

The existing spec (`TrainingJobSpec`) remains the wire contract.

---

## Engine changes

### physicalai engine (`application/backend/src/training/job.py`)

- Switch `ModelCheckpoint` to per-epoch intermediate files
  (`filename="epoch={epoch:02d}.ckpt"`), `save_top_k=-1`, and
  `save_optimizer_state=True` so every saved checkpoint embeds optimizer + LR
  state and is exactly resumable.
- A new `checkpoint_save` config controls capture:
  `"none" | "all" | [epochs]` (e.g. `[2, 3, 6, 8]`). The final epoch is always
  saved.
- Publish writes `manifest.json` and creates the `model.ckpt` symlink to the
  final epoch checkpoint.

### LeRobot engine (`application/backend/src/training/lerobot.py`)

- Stop collapsing `checkpoints/step_*` at publish; keep every saved epoch as
  `checkpoints/epoch=NN/` (each with `pretrained_model/` and `training_state/`).
- Set `save_freq = steps_per_epoch` when intermediates are requested.
- Keep `lerobot/` as a symlink to the latest epoch directory for back-compat
  resume, and keep the `model.ckpt` conversion (from
  `lerobot_to_lightning`) unchanged.

### Both engines

- At publish, export the final checkpoint to the configured backends at the
  default precision, written to the namespaced location with the canonical
  `exports/{backend}/{precision}/` symlink.

---

## Schema & payload additions

- `TrainJobPayload` / `TrainingJobSpec`:
  - `checkpoint_id` — which checkpoint of `base_model` to start from (default:
    latest).
  - `reset_training_state: bool` — `true`: load weights only, fresh
    optimizer/LR schedule; `false`: exact resume from the checkpoint's
    optimizer/LR state (errors gracefully if the checkpoint lacks it).
  - `checkpoint_save: "none" | "all" | list[int]` — intermediate checkpoint
    capture policy.
- `Model.properties` needs no required additions; the manifest is the source
  of truth.
- `available_backends` becomes manifest-driven with a filesystem fallback.

---

## UI changes

- **Retrain dialog**: a checkpoint picker (final / intermediates / variants),
  a "Resume optimizer & LR schedule" toggle, and a "Save intermediate
  checkpoints" control (None / Every epoch / custom list). Remote retraining
  is enabled once the remote checkpoint-upload lands.
- **Formats tab**: cards per `(backend × precision)` read from the manifest;
  precision is no longer hardcoded to FP16. Add an "Export from checkpoint"
  affordance.
- **Checkpoints tab** (new, in the model detail panel): lists final /
  intermediate / variant checkpoints with epoch, size, a training-state badge,
  and per-checkpoint Download + "Retrain from here".
- **Import dialog**: a two-step zip upload mirroring the dataset-import flow,
  showing the detected capabilities (trainable / inference-only) after import.
- **Prune dialog**: the three scopes with a size-impact estimate before
  confirming.
- **Model rows**: an inference-only badge for models without training
  artifacts (Retrain disabled).

---

## Phasing (each phase independently shippable)

1. **Manifest + intermediate capture.** Per-epoch checkpoint saving in both
   engines, `manifest.json` writing, the artifacts endpoint, computed fields,
   and the back-compat filesystem reader.
2. **Retrain control.** Checkpoint picker + optimizer/LR-reset toggle, local.
3. **Export as an artifact op.** Source checkpoint + precision, the canonical
   symlink layout, per-`(backend × precision)` formats tab, and the new
   download endpoints.
4. **Apply-algorithm jobs.** SnapFlow (and friends) produce immutable variant
   checkpoints that are exportable and resumable.
5. **Import via UI.** Any artifact combination, relaxed validation.
6. **Remote continue-training.** Checkpoint upload to the trainer.
7. **Prune.** All three scopes plus the size-impact UX.

---

## Edge cases & open considerations

- **Size.** Optimizer-state checkpoints are roughly 2–3× the weights (a Pi0.5
  optimizer state can be ~400 MB). Intermediate capture, download toggles, and
  the prune/compact operations exist to manage this.
- **Legacy models.** The manifest reader synthesizes entries from the existing
  layout (`model.ckpt`, `lerobot/`, flat `exports/{backend}/`), so nothing
  breaks.
- **Latest pointer.** Fixed at publish time to the final epoch. Algorithm
  variants never become `latest`; the canonical export symlink is set once.
- **Imported training provenance.** Imported models keep a synthetic completed
  training job for the Training Details tab today; the manifest's optional
  `training` section is the path to carrying real metadata for imports.
- **Mid-run cancel resume.** Out of scope: job caches are wiped on
  CANCELED/FAILED today. Resume-from-interrupted runs would be a separate
  follow-up if desired.
