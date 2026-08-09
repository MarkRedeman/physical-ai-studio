# LeRobot Training Engine

Let the user choose, in the Train Model dialog, between the **physicalai-train**
library and **LeRobot** as the training engine. For LeRobot, the user only picks
the policy; batch size, step budget, and everything else is automated from
best practices (see
https://github.com/huggingface/lerobot/blob/main/AGENT_GUIDE.md).

After training, models are exported exactly like physicalai-train models, but
**torch only** for now.

## Design

- New `training_engine: Literal["physicalai", "lerobot"]` field threaded
  `UI -> TrainJobPayload -> TrainingJobSpec`. The existing `policy_source`
  field keeps its "which policy implementation" meaning and is untouched.
- `run_training_job` dispatches on `training_engine`; `"lerobot"` runs a
  vendored lean training loop in `application/backend/src/training/lerobot.py`
  modeled on lerobot 0.6.0's own loop (`lerobot.scripts.lerobot_train.train`),
  with cooperative `should_stop` / `report` callbacks (same `ReportFn` /
  `StopFn` contract as the physicalai path).
- Dataset snapshots are already LeRobot-layout directories on disk, so
  lerobot's `LeRobotDataset` reads them directly (no conversion).
- After training: the lerobot checkpoint is converted to `model.ckpt`
  (library `physicalai.policies.lerobot.utils.checkpoint_converter.lerobot_to_lightning`),
  the model is exported to **torch only** via the `LeRobotPolicy` wrapper
  (which only supports the torch backend), and the raw lerobot checkpoint is
  kept in the model dir under `lerobot/` so later runs can resume natively.
- lerobot/accelerate has no XPU support: the engine is gated to CUDA/CPU. The
  UI disables it on XPU-only machines; the backend rejects
  `device_type="xpu"`.
- Works on local and remote trainers automatically: `TrainingJobSpec` is the
  shared wire contract, and the trainer image already ships lerobot 0.6.0.

## Best-practice automation (AGENT_GUIDE §7)

- `epochs = 5` (module constant), `total_frames` read from the dataset's
  `meta/info.json`; `steps = epochs * ceil(total_frames / batch_size)`.
- Batch size per policy: `act -> 8`, `diffusion -> 8`, `smolvla -> 4`,
  `pi05 -> 4`.
- `scheduler_decay_steps ~= steps` (+ proportional warmup) for policies that
  expose the field; `save_freq = steps` (final checkpoint only).
- `val_split` maps to `dataset.eval_split` with a modest `eval_steps` so the
  job metrics graph keeps an eval-loss line.

## Backend changes

| File | Change |
| --- | --- |
| `application/backend/src/schemas/job.py` | Add `training_engine` to `TrainJobPayload`. |
| `application/backend/src/training/job.py` | Add `training_engine` to `TrainingJobSpec`; dispatch in `run_training_job`. |
| `application/backend/src/training/lerobot.py` (new) | Vendored lean loop + hyperparameter derivation + checkpoint publish + torch export. |
| `application/backend/src/services/training_backends/local.py` | `build_spec()` threads `payload.training_engine`; resume passes the base model dir. |
| `application/backend/src/trainer/schemas.py` | Engine-aware policy validation (physicalai: act/pi0/pi05/smolvla; lerobot: act/diffusion/smolvla/pi05). |
| `application/backend/src/workers/training_worker.py` | Persist `training_engine` in `Model.properties` (no DB migration). |
| `application/backend/src/services/model_service.py` (optional) | Expose `training_engine` in `TrainingSummary`. |
| `application/backend/tests/` | Spec round-trip, dispatch, hyperparameter derivation, device rejection, output layout. |

## UI changes (`application/ui/`)

- `src/routes/models/train-model-dialog.tsx`
  - New "Training framework" picker: **physicalai-train (Recommended)** /
    **LeRobot**.
  - LeRobot mode: curated policy cards (ACT, Diffusion, SmolVLA, Pi0.5) with
    VRAM guidance; "Advanced settings" hidden; helper text that batch size,
    steps, and precision are automated.
  - Resume (`baseModel`): engine locked to
    `baseModel.properties.training_engine ?? 'physicalai'`.
  - XPU gate: LeRobot option disabled with a tooltip when the active device is
    XPU-only.
- `train-model-dialog.test.tsx`: cover engine switch, lerobot payload, resume
  engine lock, XPU disable.
- Regenerate OpenAPI types (`npm run build:api:download && npm run build:api`).

## Verification

- Backend: `pytest application/backend/tests/training/`,
  `prek run --all-files application/backend/`.
- UI: `npm run lint`, `npm run type-check`, `npm run test:unit`,
  `npm run format:check`.

## Risks / notes

- The vendored loop must match lerobot's optimizer/scheduler/data-order
  semantics; drift is bounded by the `lerobot[dataset]==0.6.0` pin in both
  `pyproject.toml`s.
- Format cards on the model page show "Unavailable" for onnx/openvino/
  executorch on lerobot-trained models (only torch is exported). Acceptable
  for now.
- `model.ckpt` produced via `lerobot_to_lightning` doubles as the torch export
  source and keeps the Runtime `InferenceModel` load contract intact.
