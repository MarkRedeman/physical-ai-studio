# Per-policy training configuration

Implements the application portion of [issue #1349](https://github.com/open-edge-platform/physical-ai-studio/issues/1349).

## Scope

- Define one explicit Studio-compatible configuration per selectable policy in
  `application/backend/src/training/policy_catalog.py`.
- Expose `GET /api/policies/{policy}/default-training-config` without importing
  policies, requiring installed model extras, or contacting a remote trainer.
- Use Pydantic models with documented, constrained fields and generated OpenAPI
  types. Separate executable `defaults`, optional typed `help`, and ordered
  `camera_slots`. Keep existing job-schema defaults backward compatible.
- Include batch size, epochs, workers, precision, compilation, Auto batch size,
  augmentation, validation split, LoRA/DoRA, and SnapFlow settings.
- Add explicit nullable gradient-checkpointing overrides and validation-split
  seeds to the training payload, shared spec, and runner. Omitted values retain
  legacy behavior. Checkpoint-owned architecture remains unchanged on resume.
- Move SmolVLA's camera slot descriptions to the backend. The UI still derives
  the actual camera mapping from the dataset; retraining skips mapping.
- Move Hub repository requirements and AWS policy instance/backend declarations
  into the catalog, without moving access checks or AWS orchestration logic.

## UI behavior

- Fetch defaults for the selected policy and apply them once per selection.
- Switching policies resets recipe-controlled fields. Navigation, device
  changes, and background refetches do not erase user edits.
- Block progression/submission until the selected configuration is applied;
  show a recoverable error if loading fails. Ignore stale selections.
- Use policy-specific contextual help when supplied, otherwise retain generic
  help. Treat schema field descriptions separately from runtime UI guidance.
- Recipe precision replaces the unconditional device-based precision reset.
- Submit all chosen values, including recipe validation split/seed and the final
  camera mapping, through the existing persisted training-job payload.

## Recipe sources and limitations

- ACT PushT XPU: batch 32, FP32, optimizer LR 1e-5, ColorJitter, 70,000 steps.
  Studio's augmentation boolean is not exact ColorJitter parity. Preserve the
  legacy epoch budget until a reviewed epoch-based recommendation is available.
- SmolVLA PushT: batch 64, 100 epochs, FP32, image augmentation.
- Pi0.5 ALOHA LoRA: batch 16, 25 epochs, BF16 mixed, compilation enabled,
  gradient checkpointing disabled, LoRA rank/alpha 32, dropout 0.05, validation
  split 0.1 and seed 42. The library already applies the 10x LoRA LR multiplier.
- MolmoAct2 SO101: batch 16, 8 epochs, 8 workers, BF16 mixed, LoRA and gradient
  checkpointing enabled. Its checkpoint/model settings differ from Studio's;
  matching the exposed fields does not establish full recipe parity.
- XR0 SO101 uses steps, accumulation, and additional model settings. RLDX-1 has
  no corresponding YAML here. Preserve unreviewed epoch budgets and describe
  these configurations as starting points, not validated recipes.

Do not add arbitrary model kwargs, optimizer/scheduler controls, custom transform
definitions, callback configuration, dataset revisions, or validation gyms in
this change. Validation batch size already tracks training batch size by default.
Auto-resolved runtime settings are distinct from the submitted configuration.

## Future training limits

Keep execution epoch-based. Do not reinterpret legacy `max_steps` or add a
nonfunctional preferred-step field. A future typed epoch/step budget can extend
the response alongside payload, runner, progress, and UI support.

## Verification

- Backend: catalog completeness, payload validity, dependency-free lookup, 404,
  documented OpenAPI, spec/constructor/datamodule propagation, compatibility,
  and unchanged Hub/AWS behavior.
- UI: initialization, overrides, switches (including cached configurations),
  loading/errors, stale responses, refetch/device edit preservation, mapping,
  retraining, and submitted payload values.
- Regenerate OpenAPI JSON and TypeScript types, run focused backend/UI tests,
  formatting, lint, TypeScript, circular-dependency checks, and production build.
