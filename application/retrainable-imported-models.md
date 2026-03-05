# Retraining Imported Models — Investigation Findings

## Problem Statement

After importing a model (either from HuggingFace or re-importing a Studio export),
the user may want to retrain it. The training worker calls `load_policy(base_model)`
which dispatches to `ACT.load_from_checkpoint()` (or `SmolVLA`, `Pi0`). When the
model directory has no `model.ckpt` (only `exports/torch/{policy}.pt`), we added a
fallback to load from the export artifact. This failed with **`Unexpected key(s) in
state_dict`** errors for all three test models.

## Root Cause: Two Incompatible Policy Class Hierarchies

There are two distinct class hierarchies that produce ACT/SmolVLA policies, each with
a different module nesting structure and therefore different `state_dict` key prefixes.

### Hierarchy A — Native physicalai policies

```
physicalai.policies.act.ACT  (Export, Policy → L.LightningModule)
  └── self.model : ACTModel (nn.Module)
        ├── self._input_normalizer : FeatureNormalizeTransform
        ├── self._output_denormalizer : FeatureNormalizeTransform
        └── self._model : _ACT (nn.Module)
              ├── self.vae_encoder
              ├── self.backbone
              ├── self.encoder
              ├── self.decoder
              └── self.action_head
```

**state_dict key pattern:** `model._model.vae_encoder...`, `model._input_normalizer...`

- Has `Export` mixin → `to_torch()`, `export()` methods
- Has `training_step` that expects `Observation`, returns `{"loss": loss}`
- `load_from_checkpoint` is Lightning's built-in (reads `hyper_parameters`)
- Used by: natively trained models, Studio exports of native models

### Hierarchy B — LeRobot wrapper policies

```
physicalai.policies.lerobot.universal.LeRobotPolicy  (Policy → L.LightningModule, LeRobotFromConfig)
  └── self._lerobot_policy : lerobot ACTPolicy (PreTrainedPolicy, nn.Module)
        └── self.model : lerobot ACT (nn.Module)
              ├── self.vae_encoder
              ├── self.backbone
              ├── self.encoder
              ├── self.decoder
              └── self.action_head
```

**state_dict key pattern:** `_lerobot_policy.model.vae_encoder...`

- Does **NOT** have `Export` mixin
- Has `from_pretrained()` for loading HuggingFace models
- Has `training_step` that accepts `Observation | dict`, returns bare tensor
- `load_from_checkpoint` is a custom override (reads `model_config` + `policy_name`)
- Used by: HuggingFace/LeRobot model imports

### Why inference works but retrain fails

**TorchAdapter (inference)** reads `metadata.yaml` → gets the `policy_class` string
(e.g., `"physicalai.policies.lerobot.universal.LeRobotPolicy"`) → resolves that exact
class → calls its `load_from_checkpoint()` → keys match because the **same class**
saved them.

**`load_policy()` (retrain)** ignores `metadata.yaml` and hardcodes class dispatch
based on `model.policy` string (e.g., `"act"` → native `ACT`). When the `.pt` was
saved by `LeRobotPolicy`, the keys don't match what native `ACT` expects.

## Import Scenarios and Retrainability

| Scenario | `.pt` saved by | state_dict keys | Retrainable? | Why |
|---|---|---|---|---|
| Native model (has `model.ckpt`) | Lightning `ModelCheckpoint` | `model._model.vae_encoder...` | **Yes** | Standard training flow, `model.ckpt` is a full Lightning checkpoint |
| Studio re-import of native model | Native `ACT.to_torch()` | `model._model.vae_encoder...` | **Yes** | Keys match native `ACT.load_from_checkpoint()` |
| HuggingFace import | `LeRobotPolicy` (manual) | `_lerobot_policy.model.vae_encoder...` | **No** | Key mismatch + missing `Export` mixin + different `setup()` |
| Studio re-import of HF-imported model | `LeRobotPolicy` (manual) | `_lerobot_policy.model.vae_encoder...` | **No** | Same `.pt` from the HF import, same problems |

## Blockers for Retraining HF-Imported Models

Even if we resolve the class dispatch to load via `LeRobotPolicy.load_from_checkpoint()`,
three additional blockers prevent retraining:

### 1. Missing `Export` mixin

The training worker calls `policy.export(export_dir, backend=backend)` after
`trainer.fit()`. `LeRobotPolicy` does not inherit from `Export`, so this raises
`AttributeError`.

### 2. Different `setup()` behavior

- **Native ACT `setup()`**: Calls `reformat_dataset_to_match_policy()` for delta
  timestamp handling.
- **LeRobotPolicy `setup()`**: Skips this step, extracts features/stats differently.

Skipping delta timestamp reformatting could cause incorrect temporal indexing
during training.

### 3. Checkpoint format incompatibility

A model retrained via `LeRobotPolicy` would save a LeRobot-format checkpoint
(with `model_config` + `policy_name` keys), which is incompatible with the native
`load_from_checkpoint` used elsewhere in the application.

## Decision

- **Block retrain for HF-imported models** (and Studio re-imports of HF models).
- **Allow retrain for Studio re-imports of native models** — these have
  self-consistent keys and the export `.pt` is loadable by native
  `ACT.load_from_checkpoint()`.
- Distinguish between the two via the `model.properties["source"]` field:
  - `"huggingface"` → not retrainable
  - Absent or other values → retrainable

## Future Work

To enable retraining of HF-imported models, one of these approaches would be needed:

1. **State dict key remapping bridge**: Load via `LeRobotPolicy`, extract weights,
   remap keys to native structure, inject into a fresh native `ACT`. Fragile (requires
   per-policy mapping tables) and would also need to handle the normalizer/denormalizer
   state that exists in native ACT but not in LeRobot's structure.

2. **Make `LeRobotPolicy` support the full train+export flow**: Add `Export` mixin (or
   equivalent), align `setup()` behavior, ensure checkpoint format compatibility. This
   is a library-level change.

3. **Convert HF models to native format at import time**: Instead of saving the
   LeRobot-style `.pt`, convert the loaded model to a native `ACT` instance and use
   `ACT.to_torch()` to save. This would require key remapping similar to option 1 but
   done once at import time rather than at retrain time.
