# Policy Configuration API Handoff

## Context

The backend training worker currently creates policy instances with hard-coded defaults in
`application/backend/src/models/utils.py`:

```python
if model.policy == "act":
    return ACT(compile_model=compile_model)
if model.policy == "pi0":
    return Pi0(compile_model=compile_model)
if model.policy == "pi05":
    return Pi05(pretrained_name_or_path="lerobot/pi05_base", compile_model=compile_model)
if model.policy == "smolvla":
    return SmolVLA(pretrained_name_or_path="lerobot/smolvla_base", compile_model=compile_model)
```

Training outputs include `version_0/metrics.csv` and `version_0/hparams.yaml`. These files are produced by
Lightning's `CSVLogger`, which is configured in `application/backend/src/workers/training_worker.py`:

```python
csv_logger = CSVLogger(cache_path.parent, name=cache_path.stem)
```

The backend reads `hparams.yaml` through `ModelService.get_hparams()` in
`application/backend/src/services/model_service.py`.

## How `hparams.yaml` Is Populated

`hparams.yaml` is not a declarative schema of configurable values. It is Lightning logger output based on each
policy's `self.hparams` at training time.

The values come from:

- Policy constructor parameters captured by `self.save_hyperparameters(...)`.
- A manually added nested `config` dictionary, usually `self.hparams["config"] = self.config.to_dict()`.
- Runtime data added later by policies, such as `dataset_stats`.

Examples:

- `library/src/physicalai/policies/act/policy.py` saves constructor args except `config` and `compile_model`, then stores `config`.
- `library/src/physicalai/policies/pi0/policy.py` saves constructor args except `config`, then stores `config`.
- `library/src/physicalai/policies/pi05/policy.py` saves constructor args except `config`, `pretrained_name_or_path`, and `compile_model`, then syncs resolved config values.
- `library/src/physicalai/policies/smolvla/policy.py` follows the same general pattern as Pi05.

Because this file is output state, it should not be treated as the API contract for training input.

## Policy vs Model

The library uses two related concepts that are easy to confuse.

`policy.py` contains the top-level policy class. This is the object the backend trains with Lightning.

A policy is a `LightningModule` and owns:

- The neural network model.
- Preprocessing and postprocessing.
- Training and validation steps.
- Optimizer and scheduler setup.
- Checkpoint loading.
- Export behavior.
- Runtime action selection through methods such as `select_action()`.

`model.py` contains the lower-level `torch.nn.Module` architecture. It usually performs the actual forward pass and
loss computation, but it is not what the backend passes to `Trainer.fit(...)`.

For backend training API work, construct policies, not just models.

## Existing Config Classes

The library already has policy config dataclasses, for example:

- `library/src/physicalai/policies/act/config.py` -> `ACTConfig`
- `library/src/physicalai/policies/pi0/config.py` -> `Pi0Config`
- `library/src/physicalai/policies/pi05/config.py` -> `Pi05Config`
- `library/src/physicalai/policies/smolvla/config.py` -> `SmolVLAConfig`

These configs inherit from `physicalai.config.Config`, which provides:

- `to_dict()`
- `from_dict()`
- `to_jsonargparse()`
- YAML save/load helpers

The config dataclasses are a good source of truth for:

- Field names.
- Python types.
- Defaults.
- Some enum-like constraints through `Literal[...]`.
- Runtime validation in `__post_init__`.

They are not currently complete REST API schemas because they do not provide:

- Structured per-field OpenAPI descriptions.
- Numeric constraints such as `ge`, `gt`, `le`.
- A separation between user-editable parameters and internal/runtime fields.
- A consistent direct construction path for all top-level policy classes.

Docstrings contain useful descriptions, but FastAPI will not automatically translate those into field-level OpenAPI
metadata.

## Current Config-Based Construction Support

The library has generic construction helpers in `library/src/physicalai/config/instantiate.py` and
`library/src/physicalai/config/mixin.py`.

The important pieces are:

- `instantiate_obj(config, target_cls=...)`
- `FromConfig.from_config(...)`
- `FromConfig.from_dataclass(...)`

However, support is inconsistent across policies:

- `Pi0` inherits `FromConfig`, so config-based construction is directly available there.
- First-party `ACT`, `Pi05`, and `SmolVLA` do not currently appear to inherit `FromConfig`.
- Some lower-level model classes are closer to config-driven construction, but the backend needs top-level policies.

Do not assume this works for every policy:

```python
policy = PolicyClass.from_config(config)
```

## Important Mismatches To Handle

The config dataclasses are not always a one-to-one match for top-level policy constructors.

Examples:

- `ACTConfig` includes `input_features` and `output_features`. For backend training these should usually be inferred from the dataset/datamodule, not supplied by the REST payload.
- `compile_model` already exists on `TrainJobPayload`, so exposing it again through policy config would duplicate state.
- `SmolVLAConfig` includes fields such as `empty_cameras` and `adapt_to_pi_aloha`; confirm whether each maps to the top-level `SmolVLA(...)` constructor before passing it through.
- Pretrained policies need special handling. The backend currently creates Pi05 and SmolVLA with `pretrained_name_or_path` values, but those are intentionally not normal user-editable hyperparameters in the existing worker.

## Recommended API Direction

Reuse the library config dataclasses as the canonical source of defaults and basic types, but introduce a small backend
adapter layer rather than passing config dictionaries blindly into policy constructors.

Suggested shape:

```python
class TrainJobPayload(BaseModel):
    project_id: UUID
    dataset_id: UUID
    policy: str
    model_name: str
    # existing training controls...
    policy_config: ACTConfig | Pi0Config | Pi05Config | SmolVLAConfig | None = None
```

Exact typing may need a discriminator or policy-specific request schema so OpenAPI presents this clearly.

Then adapt policy setup to merge validated config values with backend-managed values:

```python
def setup_policy(model: Model, *, compile_model: bool = False, policy_config: Config | None = None) -> Policy:
    ...
```

The adapter should:

- Start from `policy_config.to_dict()` if present.
- Drop internal/runtime fields such as `input_features`, `output_features`, and possibly `compile_model`.
- Set backend-managed arguments such as `compile_model` explicitly from `TrainJobPayload`.
- Preserve current pretrained defaults for Pi05 and SmolVLA unless the API intentionally exposes pretrained source selection.
- Instantiate the policy with filtered, tested kwargs.

Conceptual example:

```python
kwargs = policy_config.to_dict() if policy_config else {}
kwargs.pop("input_features", None)
kwargs.pop("output_features", None)
kwargs.pop("compile_model", None)
kwargs["compile_model"] = compile_model

if model.policy == "act":
    return ACT(**kwargs)
if model.policy == "pi05":
    return Pi05(pretrained_name_or_path="lerobot/pi05_base", **kwargs)
if model.policy == "smolvla":
    return SmolVLA(pretrained_name_or_path="lerobot/smolvla_base", **kwargs)
```

This should be implemented explicitly per policy and covered by tests. Avoid a generic `**config.to_dict()` path without
per-policy filtering.

## Better Library-Level Improvement

A cleaner long-term approach is to make all top-level policy classes expose a consistent construction API:

```python
ACT.from_config(config)
Pi0.from_config(config)
Pi05.from_config(config)
SmolVLA.from_config(config)
```

Options:

- Have each top-level policy inherit `FromConfig` where safe.
- Add policy-specific `from_config` methods that handle filtering and pretrained defaults correctly.
- Add tests proving every public config can construct the corresponding top-level training policy.

If this is added to the library, the backend can call a uniform API and avoid knowing policy constructor details.

## OpenAPI Considerations

FastAPI can infer basic schemas from dataclass types, but the current dataclasses will not produce rich OpenAPI docs.

For better OpenAPI output, consider enhancing the library config classes with structured metadata:

```python
optimizer_lr: float = field(
    default=2.5e-5,
    metadata={"description": "Learning rate for the optimizer"},
)
```

Potential improvements:

- Add `metadata={"description": ...}` to dataclass fields.
- Add validation metadata if a chosen schema generator can consume it.
- Convert configs to Pydantic dataclasses or Pydantic models if rich OpenAPI schemas become more important than plain dataclass simplicity.
- Add a library function that exposes a policy parameter schema for UI/API use.

Do not rely on class docstrings alone for OpenAPI field descriptions.

## Suggested Implementation Steps

1. Audit each config class against the corresponding top-level policy constructor.
2. Decide which fields are public REST-editable and which are backend/library-managed.
3. Add `policy_config` to `TrainJobPayload` with a type strategy that gives acceptable OpenAPI output.
4. Update `setup_policy()` to accept and apply a config object through an explicit adapter.
5. Update `TrainingWorker` to pass `payload.policy_config` into `setup_policy()`.
6. Add tests for payload parsing and OpenAPI schema generation.
7. Add tests that each policy receives the expected constructor kwargs.
8. Add validation tests for config-level errors, such as `n_action_steps > chunk_size`.
9. Consider adding `from_config()` support to all top-level library policies to reduce backend-specific construction logic.

## Testing Targets

Backend tests should cover:

- `TrainJobPayload` accepts valid config for each supported policy.
- Invalid config fails before job execution.
- `setup_policy()` filters unsupported/internal fields.
- `compile_model` continues to be sourced from the existing payload field, not duplicated config state.
- Pi05 and SmolVLA keep current pretrained defaults.
- Generated OpenAPI schema is acceptable for UI consumption.

Library tests should cover, if `from_config()` is added:

- `ACT.from_config(ACTConfig(...))` returns a trainable top-level policy.
- `Pi0.from_config(Pi0Config(...))` continues to work.
- `Pi05.from_config(Pi05Config(...))` returns a trainable top-level policy.
- `SmolVLA.from_config(SmolVLAConfig(...))` returns a trainable top-level policy.
- Config-only fields and runtime-managed fields are handled intentionally.

## Recommendation

Use the existing library config dataclasses as the canonical source of defaults and types. Add a small, explicit adapter
for backend policy construction first. In parallel or later, improve the library by adding consistent top-level
`from_config()` support and structured field metadata so FastAPI/OpenAPI integration can become cleaner over time.
