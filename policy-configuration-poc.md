# Policy Configuration POC

## Summary

This POC restructures policy configuration for three policies:

- ACT
- Pi05
- SmolVLA

The goal was to move away from very large flat config classes and long policy constructors as the primary way to model
configuration. Instead, related settings are grouped into smaller nested config dataclasses, while preserving the old
flat constructor and checkpoint behavior as much as possible.

This is not a full API integration yet. It is a library-side proof of concept that makes policy configs more structured
and prepares them for eventual FastAPI/OpenAPI exposure.

## What Changed

### ACT

`ACTConfig` now contains grouped subconfigs:

- `ACTInputOutputConfig`
- `ACTVisionBackboneConfig`
- `ACTTransformerConfig`
- `ACTVAEConfig`
- `ACTInferenceConfig`
- `ACTOptimizerConfig`

`ACT` now accepts:

```python
ACT(act_config=ACTConfig(...))
```

For compatibility, it still accepts:

```python
ACT(config=ACTConfig(...))
ACT(chunk_size=100, optimizer_lr=1e-5, ...)
```

Structured hparams are written under:

```python
policy.hparams["act_config"]
```

Flat hparams are still written for compatibility.

### Pi05

`Pi05Config` now contains grouped subconfigs:

- `Pi05BackboneConfig`
- `Pi05InputOutputConfig`
- `Pi05FlowMatchingConfig`
- `Pi05PreprocessingConfig`
- `Pi05TrainingConfig`
- `Pi05OptimizerConfig`

`Pi05` now accepts:

```python
Pi05(pi05_config=Pi05Config(...))
```

For compatibility, it still accepts:

```python
Pi05(config=Pi05Config(...))
Pi05(chunk_size=100, optimizer_lr=2.5e-5, ...)
```

Structured hparams are written under:

```python
policy.hparams["pi05_config"]
```

Flat hparams are still written for compatibility.

### SmolVLA

`SmolVLAConfig` now contains grouped subconfigs:

- `SmolVLAInputOutputConfig`
- `SmolVLAPreprocessingConfig`
- `SmolVLAArchitectureConfig`
- `SmolVLAFlowMatchingConfig`
- `SmolVLAFineTuningConfig`
- `SmolVLAOptimizerConfig`

`SmolVLA` now accepts:

```python
SmolVLA(smolvla_config=SmolVLAConfig(...))
```

For compatibility, it still accepts:

```python
SmolVLA(config=SmolVLAConfig(...))
SmolVLA(chunk_size=100, optimizer_lr=1e-4, ...)
```

Structured hparams are written under:

```python
policy.hparams["smolvla_config"]
```

Flat hparams are still written for compatibility.

## Compatibility Strategy

The grouped config classes are the new preferred shape, but the POC intentionally keeps the old flat API working.

Each top-level config class supports both:

```python
ACTConfig(io=ACTInputOutputConfig(chunk_size=50))
```

and:

```python
ACTConfig(chunk_size=50)
```

The same pattern applies to Pi05 and SmolVLA.

Each config also exposes legacy flat properties:

```python
config.chunk_size
config.optimizer_lr
config.freeze_vision_encoder
```

These properties delegate to the grouped subconfig. This keeps existing policy/model code mostly unchanged.

Each config also has:

```python
config.to_flat_dict()
```

This is used to keep the legacy flat hparams/checkpoint keys available.

`from_dict()` accepts both nested and flat dictionaries. This matters for:

- Existing checkpoints.
- HuggingFace `config.json` loading.
- Existing tests and scripts that serialize flat config dictionaries.

## Hparams Behavior

The POC writes both flat and structured hparams.

For example, ACT hparams now include flat keys such as:

```yaml
chunk_size: 100
n_action_steps: 100
optimizer_lr: 1.0e-05
```

and structured config:

```yaml
act_config:
  io:
    chunk_size: 100
    n_action_steps: 100
  optimizer:
    optimizer_lr: 1.0e-05
```

Pi05 and SmolVLA use `pi05_config` and `smolvla_config` respectively.

The generic `config` hparams key is no longer written for these POC policies. The constructor still accepts `config` as
a compatibility alias, but new hparams use policy-specific names.

## Field Metadata And Validation

Grouped config fields now include dataclass `metadata` with descriptions and lightweight validation hints, for example:

```python
optimizer_lr: float = field(
    default=1e-4,
    metadata={"description": "Learning rate.", "gt": 0.0},
)
```

Validation is implemented manually in `__post_init__()` methods.

Examples:

- `n_action_steps <= chunk_size`
- positive dimensions
- valid dtype choices
- positive learning rates
- valid optimizer betas
- image resolution tuple shape

This is not yet full FastAPI/OpenAPI integration. The metadata is a useful foundation, but FastAPI will not
automatically consume all dataclass metadata as rich OpenAPI constraints without an adapter or Pydantic model layer.

## Tests Added Or Updated

The POC updated tests for:

- Grouped config construction.
- Legacy flat config construction.
- Nested `to_dict()` / `from_dict()` round trips.
- Flat property access on grouped config classes.
- Policy construction from grouped configs.
- Policy-specific structured hparams keys.
- Metadata descriptions on grouped config fields.
- Validation behavior.

Targeted tests run successfully:

```bash
uv run pytest tests/unit/policies/test_act.py tests/unit/policies/test_pi05.py tests/unit/policies/test_smolvla.py
```

Result:

```text
173 passed
```

## Pros

### Clearer Configuration Structure

Grouped config classes make it easier to see which parameters belong together. For example, optimizer settings are no
longer mixed with image preprocessing, architecture internals, and action chunking parameters.

This is a meaningful readability improvement over very large flat dataclasses.

### Better API Shape For Future REST Integration

Nested config maps naturally to a cleaner REST payload:

```json
{
  "policy": "smolvla",
  "policy_config": {
    "io": {
      "chunk_size": 50,
      "n_action_steps": 50
    },
    "optimizer": {
      "optimizer_lr": 0.0001
    }
  }
}
```

This is easier for users and UI code to reason about than one giant flat object.

### Backward Compatibility Is Mostly Preserved

Existing flat constructor calls still work.

Existing code that reads `config.chunk_size` or `config.optimizer_lr` still works through compatibility properties.

Flat hparams are still written, reducing the risk of breaking checkpoint loading or existing analysis code.

### Policy-Specific Hparams Are Less Ambiguous

Structured hparams now use policy-specific keys:

- `act_config`
- `pi05_config`
- `smolvla_config`

This is clearer than a generic `config` key, especially when inspecting model artifacts manually.

### Better Documentation Surface

Each grouped config field now has a description in metadata. This gives us something structured to use later for REST
schema generation or a policy-parameter discovery endpoint.

### Incremental Migration Path

This approach does not require changing every policy call site immediately. The backend can keep using flat args while
we gradually move toward config-object construction.

## Cons

### More Code

This POC adds a lot of code.

The compatibility layer is especially verbose:

- grouped dataclasses
- flat constructor aliases
- flat properties
- `to_flat_dict()`
- custom `from_dict()` filtering

The result is clearer conceptually, but mechanically larger.

### Duplication Exists During Migration

For now, the same parameter exists in multiple forms:

```python
config.io.chunk_size
config.chunk_size
policy.hparams["chunk_size"]
policy.hparams["act_config"]["io"]["chunk_size"]
```

This duplication is intentional for compatibility, but it can be confusing and creates room for inconsistency if the
code is not disciplined.

### Constructors Are Still Too Large

The POC did not remove the large policy constructors. It only added a better config-object path while preserving the
old flat constructor path.

So this is not the final clean design. The final design would likely prefer:

```python
ACT(act_config=ACTConfig(...))
Pi05(pi05_config=Pi05Config(...))
SmolVLA(smolvla_config=SmolVLAConfig(...))
```

and eventually de-emphasize or remove flat args where compatibility allows.

### Dataclass Metadata Is Not Enough For FastAPI By Itself

The added metadata is useful, but FastAPI will not automatically turn all of it into ideal OpenAPI output unless we
bridge it into Pydantic fields or custom schema generation.

This POC prepares for API integration, but does not complete it.

### Nested Dataclasses May Complicate Checkpoint Evolution

Lightning checkpoint loading is sensitive to constructor signatures and hparams. The POC keeps flat keys to reduce risk,
but long-term migration needs careful checkpoint compatibility tests with real old artifacts.

Unit tests cover synthetic checkpoint paths, but they are not the same as testing a corpus of existing trained models.

### Public API Surface Is Larger

Adding many subconfig classes creates more public names that users may depend on. If we later decide the group boundaries
are wrong, changing them will be harder.

The group names chosen here are reasonable, but not necessarily final.

### Inheritance/Dataclass Custom Init Is Awkward

The top-level config classes use `@dataclass(frozen=True, init=False)` with custom constructors. This is practical for
compatibility, but it is more complex than plain dataclasses and less elegant than a clean Pydantic model hierarchy.

## Open Questions

### Should We Keep Flat Hparams Forever?

Probably not. They are useful for compatibility now, but long term it would be cleaner to store only structured config
plus any runtime training metadata.

Before removing flat hparams, we need to know whether anything downstream depends on them.

### Should Config Classes Become Pydantic Models?

Possibly. Pydantic would make FastAPI integration much cleaner, especially for OpenAPI descriptions and validation.

However, the library currently uses dataclasses and custom `Config` serialization. Switching to Pydantic would be a
larger architectural decision.

### Should Runtime/Checkpoint-Only Fields Be In The API Config?

Some config fields are probably not user-facing REST API fields.

Examples:

- dataset-derived feature definitions
- pretrained model internals
- checkpoint restoration helpers
- some architecture internals that should not be casually changed

The backend API may still need an allowlist or adapter, even with grouped configs.

### Should `compile_model` Stay In Policy Config?

The backend already has `compile_model` on `TrainJobPayload`. The config classes also include `compile_model` because
the policies historically had it in constructors/config.

This can create duplicate state. The backend should decide which one wins and avoid exposing two controls to users.

## Recommendation

The POC is directionally good, but should be treated as a migration step, not the final design.

Recommended next steps:

1. Keep this grouped config shape as the preferred library API.
2. Add backend support for accepting policy config objects, but use a small adapter/allowlist for user-editable fields.
3. Decide whether FastAPI should consume dataclasses directly or whether we should generate/define Pydantic models from these configs.
4. Test loading existing real checkpoints for ACT, Pi05, and SmolVLA.
5. Decide whether to keep flat hparams long term or plan a deprecation path.
6. Repeat the pattern for Pi0 only after deciding whether this grouping strategy is acceptable.

The biggest benefit is conceptual clarity and API readiness. The biggest cost is added compatibility complexity.
