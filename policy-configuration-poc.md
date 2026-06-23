# Policy Configuration POC

## Summary

This POC restructures policy configuration for `ACT`, `Pi05`, and `SmolVLA` from large flat config objects into grouped
dataclasses. The old flat constructor style still works for compatibility, but the preferred path is now a policy-specific
config object such as `act_config`, `pi05_config`, or `smolvla_config`.

This is a library-side POC. The config metadata is now compatible with `pydantic.Field(**metadata)`, which makes it easy
to translate these dataclasses into FastAPI/OpenAPI request schemas. The backend API wiring itself is still not
implemented.

## Before

The old shape was flat. Conceptually, a config looked like this:

```json
{
  "chunk_size": 100,
  "n_action_steps": 50,
  "vision_backbone": "resnet18",
  "pretrained_backbone_weights": "ResNet18_Weights.IMAGENET1K_V1",
  "dim_model": 512,
  "n_heads": 8,
  "use_vae": true,
  "latent_dim": 32,
  "kl_weight": 10.0,
  "optimizer_lr": 0.00001,
  "optimizer_weight_decay": 0.0001,
  "optimizer_grad_clip_norm": 10.0
}
```

And policy construction typically looked like this:

```python
ACT(chunk_size=100, n_action_steps=50, optimizer_lr=1e-5)
```

This works, but related concerns are mixed together: action chunking, vision backbone, transformer architecture, VAE
settings, and optimizer settings all live in one namespace.

## After

The new preferred shape groups related settings:

```json
{
  "io": {
    "chunk_size": 100,
    "n_action_steps": 50
  },
  "vision": {
    "vision_backbone": "resnet18",
    "pretrained_backbone_weights": "ResNet18_Weights.IMAGENET1K_V1"
  },
  "transformer": {
    "dim_model": 512,
    "n_heads": 8
  },
  "vae": {
    "use_vae": true,
    "latent_dim": 32,
    "kl_weight": 10.0
  },
  "optimizer": {
    "optimizer_lr": 0.00001,
    "optimizer_weight_decay": 0.0001,
    "optimizer_grad_clip_norm": 10.0
  }
}
```

Construction can now use policy-specific config objects:

```python
ACT(act_config=ACTConfig(io=ACTInputOutputConfig(chunk_size=100, n_action_steps=50)))
Pi05(pi05_config=Pi05Config(...))
SmolVLA(smolvla_config=SmolVLAConfig(...))
```

For compatibility, these still work:

```python
ACT(config=ACTConfig(...))
ACT(chunk_size=100, optimizer_lr=1e-5)
```

The same compatibility pattern exists for `Pi05` and `SmolVLA`.

## Hparams

New structured hparams use policy-specific keys:

```json
{
  "chunk_size": 100,
  "n_action_steps": 50,
  "optimizer_lr": 0.00001,
  "act_config": {
    "io": {
      "chunk_size": 100,
      "n_action_steps": 50
    },
    "optimizer": {
      "optimizer_lr": 0.00001
    }
  }
}
```

For `Pi05` and `SmolVLA`, the structured keys are `pi05_config` and `smolvla_config`.

Flat hparams are intentionally still written for now. This reduces the risk of breaking checkpoint loading or existing
analysis tools that expect flat keys.

## What Was Implemented

Implemented grouped config support for:

- `ACT`
- `Pi05`
- `SmolVLA`

Each policy now has:

- grouped subconfig dataclasses
- field metadata compatible with `pydantic.Field(**metadata)`
- basic validation in `__post_init__`
- support for grouped config construction
- support for legacy flat config construction
- policy-specific hparams keys
- tests for grouped config behavior and compatibility

Targeted tests passed:

```bash
uv run pytest tests/unit/policies/test_act.py tests/unit/policies/test_pi05.py tests/unit/policies/test_smolvla.py
```

Result: `176 passed`.

## FastAPI Compatibility

Config fields use dataclass `metadata` with Pydantic/FastAPI-compatible keys:

```python
optimizer_lr: float = field(
    default=1e-4,
    metadata={
        "description": "Learning rate.",
        "gt": 0.0,
    },
)
```

For enum-like schema hints, fields use `json_schema_extra`:

```python
dtype: Literal["bfloat16", "float32"] = field(
    default="bfloat16",
    metadata={
        "description": "Model weight precision.",
        "json_schema_extra": {"enum": ["bfloat16", "float32"]},
    },
)
```

For tuple/list length constraints, fields use Pydantic's `min_length` and `max_length` names:

```python
image_size: tuple[int, int] = field(
    default=(512, 512),
    metadata={
        "description": "Image preprocessing resolution as (height, width).",
        "min_length": 2,
        "max_length": 2,
    },
)
```

That means a backend adapter can mechanically translate dataclass fields into Pydantic fields without renaming common
constraint keys:

```python
Field(default=dataclass_field.default, **dataclass_field.metadata)
```

The config modules themselves still do not import Pydantic.

## Pros

### Clearer Mental Model

Grouping makes the config easier to scan. Optimizer settings, model architecture, preprocessing, and action chunking are
no longer all mixed in one flat namespace.

### Better API Shape

Nested configs map more naturally to a REST payload and UI form. It is easier to show sections like `optimizer`, `io`,
or `vision` than a long unsorted list of fields.

### Backward Compatibility

The old flat constructor style still works. Existing code that reads `config.chunk_size` or `config.optimizer_lr` still
works through compatibility properties.

### Better Hparams Naming

Structured hparams now use policy-specific names: `act_config`, `pi05_config`, and `smolvla_config`. This is clearer than
a generic `config` key.

### Metadata Foundation

Fields now have structured descriptions and Pydantic-compatible validation/schema hints. A backend adapter can translate
the dataclass metadata into FastAPI/OpenAPI fields without bespoke key mapping.

## Cons

### More Code And More Surface Area

The compatibility layer adds a lot of code: nested dataclasses, flat properties, `to_flat_dict()`, custom `from_dict()`,
and policy-specific aliases.

### Duplication During Migration

The same value can now exist in several places:

```python
config.io.chunk_size
config.chunk_size
policy.hparams["chunk_size"]
policy.hparams["act_config"]["io"]["chunk_size"]
```

This is intentional for compatibility, but it is not ideal long term.

### Constructors Are Still Large

The POC did not remove the large policy constructors. It added a cleaner config-object path while keeping the old path.
So this is a migration step, not the final design.

### FastAPI Wiring Is Not Done Yet

The metadata is compatible with Pydantic field kwargs, but FastAPI will not automatically expose these library dataclasses
as the backend training request schema. We still need an adapter, generated Pydantic models, or explicit request models.

### Checkpoint Compatibility Still Needs Real-Artifact Testing

Unit tests pass, but real old checkpoints should still be tested. Lightning checkpoint loading is sensitive to hparams
and constructor signatures.

## Recommendation

Keep this as the preferred library-side direction, but treat it as a POC rather than a final API design.

Next steps:

1. Test loading existing real checkpoints for `ACT`, `Pi05`, and `SmolVLA`.
2. Decide whether backend API schemas should use these dataclasses directly or generate Pydantic models from their metadata.
3. Add a backend adapter/allowlist so only intended user-editable fields are exposed.
4. Decide whether flat hparams should be deprecated later.
5. Apply the pattern to `Pi0` only after we are comfortable with the grouping and API story.
