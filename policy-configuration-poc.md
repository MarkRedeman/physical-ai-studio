# Policy Configuration POC

## Summary

This POC keeps the existing flat policy config classes for `ACT`, `Pi05`, and `SmolVLA`, but adds structured field
metadata to each relevant config field. The metadata lets the backend expose grouped hyperparameter information without
changing the config object shape or policy constructor behavior.

This is the safer compatibility-focused approach. The library still has flat config classes and flat constructor args,
while the API can present those fields as grouped UI/API sections.

## Before

The config shape was flat and undocumented at the field level:

```json
{
  "chunk_size": 100,
  "n_action_steps": 50,
  "vision_backbone": "resnet18",
  "dim_model": 512,
  "n_heads": 8,
  "use_vae": true,
  "optimizer_lr": 0.00001
}
```

Construction looked like this and still does:

```python
ACT(chunk_size=100, n_action_steps=50, optimizer_lr=1e-5)
```

The downside was that action chunking, vision, architecture, VAE, and optimizer settings were all mixed in one namespace
with no structured descriptions for API/UI generation.

## After

The config class remains flat, but fields now carry metadata:

```python
chunk_size: int = field(
    default=100,
    metadata={
        "description": "Number of future action steps predicted per policy invocation.",
        "title": "Chunk Size",
        "ge": 1,
        "json_schema_extra": {
            "group": "io",
            "group_title": "Input / Output"
        }
    },
)
```

The API endpoint uses that metadata to return grouped hyperparameters:

```json
{
  "policy": "act",
  "hyper_parameters": [
    {
      "name": "io",
      "field_type": "group",
      "human_name": "Input / Output",
      "description": "Input / Output hyperparameters",
      "hyper_parameters": [
        {
          "name": "chunk_size",
          "field_type": "integer",
          "default_value": 100,
          "description": "Number of future action steps predicted per policy invocation.",
          "human_name": "Chunk Size"
        }
      ]
    }
  ]
}
```

Choice fields are represented explicitly:

```json
{
  "name": "dtype",
  "field_type": "choice",
  "default_value": "bfloat16",
  "allowed_values": ["bfloat16", "float32"],
  "description": "Model weight precision.",
  "human_name": "Dtype"
}
```

## What Was Implemented

Implemented for:

- `ACTConfig`
- `Pi05Config`
- `SmolVLAConfig`

Added:

- Pydantic/FastAPI-compatible dataclass field metadata.
- `json_schema_extra.group` and `json_schema_extra.group_title` for API grouping.
- descriptions and human-friendly titles where needed.
- the `/api/policies/{policy}/hyper_parameters` endpoint.
- discriminated response types: `group`, `integer`, `boolean`, `float`, `string`, and `choice`.
- tests for metadata compatibility and endpoint responses.

The policy constructors and config classes remain flat for backward compatibility.

## Compatibility

This approach preserves the existing API:

```python
ACTConfig(chunk_size=100, optimizer_lr=1e-5)
Pi05Config(chunk_size=100, optimizer_lr=2.5e-5)
SmolVLAConfig(chunk_size=100, optimizer_lr=1e-4)
```

It also preserves hparams behavior:

```yaml
chunk_size: 100
optimizer_lr: 1.0e-05
config:
  chunk_size: 100
  optimizer_lr: 1.0e-05
```

This avoids the checkpoint compatibility risk introduced by changing the config shape to nested dataclasses.

## FastAPI Compatibility

Metadata uses Pydantic-compatible keys such as:

- `description`
- `title`
- `gt`, `ge`, `lt`, `le`
- `min_length`, `max_length`
- `json_schema_extra`

That means a backend adapter can translate fields with:

```python
Field(default=dataclass_field.default, **dataclass_field.metadata)
```

The config modules themselves still do not import Pydantic.

## Pros

### Strong Backward Compatibility

The config classes remain flat. Existing constructor calls, config serialization, hparams, and checkpoint assumptions are
much less likely to break.

### Better API Shape Without Changing Library Shape

The API still returns grouped hyperparameters, but grouping is derived from metadata rather than nested config classes.

### FastAPI-Friendly Metadata

Field metadata can be reused by a Pydantic/FastAPI adapter with minimal translation.

### Less Compatibility Glue

We no longer need nested config classes, flat compatibility properties, or `to_flat_dict()` migration helpers.

## Cons

### Config Classes Are Still Flat

The Python config classes remain large and less elegant than truly nested configs.

### Grouping Is Metadata-Driven

Grouping is not represented in the Python type structure. It depends on `json_schema_extra.group` being present and
correct.

### Still Needs Backend Training Payload Integration

The endpoint exposes available hyperparameters, but training jobs do not yet accept and apply user-provided policy config
payloads.

### Real Checkpoint Testing Still Needed

This approach is designed to preserve compatibility, but existing real checkpoints should still be tested.

## Recommendation

Prefer this flat-config-with-metadata approach for now. It gives us the API/UI benefits while minimizing checkpoint and
constructor compatibility risk.

Next steps:

1. Test loading existing real checkpoints for `ACT`, `Pi05`, and `SmolVLA`.
2. Add backend training payload support using these flat config classes.
3. Use an allowlist or metadata flag to decide which fields are actually user-editable.
4. Consider nested configs later only if we are ready for a breaking library API migration.
