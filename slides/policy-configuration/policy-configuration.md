---
slides:
  title: Policy Configuration Proposal
  theme: ./policy-configuration/theme.css 
---


# Policy Configuration Proposal

Quick reference for the proof of concept to make policy hyperparameters easier to configure, document, and expose through the backend.

Scope: all policy models. Examples use ACT as the running example.

---

## Problem Summary

Policy configuration is spread across multiple places:

- A large `ACT.__init__` constructor accepts every user-facing hyperparameter.
- `ACT.__init__` manually copies those values into `ACTConfig`.
- `ACTConfig` owns defaults and validation, but was not the main public construction path.
- Backend/API code had no structured source for hyperparameter descriptions.

ACT is shown here because it is a compact example of the broader model configuration issue.

---

## Current Setup: Big Constructor

`library/src/physicalai/policies/act/policy.py`

```python
def __init__(  # noqa: PLR0913
    self,
    n_obs_steps: int = 1,
    chunk_size: int = 100,
    n_action_steps: int = 100,
    *,
    vision_backbone: str = "resnet18",
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1",
    replace_final_stride_with_dilation: bool = False,
    image_size: tuple[int, int] = (384, 384),
    pre_norm: bool = False,
    dim_model: int = 512,
    n_heads: int = 8,
    dim_feedforward: int = 3200,
    feedforward_activation: str = "relu",
    n_encoder_layers: int = 4,
    n_decoder_layers: int = 1,
    use_vae: bool = True,
    latent_dim: int = 32,
    n_vae_encoder_layers: int = 4,
    temporal_ensemble_coeff: float | None = None,
    dropout: float = 0.1,
    kl_weight: float = 10.0,
    optimizer_lr: float = 1e-5,
    optimizer_weight_decay: float = 1e-4,
    optimizer_grad_clip_norm: float = 10.0,
    compile_model: bool = False,
    dataset_stats: dict[str, Any] | None = None,
) -> None:
```

---

## Current Setup: Manual Copying

`ACT.__init__` immediately recreates the same shape as `ACTConfig`:

```python
self.config = ACTConfig(
    input_features={},
    output_features={},
    n_obs_steps=n_obs_steps,
    chunk_size=chunk_size,
    n_action_steps=n_action_steps,
    vision_backbone=vision_backbone,
    pretrained_backbone_weights=pretrained_backbone_weights,
    replace_final_stride_with_dilation=replace_final_stride_with_dilation,
    image_size=image_size,
    pre_norm=pre_norm,
    dim_model=dim_model,
    n_heads=n_heads,
    dim_feedforward=dim_feedforward,
    feedforward_activation=feedforward_activation,
    n_encoder_layers=n_encoder_layers,
    n_decoder_layers=n_decoder_layers,
    use_vae=use_vae,
    latent_dim=latent_dim,
    n_vae_encoder_layers=n_vae_encoder_layers,
    temporal_ensemble_coeff=temporal_ensemble_coeff,
    dropout=dropout,
    kl_weight=kl_weight,
    optimizer_lr=optimizer_lr,
    optimizer_weight_decay=optimizer_weight_decay,
    optimizer_grad_clip_norm=optimizer_grad_clip_norm,
    compile_model=compile_model,
)
```

---

## Current Setup: Consequences

- Defaults exist in both the constructor and config class.
- Hyperparameter docs live in long docstrings or external docs.
- API/UI schemas need separate backend knowledge.
- Adding a hyperparameter means touching multiple surfaces.
- Constructor compatibility makes cleanup harder over time.

---

## Proposal: Config As The Entry Point

Prefer each model's config class as the canonical contract for that model's hyperparameters.

ACT example:

```python
config = ACTConfig(
    chunk_size=50,
    n_action_steps=50,
    optimizer_lr=1e-5,
)

policy = ACT(config=config)
```

This aligns policies with the repo's config-based construction helpers, such as `FromConfig.from_config(...)`, and keeps serialization/deserialization simple.

---

## What This Changes

Each model config becomes responsible for:

- Defaults.
- Types.
- Validation.
- Field-level documentation.
- API-readable metadata.

Each policy class becomes responsible for:

- Owning the Lightning policy lifecycle.
- Building the model.
- Training, inference, checkpointing, and export behavior.

---

## Document Hyperparameters With `field`

The POC adds dataclass `field(metadata=...)` to policy config fields.

ACT example:

```python
chunk_size: int = field(
    default=100,
    metadata=_metadata(
        "Number of future action steps predicted per policy invocation.",
        group="io",
        group_title="Input / Output",
        ge=1,
    ),
)
```

The documentation is now colocated with the default value and type.

---

## Metadata Helper

The config code stays independent from Pydantic/FastAPI.

```python
def _metadata(
    description: str,
    *,
    group: str,
    group_title: str,
    title: str | None = None,
    **validation: object,
) -> dict[str, object]:
    json_schema_extra = {"group": group, "group_title": group_title}
    metadata = {
        "description": description,
        "json_schema_extra": json_schema_extra,
        **validation,
    }
    if title is not None:
        metadata["title"] = title
    return metadata
```

---

## Metadata Contract

Use Pydantic-compatible keys where possible:

- `description`: field help text.
- `title`: optional human-readable label.
- `gt`, `ge`, `lt`, `le`: numeric constraints.
- `min_length`, `max_length`: collection/string constraints.
- `json_schema_extra.group`: stable API/UI group key.
- `json_schema_extra.group_title`: display label for the group.

---

## Config Groups

Metadata groups fields into API/UI sections.

ACT example groups:

- `io`: observation and action chunking settings.
- `vision`: backbone and image preprocessing settings.
- `transformer`: transformer architecture settings.
- `vae`: variational objective settings.
- `inference`: inference-time behavior.
- `training`: general training/loss settings.
- `optimizer`: optimizer and gradient clipping settings.

---

## Backend Integration

New endpoint:

```http
GET /api/policies/{policy}/hyper_parameters
```

ACT example:

```http
GET /api/policies/act/hyper_parameters
```

The backend reads policy config dataclass fields instead of maintaining separate hand-written schemas.

---

## Backend Config Registry

`application/backend/src/api/policies.py`

```python
_POLICY_CONFIG_CLASSES = {
    "act": ACTConfig,
    "pi0": Pi0Config,
    "pi05": Pi05Config,
    "smolvla": SmolVLAConfig,
}

@router.get("/{policy}/hyper_parameters")
def get_policy_hyper_parameters(policy: str) -> PolicyHyperParametersResponse:
    config_cls = _POLICY_CONFIG_CLASSES.get(policy)
    if config_cls is None:
        raise HTTPException(status_code=404)

    return PolicyHyperParametersResponse(
        policy=policy,
        hyper_parameters=_hyper_parameters_from_config(config_cls()),
    )
```

---

## Field Extraction

The backend converts dataclass fields into API descriptors.

```python
for config_field in dataclasses.fields(config):
    value = getattr(config, config_field.name)
    human_name = str(
        config_field.metadata.get("title")
        or _humanize_field_name(config_field.name)
    )
    description = str(config_field.metadata.get("description", ""))
    group_name, group_title = _group_info(config_field.metadata)
```

This keeps the backend generic while preserving model-specific docs in model config code.

---

## Response Shape

Simplified response for `chunk_size`:

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

---

## Response Types

The API uses discriminated hyperparameter types:

```python
PolicyHyperParameter = Annotated[
    GroupHyperParameter
    | IntHyperParameter
    | BooleanHyperParameter
    | FloatHyperParameter
    | StringHyperParameter
    | ChoiceHyperParameter,
    Field(discriminator="field_type"),
]
```

This gives frontend/API clients a stable shape for rendering controls.

---

## Why Not Use `hparams.yaml` As The Contract?

`hparams.yaml` is training output, not an input schema.

It can contain:

- Constructor values captured by Lightning.
- A nested `config` dict.
- Runtime values such as `dataset_stats`.

The API contract should come from model config classes, not from logger output.

---

## Optional: Smaller Config Classes

An earlier POC commit explored splitting ACT config into focused classes. The same idea can apply to other large model configs.

```python
ACTInputOutputConfig
ACTVisionBackboneConfig
ACTTransformerConfig
ACTVAEConfig
ACTInferenceConfig
ACTOptimizerConfig
```

Then `ACTConfig` grouped those sections:

```python
ACTConfig(
    io=ACTInputOutputConfig(...),
    vision=ACTVisionBackboneConfig(...),
    optimizer=ACTOptimizerConfig(...),
)
```

---

## Optional Split: Tradeoff

Benefits:

- Smaller classes.
- Cleaner validation boundaries.
- More explicit structure in Python.

Costs:

- Changes serialized config shape.
- Requires compatibility helpers for existing flat checkpoints/configs.
- Adds migration complexity before the backend has full config payload support.

Recommendation: keep flat config classes with metadata first; consider nested classes later as a breaking cleanup.

---

## Remove Old Constructor Arguments

Long-term target:

```python
policy = ACT(
    config=ACTConfig(
        chunk_size=50,
        n_action_steps=50,
        optimizer_lr=1e-5,
    ),
)
```

Instead of:

```python
policy = ACT(
    chunk_size=50,
    n_action_steps=50,
    optimizer_lr=1e-5,
)
```

---

## Why Remove Compatibility Later?

Once callers and checkpoints are migrated, removing legacy constructor kwargs gives us:

- One source of defaults.
- One validation path.
- Smaller `ACT.__init__`.
- Less duplicated documentation.
- Cleaner backend policy construction.

Do not keep compatibility code indefinitely unless there is a concrete external compatibility requirement.

---

## Suggested Migration Path

1. Add `field(metadata=...)` to policy config classes.
2. Expose `/api/policies/{policy}/hyper_parameters` from config metadata.
3. Add backend training payload support for policy config values.
4. Prefer config-first policy construction in new code.
5. Optionally split large config classes into smaller classes.
6. Remove old flat constructor arguments when migration risk is acceptable.

---

## Team Takeaway

The proposal makes each model's config class the source of truth.

- Library config owns defaults, types, validation, and docs.
- Backend reads config metadata to expose hyperparameters.
- UI/API can render grouped controls without duplicated schemas.
- Policy constructors can eventually shrink to config-first APIs.
