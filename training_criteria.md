# Training Criteria: Dataset Requirements for Policy Training

This document describes the dataset requirements for training a policy in geti-action,
with particular emphasis on the **statistical metadata** that LeRobot computes and how
it flows through the training pipeline.

---

## 1. High-Level Training Flow

```
Dataset on disk (LeRobot v2.1 format)
  -> SnapshotService copies the entire directory tree
  -> LeRobotDataModule wraps the snapshot as a _LeRobotDatasetAdapter
  -> Policy.setup() reads dataset.stats and dataset features
  -> Trainer.fit() trains the policy
```

Key entry point: `application/backend/src/workers/training_worker.py`

The `TrainingWorker._train_model()` method (line 93) creates a `LeRobotDataModule`
pointing at the snapshot path, then calls `trainer.fit(model=policy, datamodule=l_dm)`.
The policy's `setup()` hook (called by Lightning during `fit()`) extracts the dataset
statistics and feature metadata to configure normalization and architecture parameters.

---

## 2. On-Disk Dataset Structure (LeRobot v2.1)

A valid dataset must have this directory layout:

```
<dataset_root>/
├── meta/
│   ├── info.json              # REQUIRED - the only existence check performed
│   ├── episodes.jsonl         # Episode boundaries and metadata
│   ├── episodes_stats.jsonl   # Per-episode statistics (v2.1+)
│   ├── stats.json             # Aggregated statistics (v2.0 compat)
│   └── tasks.jsonl            # Task descriptions
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       ├── episode_000001.parquet
│       └── ...
└── videos/
    └── chunk-000/
        └── observation.images.<camera_name>/
            ├── episode_000000.mp4
            └── ...
```

**Existence check**: The application only validates that `meta/info.json` exists
(`InternalLeRobotDataset._check_repository_exists()` at
`application/backend/src/internal_datasets/lerobot/lerobot_dataset.py:198`).
Everything else is delegated to the upstream `lerobot` library.

### `info.json` Contents

| Field | Type | Description |
|---|---|---|
| `codebase_version` | string | LeRobot format version (e.g., `"v2.1"`) |
| `robot_type` | string | Robot identifier (e.g., `"so100"`, `"aloha"`) |
| `total_episodes` | int | Number of episodes in the dataset |
| `total_frames` | int | Total number of frames across all episodes |
| `total_tasks` | int | Number of unique tasks |
| `total_videos` | int | Number of video files |
| `total_chunks` | int | Number of data chunks |
| `chunks_size` | int | Max episodes per chunk (default: 1000) |
| `fps` | int | Frames per second of the recording |
| `splits` | dict | Train/test splits |
| `data_path` | string | Format string for parquet file paths |
| `video_path` | string | Format string for video file paths |
| `features` | dict | Feature definitions (see below) |

### Feature Definitions (in `info.json`)

Each feature in the `features` dict has:

```json
{
  "observation.state": {
    "dtype": "float32",
    "shape": [6],
    "names": ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
  },
  "observation.images.top": {
    "dtype": "video",
    "shape": [480, 640, 3],
    "names": ["height", "width", "channels"]
  },
  "action": {
    "dtype": "float32",
    "shape": [6],
    "names": ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
  },
  "episode_index": { "dtype": "int64", "shape": [] },
  "frame_index": { "dtype": "int64", "shape": [] },
  "timestamp": { "dtype": "float32", "shape": [] },
  "task_index": { "dtype": "int64", "shape": [] },
  "index": { "dtype": "int64", "shape": [] }
}
```

---

## 3. Required Per-Sample Data

Every sample (frame) returned by the dataset's `__getitem__` must contain:

| Key | Type | Required | Description |
|---|---|---|---|
| `observation.state` | Tensor `(D,)` | Yes | Robot joint positions / proprioceptive state |
| `observation.images.<camera>` | Tensor `(C, H, W)` | Yes (at least one) | Camera frame (decoded from video) |
| `action` | Tensor `(D,)` | Yes | Robot joint commands / action |
| `episode_index` | int tensor | Yes | Which episode this frame belongs to |
| `frame_index` | int tensor | Yes | Frame position within its episode |
| `index` | int tensor | Yes | Global frame index across all episodes |
| `task_index` | int tensor | Yes | Index into the tasks table |
| `timestamp` | float tensor | Yes | Timestamp in seconds |

These are validated in `_convert_lerobot_dict_to_observation()`
(`library/src/getiaction/data/lerobot/converters.py:72-90`).

**Gotcha**: The `observation.images.*` key naming matters. The adapter strips
prefixes differently depending on whether it's `observation.images.<name>` or
`observation.image` (singular). Multi-camera setups must use the `observation.images.`
prefix format.

---

## 4. Statistical Data (The Core Focus)

### 4.1. What Statistics Are Stored

The LeRobot library computes per-feature normalization statistics when a dataset is
finalized. These are **not computed by geti-action code** -- they come entirely from
the upstream `lerobot` library. The stats contain four values per feature:

| Stat | Description |
|---|---|
| `mean` | Element-wise mean across all frames in the dataset |
| `std` | Element-wise standard deviation |
| `min` | Element-wise minimum value |
| `max` | Element-wise maximum value |

For **v2.0** datasets, stats are stored in `meta/stats.json` (single aggregated file).
For **v2.1+** datasets, per-episode stats are stored in `meta/episodes_stats.jsonl`
and aggregated at load time.

### 4.2. Stats Structure in Memory

After loading, `dataset.meta.stats` is a dict of dicts:

```python
{
    "observation.state": {
        "mean": tensor([...]),   # shape: (state_dim,)
        "std":  tensor([...]),   # shape: (state_dim,)
        "min":  tensor([...]),   # shape: (state_dim,)
        "max":  tensor([...]),   # shape: (state_dim,)
    },
    "observation.images.top": {
        "mean": tensor([...]),   # shape: (C, 1, 1) for channel-wise stats
        "std":  tensor([...]),   # shape: (C, 1, 1)
        "min":  tensor([...]),   # shape: (C, 1, 1)
        "max":  tensor([...]),   # shape: (C, 1, 1)
    },
    "action": {
        "mean": tensor([...]),   # shape: (action_dim,)
        "std":  tensor([...]),   # shape: (action_dim,)
        "min":  tensor([...]),   # shape: (action_dim,)
        "max":  tensor([...]),   # shape: (action_dim,)
    }
}
```

### 4.3. How Stats Flow Through the Pipeline

```
LeRobot meta.stats[key]  (tensors)
  ↓
_LeRobotDatasetAdapter.observation_features / action_features
  → converts to NormalizationParameters(mean=.tolist(), std=.tolist(), ...)
  ↓
Dataset.stats property
  → builds canonical dict: { "observation.<name>": { mean, std, min, max, type, name, shape } }
  ↓
Policy.setup()
  → reads train_dataset.stats
  → saves to self.hparams["dataset_stats"]
  → passes to _initialize_model(stats_dict)
  ↓
Pre/Post-processors created with stats
  → Stats registered as nn.Module buffers (requires_grad=False)
  → Used for normalize/denormalize during forward pass
```

The key adapter code is in `library/src/getiaction/data/lerobot/dataset.py:154-215`.

### 4.4. Canonical Stats Dict Format (After Adapter)

The `Dataset.stats` property (`library/src/getiaction/data/dataset.py:63-106`) builds
a dict with this structure:

```python
{
    "observation.state": {
        "mean": [0.1, -0.3, ...],     # list[float]
        "std":  [0.5, 0.8, ...],
        "min":  [-1.0, -2.0, ...],
        "max":  [1.0, 2.0, ...],
        "type": "STATE",              # FeatureType enum value
        "name": "state",              # Feature name (prefix stripped)
        "shape": (6,),                # Feature shape tuple
    },
    "observation.top": {
        "mean": [0.485, 0.456, 0.406],
        "std":  [0.229, 0.224, 0.225],
        "min":  [0.0, 0.0, 0.0],
        "max":  [1.0, 1.0, 1.0],
        "type": "VISUAL",
        "name": "top",
        "shape": (3, 480, 640),       # (C, H, W) -- always channel-first
    },
    "action": {
        "mean": [...],
        "std":  [...],
        "min":  [...],
        "max":  [...],
        "type": "ACTION",
        "name": "action",
        "shape": (6,),
    }
}
```

**Gotcha - Visual feature name prefix stripping**: The adapter strips two prefixes:
1. `"observation."` is always removed
2. `"images."` is additionally removed for visual features

So `observation.images.top` becomes `top` in the stats dict key `"observation.top"`.
This is done at `dataset.py:164,172`.

**Gotcha - Channel order for visual features**: LeRobot v2.0 has a known bug where
some ported datasets store shape as `(H, W, C)` with `names` containing `"channel"`
or `"channels"` at index 2. The adapter detects this and transposes to `(C, H, W)`.
See `dataset.py:169-174`.

### 4.5. Stats Must Be Present

Stats are **not optional**. If `dataset.meta.stats[key]` is missing for any feature,
the adapter will raise a `KeyError`. The stats are consumed by every policy during
`setup()` and are critical for correct normalization.

**Gotcha**: Stats are computed by LeRobot's `dataset.save_episode()` and
`dataset.finalize()` calls. If a dataset is created but never finalized properly,
the stats may be missing or incomplete, causing training to fail.

---

## 5. Per-Policy Normalization Approaches

Different policies use stats differently. This affects which stat fields are actually
required in practice.

### 5.1. ACT and SmolVLA: `FeatureNormalizeTransform`

Uses `library/src/getiaction/policies/utils/normalization.py`.

- **STATE features**: Normalized with `MEAN_STD` (uses `mean` and `std`)
- **VISUAL features**: Normalized with `IDENTITY` (no normalization, stats ignored)
- **ACTION features**: Normalized with `MEAN_STD` (uses `mean` and `std`)

Stats are stored as `nn.Parameter` buffers (non-trainable). Initialized to `inf` and
overwritten with actual stats. During forward pass, an infinity check raises
`ValueError` if buffers weren't properly initialized.

**Gotcha - Visual feature shape validation**: Visual features MUST have exactly 3
dimensions `(C, H, W)` in channel-first order. The normalization code validates
`c < h and c < w` at `normalization.py:200-203`. This will fail on non-standard
image shapes.

### 5.2. Groot: Min-Max Normalization

Uses `library/src/getiaction/policies/groot/transforms.py`.

- **STATE**: Normalized to `[-1, 1]` using `min` and `max`
- **ACTION**: Normalized to `[-1, 1]` using `min` and `max`
- **VISUAL**: Not normalized by the normalization module (processed by Eagle vision encoder)

Only reads `stats["observation.state"]["min"]`, `stats["observation.state"]["max"]`,
`stats["action"]["min"]`, `stats["action"]["max"]`.

Stats are padded/truncated to `max_state_dim=64` and `max_action_dim=32`.

### 5.3. Pi0: Mean/Std or Quantile Normalization

Uses `library/src/getiaction/policies/pi0/preprocessor.py`.

- **STATE**: Normalized with `mean`/`std` (or `q01`/`q99` if `use_quantile_norm=True`)
- **ACTION**: Normalized with `mean`/`std` (or `q01`/`q99`)
- **VISUAL**: Not normalized by the normalization module (scaled to `[-1, 1]` directly)

Has `NormStats` dataclass with `mean`, `std`, `q01`, `q99` fields. The `q01`/`q99`
quantile fields are **not provided by LeRobot's default stats** -- they would need
to be computed separately if quantile normalization is desired.

State is padded to `max_state_dim=32`, action to `max_action_dim=32`, action
sequence padded to `chunk_size=50`.

---

## 6. Episode Structure Requirements

### 6.1. `episode_data_index`

The `episode_data_index` attribute on a LeRobot dataset provides:

```python
{
    "from": LongTensor([0, 50, 120, ...]),    # Start frame index per episode
    "to":   LongTensor([50, 120, 200, ...]),  # End frame index per episode (exclusive)
}
```

This is computed from cumulative episode lengths and is used for:
- Clamping `delta_indices` to stay within episode boundaries
- Preventing temporal windows from crossing episode boundaries
- Generating `*_is_pad` masks for out-of-bounds frames

### 6.2. Delta Timestamps / Temporal Windowing

Policies that operate on sequences (not just single frames) use `delta_timestamps` to
query multiple timesteps from the dataset. Example for ACT at 10fps:

```python
{
    "action": [0.0, 0.1, 0.2, ..., 0.9],           # 10 future action steps
    "observation.state": [0.0],                       # Current state only
    "observation.images.top": [0.0],                  # Current image only
}
```

These are converted to integer frame offsets (`delta_indices`) and used in the
dataset's `__getitem__` to return multi-timestep data. Frames that would fall outside
the current episode boundary are padded and marked with `*_is_pad = True`.

**Gotcha**: `delta_timestamps` must be consistent with the dataset's `fps`. The
`check_delta_timestamps()` function validates that each timestamp is an integer
multiple of `1/fps` within `tolerance_s` (default `1e-4`).

The `reformat_dataset_to_match_policy()` utility
(`library/src/getiaction/train/utils.py:53`) automatically computes delta_timestamps
from the policy's config and sets them on the dataset.

---

## 7. Feature Type Classification

The adapter classifies features into types based on metadata:

| Feature Key Pattern | `dtype` | Classified As |
|---|---|---|
| `observation.state` | `float32` | `FeatureType.STATE` |
| `observation.environment_state` | `float32` | `FeatureType.ENV` |
| `observation.images.*` | `image` or `video` | `FeatureType.VISUAL` |
| `action` | `float32` | `FeatureType.ACTION` |

The `dtype` field in the feature definition determines visual vs. state:
- `"image"` or `"video"` -> `VISUAL`
- Everything else starting with `observation.` -> `STATE`
- `observation.environment_state` is special-cased as `ENV`

---

## 8. Summary Checklist for a Valid Training Dataset

- [ ] `meta/info.json` exists and contains valid metadata
- [ ] `meta/episodes.jsonl` lists all episodes with boundaries
- [ ] `meta/tasks.jsonl` contains at least one task
- [ ] `meta/stats.json` or `meta/episodes_stats.jsonl` contains normalization
      statistics for ALL features (observation.state, observation.images.*, action)
- [ ] Each stat entry has all four fields: `mean`, `std`, `min`, `max`
- [ ] Parquet files exist in `data/chunk-NNN/` with all non-video features as columns
- [ ] Video files exist in `videos/chunk-NNN/observation.images.<camera>/` for each camera
- [ ] At least one `observation.state` feature with joint positions
- [ ] At least one `observation.images.*` feature (video/image)
- [ ] Exactly one `action` feature matching the robot's joint dimensions
- [ ] `fps` is set correctly and consistently with temporal data
- [ ] Video frame count matches the parquet row count per episode
- [ ] Feature shapes in `info.json` match actual data dimensions
- [ ] Visual feature shapes are `(H, W, C)` in metadata (adapter handles transposition to `(C, H, W)`)

---

## 9. Gotchas and Edge Cases

1. **No stats = training crash**. Stats are required by all policies. If the dataset
   was not properly finalized (via `dataset.finalize()`), stats will be missing and
   training will fail with a `KeyError` when accessing `dataset_meta.stats[k]`.

2. **Channel order confusion**. LeRobot stores visual shapes as `(H, W, C)` in
   metadata but the adapter converts to `(C, H, W)` for PyTorch. Some v2.0 ported
   datasets have the `names` field wrong (`"channel"` vs `"channels"` at index 2),
   which the adapter handles, but it's a source of bugs.

3. **Quantile stats (`q01`/`q99`) not provided by LeRobot**. Pi0 supports quantile
   normalization, but LeRobot's default stats only include `mean`/`std`/`min`/`max`.
   If `use_quantile_norm=True` is set but quantiles aren't available, Pi0 falls back
   to `mean`/`std` normalization silently.

4. **Stats shape for images**. Image stats are channel-wise: shape `(C, 1, 1)` not
   `(C, H, W)`. The normalization code reshapes to `(channels, 1, 1)` for
   broadcasting. If stats have the wrong shape, normalization will fail.

5. **`tasks.jsonl` vs `tasks.parquet`**. Older LeRobot versions used parquet for
   tasks; current v2.1 uses JSONL. A version mismatch causes
   `FileNotFoundError: No such file or directory: .../meta/tasks.parquet`.

6. **Snapshot is a full directory copy**. The `SnapshotService` does a
   `shutil.copytree()` of the entire dataset directory. For large datasets with
   video files, this can be very slow and consume significant disk space.

7. **Infinity sentinel in normalization buffers**. The `FeatureNormalizeTransform`
   initializes stat buffers to `inf`. If stats aren't properly loaded (e.g., missing
   features), the forward pass will raise `ValueError("Normalization buffer '...' is
   infinity")`.

8. **Max dimension padding**. Groot pads state to 64 dims and action to 32 dims.
   Pi0 and SmolVLA pad state and action to 32 dims each. If the actual robot has
   more joints than the max dimension, values are **silently truncated**.

9. **`delta_timestamps` must align with `fps`**. The `check_delta_timestamps()`
   function validates that `timestamp * fps` is close to an integer (within
   `tolerance_s=1e-4`). Mismatched fps will cause a validation error.

10. **Dataset `repo_id` is ignored for local datasets**. When training, the
    `LeRobotDataModule` is created with `repo_id="snapshot"` (line 105 of
    training_worker.py). The actual repo_id doesn't matter for local-path loading,
    but it must still be provided as a non-empty string.

11. **No data validation at training time**. The training worker does not validate
    the dataset contents before starting training. If the dataset is malformed
    (e.g., missing features, corrupted videos), the error will surface during
    the training loop, potentially after significant computation.
