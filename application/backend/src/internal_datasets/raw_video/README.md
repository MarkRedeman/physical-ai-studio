# Raw Video Dataset Adapter

## Overview

The raw video dataset adapter enables training robot policies directly from raw video
files and JSONL telemetry logs. It eliminates the need to first convert data into the
LeRobot HuggingFace dataset format, reducing preprocessing friction when working with
newly collected demonstrations.

Use this adapter when you have:

- One or more MP4 video files per episode (one per camera).
- A JSONL file per episode containing timestamped state and action vectors.
- A `manifest.json` file describing the dataset layout.

The adapter implements the `getiaction.data.Dataset` abstract base class, so it plugs
directly into the existing training pipeline.

## Dataset Layout

The adapter expects the following on-disk structure:

```
raw_dataset/
├── manifest.json
├── episode_000/
│   ├── data.jsonl
│   ├── cam_top.mp4
│   └── cam_gripper.mp4
├── episode_001/
│   ├── data.jsonl
│   ├── cam_top.mp4
│   └── cam_gripper.mp4
└── .cache/
    └── stats.json          # auto-generated on first load
```

- `manifest.json` -- Top-level schema describing the dataset (see below).
- `episode_NNN/` -- One directory per episode, containing a JSONL data file and one
  video file per camera.
- `.cache/stats.json` -- Normalization statistics, automatically computed on first load
  and cached for subsequent runs.

## manifest.json Schema

| Field              | Type                  | Required | Default | Description                                                                 |
|--------------------|-----------------------|----------|---------|-----------------------------------------------------------------------------|
| `name`             | `string`              | yes      | --      | Human-readable dataset name.                                                |
| `fps`              | `int` (> 0)           | yes      | --      | Target frames per second for training.                                      |
| `state_dim`        | `int` (> 0)           | yes      | --      | Dimensionality of the state vector.                                         |
| `action_dim`       | `int` (> 0)           | yes      | --      | Dimensionality of the action vector.                                        |
| `cameras`          | `list[CameraConfig]`  | yes      | --      | Camera configurations. Each entry has a `name` field (valid Python identifier). |
| `episodes`         | `list[EpisodeEntry]`  | yes      | --      | Ordered list of episode entries (see below).                                |
| `task_description` | `string`              | no       | `""`    | Optional human-readable description of the task.                            |

Each **EpisodeEntry** has:

| Field         | Type              | Required | Default        | Description                                                         |
|---------------|-------------------|----------|----------------|---------------------------------------------------------------------|
| `episode_dir` | `string`          | yes      | --             | Relative directory name within the dataset root (e.g. `"episode_000"`). |
| `data_file`   | `string`          | no       | `"data.jsonl"` | Filename of the JSONL data file inside `episode_dir`.               |
| `video_files` | `dict[str, str]`  | yes      | --             | Mapping from camera name to video filename (e.g. `{"top": "cam_top.mp4"}`). |

**Validation rules:**

- Every episode must reference exactly the set of camera names declared in `cameras`.
- On load, every referenced directory and file must exist on disk.

### Example manifest.json

```json
{
  "name": "pick_and_place_v1",
  "fps": 30,
  "state_dim": 7,
  "action_dim": 7,
  "cameras": [
    {"name": "top"},
    {"name": "gripper"}
  ],
  "episodes": [
    {
      "episode_dir": "episode_000",
      "data_file": "data.jsonl",
      "video_files": {
        "top": "cam_top.mp4",
        "gripper": "cam_gripper.mp4"
      }
    },
    {
      "episode_dir": "episode_001",
      "data_file": "data.jsonl",
      "video_files": {
        "top": "cam_top.mp4",
        "gripper": "cam_gripper.mp4"
      }
    }
  ],
  "task_description": "Pick up the red block and place it in the bin"
}
```

## data.jsonl Format

Each line in the JSONL file is a JSON object with three required fields:

```json
{"timestamp": 0.0, "state": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], "action": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]}
{"timestamp": 0.033, "state": [0.11, 0.21, 0.31, 0.41, 0.51, 0.61, 0.71], "action": [0.51, 0.61, 0.71, 0.81, 0.91, 1.01, 1.11]}
```

| Field       | Type            | Description                                                  |
|-------------|-----------------|--------------------------------------------------------------|
| `timestamp` | `float`         | Time in seconds from the start of the episode.               |
| `state`     | `list[float]`   | State vector. Length must equal `state_dim` from the manifest.|
| `action`    | `list[float]`   | Action vector. Length must equal `action_dim` from the manifest.|

Blank lines are skipped. The number of data rows does not need to match the video
frame count -- the adapter applies nearest-neighbor resampling to align them (see
[Architecture](#architecture)).

## Quick Start

```python
from pathlib import Path
from internal_datasets.raw_video import RawVideoDatasetAdapter

dataset = RawVideoDatasetAdapter(Path("data/my_raw_dataset"))
print(f"Dataset has {len(dataset)} frames")

obs = dataset[0]
print(f"State shape: {obs.state.shape}")
print(f"Action shape: {obs.action.shape}")
for cam_name, img in obs.images.items():
    print(f"Camera '{cam_name}': {img.shape}")  # (C, H, W)
```

### Temporal windowing

When `delta_indices` are set, the adapter returns stacked tensors along a temporal
axis, with boolean pad masks for positions clamped at episode boundaries:

```python
dataset.delta_indices = {
    "action": [0, 1, 2, 3],
    "observation.state": [-1, 0],
    "observation.images.top": [-1, 0],
}
obs = dataset[0]
print(obs.action.shape)   # (4, action_dim)
print(obs.state.shape)    # (2, state_dim)
print(obs.extra)           # {"action_is_pad": ..., "observation.state_is_pad": ..., ...}
```

## Integration with Training

The adapter implements the `Dataset` ABC, so it can be passed directly to the training
pipeline's `DataModule`:

```python
from pathlib import Path
from getiaction.data.datamodules import DataModule
from internal_datasets.raw_video import RawVideoDatasetAdapter

dataset = RawVideoDatasetAdapter(Path("data/my_raw_dataset"))
datamodule = DataModule(train_dataset=dataset)
```

## Architecture

### Module data flow

```
manifest.py --> frame_index.py --> stats.py --> adapter.py
```

1. **manifest.py** -- Parses and validates `manifest.json` using Pydantic models
   (`CameraConfig`, `EpisodeEntry`, `DatasetManifest`). `load_manifest()` also
   verifies that all referenced files and directories exist on disk.

2. **frame_index.py** -- Builds a flat, pre-computed lookup table (`FrameIndex`) that
   maps a global frame index `[0, total_frames)` to `(episode_idx, video_frame_idx,
   data_row_idx)` in O(1) time using numpy arrays.

3. **stats.py** -- Computes element-wise normalization statistics (mean, std, min, max)
   for state/action vectors using Welford's online algorithm (single pass, numerically
   stable). Per-channel image statistics are computed from a uniform sample of video
   frames (default 500). Results are cached to `.cache/stats.json`.

4. **adapter.py** -- `RawVideoDatasetAdapter(Dataset)` ties everything together. It
   pre-loads all scalar data into memory, decodes video frames lazily, and implements
   both simple (single-frame) and temporal windowing modes.

### FPS resampling

The video FPS is authoritative. The global frame index space is sized to the total
number of video frames across all episodes. When the number of JSONL data rows differs
from the number of video frames in an episode, nearest-neighbor resampling maps each
video frame to the closest data row:

```
data_row[i] = round(i * (num_data_rows - 1) / (num_video_frames - 1))
```

This allows recording telemetry at a different rate than the camera without requiring
manual alignment.

### Temporal windowing

When `delta_indices` are set (e.g. `{"action": [0, 1, 2, 3]}`), the adapter resolves
each offset relative to the current global index. At episode boundaries, indices are
clamped to stay within the same episode, and a boolean pad mask flags the clamped
positions. Supported delta keys:

- `"action"` -- stacks action vectors along a temporal axis.
- `"observation.state"` -- stacks state vectors.
- `"observation.images.<cam_name>"` -- batch-decodes video frames (uses
  `decode_frames()` for efficiency when all frames come from the same episode).

### Stats caching

Normalization statistics are stored at `<dataset_root>/.cache/stats.json`. The cache
is considered valid when its modification time is newer than `manifest.json`. If the
manifest is modified after the cache was written, stats are recomputed automatically.

## Video Backend

The adapter supports two video decoding backends, checked in order of preference:

| Backend      | Package           | Install command            | Notes                        |
|--------------|-------------------|----------------------------|------------------------------|
| **torchcodec** | `torchcodec`    | `pip install torchcodec`   | Preferred. GPU-capable, faster random access. |
| **pyav**       | `av`            | `pip install av`           | Fallback. Pure-software decoding.            |

**torchcodec** is preferred when available. If neither is installed, the adapter raises
an `ImportError` at construction time.

Both backends return frames as `float32` tensors in NCHW format with values in `[0, 1]`
and RGB channel order.

## Performance Notes

- **Scalar data is pre-loaded**: All state vectors, action vectors, and timestamps are
  read into contiguous numpy arrays during construction. Random access is a simple
  array index operation.
- **Video frames are decoded lazily**: Frames are decoded on demand per `__getitem__`
  call. No frames are held in memory between samples.
- **O(1) frame lookup**: The `FrameIndex` maps global indices to episode-local indices
  via pre-computed numpy arrays -- no binary search required.
- **Stats are cached**: Normalization statistics are computed once and persisted to
  `.cache/stats.json`. Subsequent loads skip computation entirely (validated by mtime).
- **Batch video decode**: When temporal windowing requests multiple frames from the
  same episode and camera, they are decoded in a single `decode_frames()` call.

## Limitations

- **All cameras in an episode must have the same frame count.** The adapter raises a
  `ValueError` during construction if cameras within an episode disagree on frame
  count.
- **Video resolution is assumed consistent.** Camera image shapes are probed from the
  first episode only and assumed identical across all episodes.
- **No built-in image augmentations.** Augmentations (cropping, color jitter, etc.)
  must be applied externally, e.g. via transforms in the training pipeline.
- **Variable-length episodes are not batched internally.** The adapter returns
  individual observations; batching and padding across episodes is handled by the
  DataLoader and collation logic.
