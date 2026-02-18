# Training From a Raw Dataset (Video + Action/State Files)

This document analyzes the feasibility of training policies in geti-action from raw
datasets consisting of video files plus corresponding per-frame action and observation
state vectors, without first converting to the LeRobot on-disk format.

---

## 1. What You Have

- Video files (at some FPS)
- Per-frame action vectors (robot joint commands)
- Per-frame observation state vectors (robot joint positions)
- The video and action/state data may be at different FPS

---

## 2. What Training Actually Needs

The interface contract between the dataset layer and the policy layer is clean.
Policies **never** touch the LeRobot dataset directly. They go through this chain:

```
Your data -> _LeRobotDatasetAdapter (or replacement) -> Dataset abstract class -> Policy.setup()
```

Policies only access:

| Property / Method              | Purpose                                                                  |
|--------------------------------|--------------------------------------------------------------------------|
| `dataset.observation_features` | `dict[str, Feature]` with `NormalizationParameters`                      |
| `dataset.action_features`      | `dict[str, Feature]` with `NormalizationParameters`                      |
| `dataset.stats`                | Derived automatically from the above two properties                      |
| `dataset.fps`                  | Integer FPS of the dataset                                               |
| `dataset.tolerance_s`          | Float tolerance for delta timestamp validation                           |
| `dataset.delta_indices`        | Temporal windowing offsets (set by `reformat_dataset_to_match_policy()`) |
| `dataset.__getitem__(idx)`     | Returns an `Observation` dataclass                                       |
| `dataset.__len__()`            | Total number of frames                                                   |

**Nothing in the training path reads `info.json`, `tasks.parquet`, `episodes/`
metadata, or `stats.json` directly.** All of that is consumed by the upstream
`LeRobotDataset` class, which exposes it through `dataset.meta.stats` and
`dataset.meta.features`. The `_LeRobotDatasetAdapter` converts those into the
`Feature` / `NormalizationParameters` objects that policies consume.

### 2.1. The Abstract Interface Contract

The base class is `library/src/getiaction/data/dataset.py`. A valid dataset
implementation must provide:

| Member                 | Type              | Signature                         |
|------------------------|-------------------|-----------------------------------|
| `__getitem__`          | abstractmethod    | `(self, idx: int) -> Observation` |
| `__len__`              | abstractmethod    | `(self) -> int`                   |
| `raw_features`         | abstract property | `-> dict`                         |
| `observation_features` | abstract property | `-> dict[str, Feature]`           |
| `action_features`      | abstract property | `-> dict[str, Feature]`           |
| `fps`                  | abstract property | `-> int`                          |
| `tolerance_s`          | abstract property | `-> float`                        |
| `delta_indices`        | abstract property | getter + setter                   |

The `stats` property is **concrete** on the base class -- it derives entirely from
`observation_features` and `action_features`. You do not need to implement it; you
just need to ensure your features carry correct `NormalizationParameters`.

### 2.2. What Each Policy Reads

| Policy  | What It Reads From the Dataset                                   |
|---------|------------------------------------------------------------------|
| ACT     | `observation_features`, `action_features`                        |
| Pi0     | `stats` (which reads `observation_features` + `action_features`) |
| Groot   | `stats`, `action_features`                                       |
| SmolVLA | `stats` (same as Pi0)                                            |

All policies also interact with `delta_indices` via `reformat_dataset_to_match_policy()`.

### 2.3. What Is NOT Needed

- `episode_data_index` -- never accessed during training
- `meta.info` (the raw `info.json` dict) -- never accessed by policies
- `hf_dataset` -- only used in `video_repair.py`, not in training
- Parquet files, tasks metadata, episode metadata -- all consumed by the upstream
  `LeRobotDataset` internals, which we bypass entirely

---

## 3. What You Need to Build

A custom adapter that implements the `Dataset` abstract interface. The requirements
break down as follows:

| Requirement                               | How to Satisfy                                                                                                                           | Online Computable?                                         |
|-------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------|
| **Stats** (mean/std/min/max per feature)  | Single pass over all frames. Running stats for state/action, channel-wise stats for images.                                              | Yes. Compute once, cache locally.                          |
| **Feature metadata** (shape, dtype, name) | Infer from data: video resolution gives image shape, vector lengths give state/action shapes.                                            | Yes. Trivial.                                              |
| **`__getitem__` returning `Observation`** | Decode video frame at index, pair with corresponding action/state. Handle FPS mismatch.                                                  | Yes.                                                       |
| **`episode_data_index`**                  | **Not needed.** Never accessed during training.                                                                                          | N/A                                                        |
| **`delta_indices`** (temporal windowing)  | Must be a settable attribute. `reformat_dataset_to_match_policy()` writes it based on policy config. Your `__getitem__` must respect it. | Yes, but `__getitem__` must handle multi-timestep queries. |
| **FPS alignment**                         | If video and state/action are at different FPS, resample to a unified FPS.                                                               | Yes.                                                       |

---

## 4. The FPS Mismatch Problem

LeRobot assumes one unified FPS: every frame has exactly one observation state, one
action, and one image, all at the same timestamp. If your video is 30fps but actions
are recorded at 50Hz, you need a resampling strategy.

**Option A -- Resample actions/states to match video FPS** (simpler):
Each video frame gets the nearest (or linearly interpolated) action/state. This
loses some action resolution but keeps video decode simple.

**Option B -- Resample video to match action FPS**:
Decode intermediate video frames or duplicate nearest frames. Preserves action
temporal resolution but is more expensive and wastes memory on redundant frames.

This resampling should happen during the initial data scan / indexing step, not in
every `__getitem__` call. Pre-compute a frame alignment table mapping unified frame
indices to (video_frame_idx, state_idx, action_idx) tuples.

---

## 5. Online Stats Computation With Caching

Stats computation is the main "extra" work. The approach:

1. On construction, check for a cached stats file in `cache_dir`.
2. If not cached, do a **single pass** over the data:
   - For **state/action**: Use Welford's online algorithm for running mean/std.
     Track min/max element-wise. This requires O(D) memory where D is the
     feature dimension -- no need to load all frames into memory.
   - For **images**: Channel-wise running stats over all decoded frames. This is
     the expensive part (requires decoding every video frame once). Consider
     sampling a subset of frames if the dataset is very large.
3. Save stats to a local cache file (JSON or pickle).
4. On subsequent loads, read from cache. Skip computation entirely.

The cached stats must contain, per feature:

```python
{
    "mean": list[float],   # element-wise mean
    "std": list[float],    # element-wise standard deviation
    "min": list[float],    # element-wise minimum
    "max": list[float],    # element-wise maximum
}
```

These are wrapped into `NormalizationParameters` and attached to each `Feature`
object returned by `observation_features` / `action_features`. The base class
`stats` property then derives the canonical stats dict automatically.

---

## 6. Skeleton Implementation

```python
class RawVideoDatasetAdapter(Dataset):
    """Wraps raw video + state/action files for training.

    Does not require LeRobot on-disk format. Computes stats online
    and caches them locally.
    """

    def __init__(
        self,
        video_paths: list[Path],
        state_data: list[np.ndarray],    # per-episode arrays, shape (T, D_state)
        action_data: list[np.ndarray],   # per-episode arrays, shape (T, D_action)
        video_fps: int,
        data_fps: int,                   # FPS of state/action data
        target_fps: int,                 # unified FPS for training
        cache_dir: Path,
        camera_names: list[str] | None = None,
    ):
        # 1. Build frame alignment table (handles FPS mismatch)
        # 2. Compute episode boundaries
        # 3. Infer feature shapes from data
        # 4. Compute or load cached stats
        ...

    # --- Abstract interface implementation ---

    @property
    def observation_features(self) -> dict[str, Feature]:
        """Return Features with NormalizationParameters from cached stats."""
        # Build Feature for each camera (VISUAL type)
        # Build Feature for observation.state (STATE type)
        # Attach NormalizationParameters from cached stats
        ...

    @property
    def action_features(self) -> dict[str, Feature]:
        """Return action Feature with NormalizationParameters."""
        ...

    @property
    def raw_features(self) -> dict:
        """Return dict with keys like 'observation.state',
        'observation.images.<name>', 'action'. Used by
        reformat_dataset_to_match_policy() to match delta indices."""
        ...

    @property
    def fps(self) -> int:
        return self._target_fps

    @property
    def tolerance_s(self) -> float:
        return 1e-4

    @property
    def delta_indices(self) -> dict[str, list[int]] | None:
        return self._delta_indices

    @delta_indices.setter
    def delta_indices(self, indices: dict[str, list[int]]) -> None:
        self._delta_indices = indices

    def __len__(self) -> int:
        return self._total_frames

    def __getitem__(self, idx: int) -> Observation:
        """Decode video frame, look up aligned state/action.

        Must handle delta_indices for temporal windowing:
        - When delta_indices is set, return stacked tensors for
          multiple timesteps per feature
        - Generate *_is_pad masks for frames outside episode boundaries
        - Clamp indices to stay within the current episode
        """
        ...

    # --- Stats computation ---

    def _compute_or_load_stats(self, cache_dir: Path) -> dict:
        """Compute stats with single-pass Welford's algorithm,
        or load from cache if available."""
        cache_path = cache_dir / "raw_dataset_stats.json"
        if cache_path.exists():
            return self._load_cached_stats(cache_path)

        stats = {}
        # State/action: running mean/std/min/max (Welford's)
        # Images: channel-wise running stats over decoded frames
        ...

        self._save_stats_cache(stats, cache_path)
        return stats
```

---

## 7. Integration Point

The training worker currently constructs a `LeRobotDataModule`:

```python
# application/backend/src/workers/training_worker.py
l_dm = LeRobotDataModule(repo_id="snapshot", root=snapshot.path, train_batch_size=8)
```

To use the raw adapter, you would bypass `LeRobotDataModule` and use the base
`DataModule` directly:

```python
from getiaction.data.datamodule import DataModule

adapter = RawVideoDatasetAdapter(
    video_paths=...,
    state_data=...,
    action_data=...,
    video_fps=30,
    data_fps=50,
    target_fps=30,
    cache_dir=snapshot_path / ".cache",
)
dm = DataModule(train_dataset=adapter, train_batch_size=8)
```

The base `DataModule` (`library/src/getiaction/data/datamodule.py`) accepts a
`train_dataset` directly, so no changes to the datamodule layer are needed.

The policy setup code does not need changes either -- it accesses the dataset
only through the abstract `Dataset` interface, which your adapter implements.

---

## 8. Gotchas

1. **`delta_indices` / temporal windowing**: Your `__getitem__` must handle
   multi-timestep queries. When `delta_indices` is set, each feature key maps to
   a list of frame offsets. You need to return stacked tensors and `*_is_pad`
   masks for frames that fall outside episode boundaries. Study
   `LeRobotDataset.__getitem__` in the upstream library for the exact logic.

2. **Image format**: Policies expect images as `(C, H, W)` float32 tensors in
   `[0, 1]` range. LeRobot's `hf_transform_to_torch` does this conversion
   (divide by 255, permute HWC -> CHW). Your adapter must replicate this.

3. **Episode boundaries**: Your adapter must know where episodes start and end.
   Even though `episode_data_index` is never accessed externally, your own
   `__getitem__` needs this information to prevent temporal windows from crossing
   episode boundaries and to generate correct `*_is_pad` masks.

4. **`raw_features` must use LeRobot key conventions**:
   `reformat_dataset_to_match_policy()` iterates over `raw_features` looking for
   keys starting with `"observation."` and the key `"action"`. Your features must
   follow this naming: `"observation.state"`, `"observation.images.<camera_name>"`,
   `"action"`.

5. **`Observation` dataclass**: Your `__getitem__` must return an `Observation`
   instance (from `library/src/getiaction/data/observation.py`), not a raw dict.
   Study `FormatConverter.to_observation()` in
   `library/src/getiaction/data/lerobot/converters.py` for the expected structure.

6. **Stats for images are channel-wise**: Image stats must have shape `(C, 1, 1)`,
   not `(C, H, W)`. The normalization code reshapes to `(channels, 1, 1)` for
   broadcasting. If you compute full-resolution stats, squeeze the spatial dims.

7. **FPS alignment is a one-time cost**: Pre-compute the frame alignment table
   during `__init__`, not during `__getitem__`. The table maps each unified frame
   index to (episode_idx, video_frame_idx, state_row_idx, action_row_idx).

8. **Video decode performance**: Decoding video frames in `__getitem__` is the
   bottleneck. Consider using `torchvision.io` or `decord` for efficient random
   access. If your videos are H.264/H.265, seeking to arbitrary frames requires
   keyframe-based seeking followed by sequential decode, which can be slow for
   frames far from keyframes.

9. **Stats computation for images is expensive**: A full pass over all video
   frames to compute channel-wise mean/std/min/max can take a long time for large
   datasets. Consider computing stats over a random sample of frames (e.g., every
   Nth frame) and document this as an approximation. The normalization is not
   highly sensitive to exact stats values for images since most policies use
   `IDENTITY` normalization for visual features anyway (ACT, SmolVLA). Only
   Groot and Pi0 might use image stats, and even then, the vision encoders
   typically have their own internal normalization.

---

## 9. Effort Estimate

| Component                             | Complexity      | Notes                                                                                    |
|---------------------------------------|-----------------|------------------------------------------------------------------------------------------|
| FPS alignment / frame indexing        | Moderate        | One-time computation, but must handle edge cases (rounding, boundary alignment)          |
| Video frame decoding in `__getitem__` | Moderate        | Standard with `decord` or `torchvision.io`; temporal windowing adds complexity           |
| Temporal windowing (`delta_indices`)  | Moderate        | Must replicate LeRobot's `__getitem__` logic for clamping, padding, and `*_is_pad` masks |
| Online stats computation + caching    | Straightforward | Single-pass Welford's for state/action; channel-wise for images                          |
| Feature metadata inference            | Trivial         | Read shapes from data files                                                              |
| `Observation` construction            | Straightforward | Follow `FormatConverter.to_observation()` pattern                                        |
| Integration with `DataModule`         | Trivial         | Base `DataModule` accepts `train_dataset` directly                                       |

**No changes needed to**: policies, normalization code, training worker (beyond
the `DataModule` construction), or the library's data abstract classes.

**No LeRobot on-disk format required**: No parquet files, no `info.json`, no
`stats.json`, no episode metadata, no `tasks.parquet`.
