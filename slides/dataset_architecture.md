---
slides:
  title: "Raw Video Dataset Architecture"
---

# Raw Video Dataset Architecture

### Episode-isolated storage for teleoperation data

---

## Agenda

1. Why -- where LeRobot v3 creates friction
2. Architecture -- episode-isolated storage
3. Stats Pipeline -- per-episode caching with Welford accumulators
4. Code Architecture -- key abstractions and flows
5. Converters -- bidirectional LeRobot interop
6. Snapshots -- current approach and future direction
7. What's Next

---

## 1. Why

---

### Where LeRobot v3 creates friction

| Operation | Why it's hard |
|-----------|---------------|
| **Delete an episode** | Rewrite consolidated Parquet + video chunks, update all offsets, recompute stats |
| **Truncate an episode** | Re-encode video chunk, rewrite Parquet, update frame counts |
| **Review one episode** | Episode is spread across consolidated chunks -- no self-contained artifact |
| **Split / merge** | Episodes share physical files -- must duplicate or rewrite |

LeRobot packs multiple episodes into shared files. Great for sequential training reads, bad for everything else.

---

### Right tool, wrong job

LeRobot v3 is **optimized for training**.

We need a format that also supports:

- Data acquisition and recording
- Episode review and quality control
- Dataset curation (delete, truncate, split, merge)
- Iterative training with cheap re-runs

**Proposal**: stop using LeRobot as the source of truth. Derive it when needed.

---

## 2. Architecture

---

### Core principle

> **Store data in a manipulation-friendly, episode-isolated format.**
> **Convert to LeRobot v3 on demand for training.**

Each episode is a self-contained directory. The dataset is a manifest that references episodes.

---

### Episode-isolated storage layout

```
my_dataset/
  manifest.json
  episode_000/
    data.jsonl            # Per-frame state/action timeseries
    cam_overview.mp4      # One video per camera, original codec
    cam_gripper.mp4
    stats.json            # Per-episode normalization stats (Welford state)
  episode_001/
    ...
  stats.json              # Aggregated global stats (derived cache)
```

- Each episode directory is self-contained and independently deletable
- Videos stored in original codec -- no re-encoding on ingest
- Per-episode stats enable cheap global aggregation

-v-

### Detail: actual directory paths

Three key locations in the system:

| Location | Path | Purpose |
|----------|------|---------|
| **Source** | `~/.cache/physicalai/datasets/{name}/` | Persistent dataset on disk |
| **Cache** | `~/.cache/physicalai/cache/{uuid}/` | Temp copy during recording mutation |
| **Snapshot** | `~/.cache/physicalai/models/{model_uuid}/snapshot_YYYY-MM-DD_HH-MM-SS/` | Immutable copy for training |

-v-

### Detail: data flow -- recording

```
User records episode
        |
        v
RecordingMutation.start()
  copytree(source -> cache/{uuid})
        |
        v
Frames written to cache/{uuid}/episode_NNN/
  - VideoWriter encodes MP4
  - JSONL rows appended per frame
        |
        v
save_episode()
  - Updates manifest.json in cache
  - Spawns background thread: compute stats -> stats.json
        |
        v
RecordingMutation.teardown()
  copytree(cache -> source)   # overwrites source
  delete(cache)
```

-v-

### Detail: data flow -- training

```
Training job starts
        |
        v
SnapshotService.create()
  copytree(source -> snapshot dir)     # one-way copy
        |
        v
RawVideoDatasetAdapter.__init__()
  - Loads manifest.json
  - Builds frame index (JSONL -> memory)
  - load_or_compute_stats()
    - Checks per-episode stats.json cache
    - Computes missing, writes back to SOURCE
    - Merges all episodes -> global stats
        |
        v
DataLoader iterates
  __getitem__ -> lazy video decode per frame
```

The write-back to **source** is critical: without it, every new snapshot would recompute stats from scratch.

---

### Manipulation becomes trivial

| Operation | Before (LeRobot v3) | After (episode-isolated) |
|-----------|---------------------|--------------------------|
| **Delete episode** | Rewrite chunk files | `rm -rf` episode dir + update manifest |
| **Truncate episode** | Re-encode video chunk | New dir with trimmed video (remux) |
| **Split dataset** | Duplicate consolidated files | New manifest referencing same episodes |
| **Review one episode** | Seek into consolidated MP4 | Open `episode_000/cam_overview.mp4` |
| **Merge datasets** | Re-consolidate everything | Merge manifests, copy episode dirs |

---

## 3. Stats Pipeline

---

### Per-episode stats overview

Each episode stores Welford accumulator state in `stats.json`:

- **n** (count), **mean**, **m2** (sum of squared deviations), **min**, **max**
- Separate accumulators for state, action, and each camera

Global stats are **merged from per-episode stats** -- no raw data scan needed.

| Event | Cost |
|-------|------|
| Record an episode | ~200ms (background thread) |
| Delete an episode | Re-merge remaining (~1ms) |
| Train (warm cache) | Load + merge (~1ms) |
| Train (cold cache) | Decode + compute per episode, then merge |

-v-

### Detail: Welford accumulators

`WelfordAccumulator` tracks running statistics without storing raw data:

```python
class WelfordAccumulator:
    n: int           # sample count
    mean: np.ndarray # running mean
    m2: np.ndarray   # sum of squared deviations
    min: np.ndarray  # element-wise minimum
    max: np.ndarray  # element-wise maximum
```

**Key operations:**

- `update_batch(values)` -- vectorized numpy batch update (no per-pixel loops)
- `merge(other)` -- combine two accumulators using parallel Welford algorithm
- `to_dict()` / `from_dict()` -- JSON serialization for `stats.json`
- `variance` / `std` -- derived from m2/n

Merging two accumulators is O(dim), not O(samples). This is what makes global stats cheap.

-v-

### Detail: caching strategy

Stats are computed and cached at three points:

**1. Recording time** (background, non-blocking):
```
save_episode() -> Thread(compute_episode_stats_background)
  -> writes stats.json to cache dataset dir
  -> teardown copies it to source along with episode data
```

**2. Training time** (lazy, on cache miss):
```
load_or_compute_stats()
  for each episode:
    if stats.json exists in snapshot -> use it
    elif stats.json exists in source -> copy to snapshot, use it
    else -> decode video, compute, write to snapshot AND source
  merge all -> global stats
```

**3. Conversion time** (inline):
```
LeRobotToRawVideoConverter.convert()
  for each episode:
    write JSONL + video
    compute stats -> write stats.json
```

The **source write-back** in step 2 is the key insight: it ensures future snapshots benefit from any stats computed during previous training runs.

-v-

### Detail: the two caching bugs (fixed)

**Bug 1: Race condition in recording flow**

The background stats thread writes `stats.json` to the **cache** dir, but `teardown()` runs `copytree(cache -> source)` then `delete(cache)` without waiting for the thread to finish.

**Fix**: The thread targets the cache dir. Since `teardown()` copies the entire cache tree, if the thread finishes before teardown, the stats file gets copied along with everything else.

**Bug 2: Snapshot one-way copy (the main bug)**

When `compute_stats()` runs during training, it writes per-episode `stats.json` only into the **snapshot** directory. But snapshots are created fresh from source each time via `copytree`. Stats were recomputed from scratch on every training run.

**Fix**: Added `source_dataset_root` parameter throughout the stats pipeline. Newly computed stats are written back to the source dataset so future snapshots inherit them.

---

## 4. Code Architecture

---

### Key abstractions

| Class | Role |
|-------|------|
| `DatasetClient` (ABC) | 13 abstract methods: create, read, write, delete, mutation lifecycle |
| `RawVideoDatasetClient` | Implements all 13 methods for the raw video format |
| `RecordingMutation` | Cache-copy-then-overwrite pattern for safe recording |
| `SnapshotService` | `copytree(source, destination)` for training immutability |
| `RawVideoDatasetAdapter` | PyTorch `Dataset` subclass -- frame index, lazy decode, stats |

---

### Recording flow

```
start_recording_mutation()
  -> copytree(source -> cache/{uuid})
  -> sets _source_dataset_root on cache client

prepare_for_writing() -> creates episode dir, starts VideoWriters

add_frame() -> writes video frame + JSONL row

save_episode()
  -> closes VideoWriters
  -> updates manifest
  -> spawns background stats thread (passes source_dataset_root)

teardown()
  -> finalize() invalidates global stats cache
  -> overwrite() copies cache -> source
  -> delete() removes cache dir
```

---

### Training flow

```
run_loop()
  -> _train_model(source_dataset_path)
     -> _create_datamodule(source_dataset_root)
        -> RawVideoDatasetAdapter(
             dataset_root=snapshot_path,
             source_dataset_root=source_dataset_path
           )
           -> load_or_compute_stats(source_dataset_root=...)
              -> per-episode stats with write-back
```

The `source_dataset_root` parameter threads through from the top-level training loop all the way down to the stats computation functions.

---

## 5. Converters

---

### Bidirectional conversion

```
   LeRobot v3  <-------->  Raw Video
               convert-to-raw
               convert-to-lerobot
```

- **LeRobot -> Raw Video**: primary path for migrating existing datasets
- **Raw Video -> LeRobot**: for HuggingFace Hub upload, community sharing

Both directions are available via CLI (`converter_cli.py`).

-v-

### Detail: LeRobot to Raw Video conversion

For each episode:

1. Read frames from LeRobot's consolidated Parquet + video chunks
2. Write per-episode `data.jsonl` (state/action timeseries)
3. Remux or re-encode video to per-episode MP4 (one per camera)
4. **Compute per-episode stats** and write `stats.json`
5. Append episode entry to manifest

Step 4 is new -- it pre-populates the stats cache so the first training run on a converted dataset doesn't pay a cold-cache penalty.

Stats failures during conversion are **non-fatal**: logged as a warning, conversion continues. Stats will be lazily computed at training time instead.

-v-

### Detail: Raw Video to LeRobot conversion

Reverse direction -- used for ecosystem interop:

1. Read manifest + per-episode JSONL and video files
2. Decode video frames
3. Write LeRobot v3 format (Parquet + consolidated video)
4. LeRobot's own `save_episode()` handles its internal stats

No special stats handling needed -- LeRobot computes its own stats during `save_episode()`.

---

## 6. Snapshots

---

### Current approach: full physical copy

Before each training run, `SnapshotService` runs `shutil.copytree` on the entire dataset.

| Aspect | Current behavior |
|--------|-----------------|
| **Creation time** | Seconds to minutes (proportional to dataset size) |
| **Storage per snapshot** | Full dataset size |
| **10 snapshots of 5 GB** | ~50 GB total |
| **Write-back** | Stats written back to source via `source_dataset_root` |

This works correctly and provides training immutability. The stats write-back ensures it's not as expensive as it looks -- stats are only computed once across all snapshots.

---

### Future: manifest-based snapshots

**A snapshot could be a frozen copy of the manifest, not a copy of the data.**

```
my_dataset/
  manifest.json                      # Current (mutable)
  snapshots/
    snap_2026-02-10_14-30.json       # Frozen manifest (~10 KB)
  episode_000/                       # Shared by all snapshots
  episode_001/
```

| Aspect | Current (copytree) | Manifest snapshots |
|--------|--------------------|--------------------|
| **Creation** | O(dataset size) | O(1) -- copy one JSON |
| **Storage** | Full copy per snapshot | ~10 KB per snapshot |
| **10 snapshots of 5 GB** | ~50 GB | ~5 GB + ~100 KB |
| **Provenance** | Model -> opaque copy | Model -> snapshot -> exact episode list |

Requires copy-on-write semantics for episode mutations (truncate creates a new dir, old dir stays for prior snapshots) and garbage collection of unreferenced episode dirs.

---

## 7. What's Next

---

### Remaining work

| Area | Status | Notes |
|------|--------|-------|
| Episode-isolated storage | Done | Manifest, JSONL, per-episode video |
| Per-episode stats + caching | Done | Welford, background compute, source write-back |
| Training adapter (Path A) | Done | Lazy video decode, frame index, stats from cache |
| Bidirectional converters | Done | With stats pre-computation in LeRobot -> Raw |
| Recording flow | Done | RecordingMutation, background stats |
| Manifest-based snapshots | Not started | Current copytree works, this is an optimization |
| Episode manipulation (truncate, split) | Not started | Architecture supports it, needs UI + API |
| Optimized conversion (bypass PNG) | Not started | ~10x faster LeRobot conversion path |

---

# Questions?

---
