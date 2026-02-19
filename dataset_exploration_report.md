# Dataset Architecture Exploration Report

This document evaluates LeRobot's v3.0 dataset format from the perspective of a
platform that needs to **acquire**, **review/manipulate**, and **train on**
teleoperation data. It identifies where the current format creates friction for
non-training workflows and proposes an architecture that addresses these gaps.

---

## 1. Background: LeRobot v3.0 Format Design

LeRobot v3.0 stores datasets in a consolidated layout:

```
dataset/
├── meta/
│   ├── info.json              # Feature definitions, FPS, totals
│   ├── stats.json             # Pre-computed global normalization statistics
│   ├── tasks.parquet          # Task index → task string mapping
│   └── episodes/
│       └── chunk-000/
│           └── file-000.parquet   # Episode metadata + per-episode stats
├── data/
│   └── chunk-000/
│       ├── file-000.parquet   # Multiple episodes of frame data consolidated
│       └── ...
└── videos/
    └── observation.images.<camera>/
        └── chunk-000/
            ├── file-000.mp4   # Multiple episodes concatenated into one video
            └── ...
```

Key design decisions:

- **Consolidated Parquet files**: multiple episodes packed into single files,
  capped at ~100 MB per file. Great for sequential training scans; bad for
  editing a single episode.
- **Consolidated video files**: multiple episodes concatenated into single MP4
  files, capped at ~200 MB. Episode boundaries are tracked via timestamp offsets
  in episode metadata. Fast sequential reads; but you cannot delete episode 37
  without rewriting the entire chunk.
- **Pre-computed global stats** in `meta/stats.json`: normalization parameters
  (mean, std, min, max, quantiles) are ready to use at training time. But any
  episode modification requires full recomputation.
- **Per-episode stats** embedded in episode metadata Parquet files: enables
  incremental aggregation when adding new episodes, but the storage format
  (flattened columns in consolidated Parquet) makes them hard to access or
  update independently.

---

## 2. Where LeRobot v3.0 Creates Friction

### 2.1. Dataset Manipulation

| Operation                                   | Difficulty | Why                                                                                                                   |
|---------------------------------------------|------------|-----------------------------------------------------------------------------------------------------------------------|
| **Delete an episode**                       | Hard       | Must rewrite the consolidated Parquet + video chunk files, update all episode offset metadata, recompute global stats |
| **Truncate an episode** (remove bad frames) | Hard       | Must re-encode the video chunk, rewrite the Parquet chunk, update frame counts and timestamp offsets, recompute stats |
| **Split a dataset** (train/val by episode)  | Medium     | Requires duplicating or rewriting consolidated files since episodes share physical files                              |
| **Merge datasets** from different sessions  | Medium     | Must re-consolidate Parquet and video files, reconcile episode indices and task tables                                |
| **Reorder episodes**                        | Hard       | Consolidated files encode ordering implicitly via offsets; reordering means rewriting everything                      |

### 2.2. Episode Review

Reviewing individual episodes in a LeRobot dataset requires:

1. Reading episode metadata to find the chunk file and timestamp offsets
2. Seeking into a consolidated MP4 by timestamp (codec-dependent, fragile)
3. Cross-referencing Parquet data by global frame index range

There is no way to hand someone "episode 12" as a self-contained artifact --
it is physically spread across consolidated chunk files that also contain other
episodes.

### 2.3. Conversion Performance

Converting from a per-episode format (like raw video files from teleoperation)
into LeRobot v3.0 is expensive because of LeRobot's `add_frame()` API design:

```
Source video → decode frames → add_frame() writes PNG to disk
                                        ↓
       encode to AV1 MP4 ← save_episode() reads PNGs back from disk
```

Every frame goes through a **PNG round-trip**: decoded from the source video,
written as a PNG file by `add_frame()`, then read back from disk during
`save_episode()` for both stats computation and AV1 video encoding. This
round-trip accounts for roughly 55% of total conversion time. See
`converting_episodes.md` for detailed benchmarks.

For 100 episodes of 30 seconds each (640x480, 30 FPS, 2 cameras), conversion
takes approximately **90-100 minutes**. The breakdown:

| Component                       | % of time | Notes                                          |
|---------------------------------|-----------|------------------------------------------------|
| AV1 video encoding (SVT-AV1)    | ~50%      | Re-encodes every frame from PNG                |
| PNG writes + reads (round-trip) | ~30%      | Forced by `add_frame()` API                    |
| Source video decoding           | ~13%      | Our code, unavoidable                          |
| Stats computation               | ~5%       | Subsampled image scan + trivial numerical scan |
| Parquet + metadata              | ~2%       | Fast                                           |

### 2.4. Summary

LeRobot v3.0 is **heavily optimized for training**: pre-computed stats, consolidated
files for fast sequential I/O, everything ready to feed directly into a DataLoader.
But it is **poorly suited as a source-of-truth format** for a platform where data
acquisition, review, quality control, and iterative dataset curation are first-class
workflows.

---

## 3. Recommended Architecture

### 3.1. Core Principle

**Store data in a manipulation-friendly, episode-isolated format. Convert to
LeRobot v3.0 on demand for training.**

The primary storage format should treat each episode as a self-contained unit
that can be independently created, inspected, edited, deleted, tagged, and
moved. The training-optimized format (LeRobot v3.0) becomes a derived/compiled
artifact, generated when the user initiates training and cached for reuse.

### 3.2. Primary Storage: Episode-Isolated Format

```
dataset/
├── manifest.json                  # Dataset-level metadata
├── snapshots/                     # Frozen manifests (see section 5)
│   └── snap_2026-02-10_14-30.json
├── episodes/
│   ├── ep_000_abc1/
│   │   ├── data.jsonl             # Per-frame state/action timeseries
│   │   ├── cam_overview.mp4       # One video per camera, original codec
│   │   ├── cam_gripper.mp4
│   │   └── stats.json             # Per-episode normalization stats
│   ├── ep_001_def2/
│   │   └── ...
│   └── ...
└── .cache/
    └── stats.json                 # Aggregated global stats (derived)
```

Each episode directory is fully self-contained. The manifest tracks dataset-level
metadata (FPS, joint dimensions, joint names, robot type, camera list) and an
ordered list of episode references.

**Manipulation becomes trivial:**

| Operation           | How                                                                               |
|---------------------|-----------------------------------------------------------------------------------|
| Delete an episode   | Remove from manifest, garbage-collect episode directory (see section 5.5)         |
| Truncate an episode | Copy-on-write: new episode directory with trimmed data (see section 5.4)          |
| Split dataset       | Create new manifests referencing the same episode directories                     |
| Merge datasets      | Merge manifests, copy episode directories into shared `episodes/`                 |
| Review an episode   | Open `episodes/ep_000_abc1/cam_overview.mp4` in any media player                 |
| Tag/annotate        | Add fields to manifest's episode entry or a per-episode metadata file             |

### 3.3. Per-Episode Stats for Cheap Global Aggregation

Each episode stores its own normalization statistics immediately after recording
(or after any modification). These per-episode stats include:

- `mean`, `std`, `min`, `max` for state and action vectors (element-wise)
- `mean`, `std`, `min`, `max` for image features (per-channel)
- `count` (number of frames in the episode)

Global dataset stats are then **aggregated from per-episode stats in O(episodes)
time** without touching any raw data:

| Statistic     | Aggregation method                                                                          |
|---------------|---------------------------------------------------------------------------------------------|
| **mean**      | Weighted average: `global_mean = sum(ep_mean * ep_count) / sum(ep_count)`                   |
| **std**       | Parallel variance algorithm (Welford-style): combines per-episode mean, variance, and count |
| **min / max** | Element-wise min/max across all per-episode values                                          |
| **count**     | Sum of all per-episode counts                                                               |

This means:
- Adding a new episode: compute its stats (~200ms for numerical, ~2s for
  subsampled images), then re-aggregate (~1ms)
- Deleting an episode: re-aggregate from remaining episodes' stats (~1ms)
- No operation ever requires re-scanning all raw frames

### 3.4. On-Demand Conversion to LeRobot v3.0

When training is requested, the platform converts the selected episodes into
a LeRobot v3.0 dataset. Three levels of optimization are possible:

#### Level 1: Naive approach (functional, slow)

Go through LeRobot's `add_frame()` / `save_episode()` / `finalize()` API,
feeding it decoded frames from the episode-isolated source. This works
correctly but is slow (~1 minute per episode) due to the PNG round-trip
and AV1 re-encoding.

#### Level 2: Bypass the PNG round-trip (significant speedup)

Write Parquet data files directly (we already have the frame data in
memory/on disk) and use ffmpeg to remux/concatenate episode videos into
the consolidated chunk format without decoding and re-encoding. Compute
global stats from cached per-episode stats. This eliminates ~55% of
conversion time.

#### Level 3: Cached incremental conversion (best steady-state performance)

Cache the LeRobot output directory. Track a content hash or modification
timestamp per episode. When training is re-requested:

- If nothing changed: reuse the cache entirely (instant)
- If 3 episodes were added: convert only those 3, append to existing
  Parquet/video chunks, re-aggregate stats
- If an episode was deleted: rewrite only the affected chunk files,
  re-aggregate stats

This makes the common case (re-training after small dataset changes)
nearly free.

---

## 4. Implementation Plan

### 4.1. Key Components

To realize the architecture described in section 3, the following components
are needed:

#### A. Manifest and Episode Models

A Pydantic-based schema defining the dataset-level manifest and per-episode
metadata. The manifest should capture:

- Dataset-level: FPS, state/action dimensions, joint names, robot type,
  camera configurations (name, resolution, codec)
- Per-episode: directory path, data file reference, video file references
  per camera, frame count, optional annotations/tags

A `load_manifest()` function validates the on-disk JSON and returns a typed
model. All manipulation operations work through the manifest.

#### B. Per-Episode Stats Computation and Aggregation

Each episode stores its own normalization statistics immediately after
recording or modification. Per-episode stats include:

- `mean`, `std`, `min`, `max` for state and action vectors (element-wise)
- `mean`, `std`, `min`, `max` for image features (per-channel)
- `count` (number of frames in the episode)

An online accumulator (e.g., Welford's algorithm) computes these in a single
pass. A separate aggregation function combines per-episode stats into global
dataset stats in O(episodes) time without touching raw data, using the
parallel variance algorithm.

A cache-aware loader checks staleness (via mtime or content hash) and
recomputes only when necessary.

#### C. Frame Index

An O(1) lookup structure that maps a global frame index to the corresponding
episode and local frame offset. Built from episode frame counts in the
manifest. Used by training adapters and any UI that needs random access by
global index.

#### D. Video Decoding

Frame-level video decoding supporting at minimum torchcodec and pyav
backends. Must support decoding single frames (for random access during
training) and frame ranges (for episode review/export). Should handle both
keyframe-seeking and sequential decoding efficiently.

#### E. Training Adapter

A dataset adapter implementing the platform's training `Dataset` interface.
Loads scalar data (state, action) into memory from JSONL, decodes video
frames lazily on `__getitem__`, applies temporal windowing (observation
history, action chunking), and provides normalization parameters from the
cached global stats. This is the **direct training path** -- no LeRobot
conversion needed.

#### F. Bidirectional Converters

Two converters to bridge the episode-isolated format and LeRobot v3.0:

1. **LeRobot → Episode-Isolated**: Imports existing LeRobot datasets into
   the manipulation-friendly format. Extracts per-episode data from
   consolidated Parquet/video files, preserves joint names and robot type.

2. **Episode-Isolated → LeRobot**: Compiles the episode-isolated dataset
   into LeRobot v3.0 for workflows that require it (e.g., using LeRobot's
   training scripts, uploading to HuggingFace Hub).

#### G. Episode Manipulation Helpers

A small set of manifest-level operations:

- **`remove_episode(manifest, episode_idx)`** -- removes an episode entry.
  Does not touch files (caller handles deletion). Re-aggregates global stats
  from remaining per-episode stats.
- **`add_episode(manifest, episode_entry)`** -- appends a new episode,
  computes its per-episode stats, re-aggregates global stats.
- **`subset_manifest(manifest, episode_indices)`** -- creates a new manifest
  containing only the selected episodes. Useful for train/val splits.

These are purely metadata operations. Episode directories remain untouched.

### 4.2. Architecture Diagram

```
                        PRIMARY STORAGE (episode-isolated)
                        ──────────────────────────────────
                        manifest.json
                        snapshots/          ← frozen manifests
                        episodes/
                          ep_000_abc1/      ← self-contained, deletable
                            data.jsonl
                            cam_overview.mp4
                            cam_gripper.mp4
                            stats.json      ← per-episode stats
                          ep_001_def2/
                            ...

                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        ┌───────────┐  ┌───────────┐  ┌──────────────┐
        │  REVIEW   │  │ MANIPULATE│  │    TRAIN     │
        │           │  │           │  │              │
        │ Play .mp4 │  │ rm -rf ep │  │ Path A:      │
        │ Thumbnail │  │ Truncate  │  │ Direct via   │
        │ Annotate  │  │ Split     │  │ training     │
        │ Score     │  │ Merge     │  │ adapter      │
        └───────────┘  └─────┬─────┘  │ (no convert) │
                             │        │              │
                             ▼        │ Path B:      │
                    Re-aggregate      │ Convert to   │
                    global stats      │ LeRobot v3   │
                    from per-ep       │ (cached,     │
                    stats (~1ms)      │  incremental) │
                                      └──────┬───────┘
                                             │
                                             ▼
                                    .cache/lerobot/
                                    (derived, disposable)
```

**Training path A** (direct adapter) reads from the episode-isolated format,
decodes video on demand, and provides normalization stats from the cached
global stats. No conversion required.

**Training path B** (LeRobot conversion) is available for workflows that
specifically need the LeRobot v3.0 format. The optimizations described in
section 3.4 (Levels 2 and 3) would make this path significantly faster.

### 4.3. Priority Order

| Priority | Change                                               | Effort | Impact                                                   |
|----------|------------------------------------------------------|--------|----------------------------------------------------------|
| 1        | Manifest schema, episode models, manifest loading    | Low    | Foundation for everything else                           |
| 2        | Per-episode stats computation + global aggregation   | Low    | Enables O(1ms) re-aggregation after any dataset mutation |
| 3        | Frame index and video decoding                       | Low    | Required by training adapter and review workflows        |
| 4        | Training adapter (direct path A)                     | Medium | Enables training without LeRobot conversion              |
| 5        | Bidirectional converters (path B)                    | Medium | Interop with LeRobot ecosystem                           |
| 6        | Episode manipulation helpers (remove, add, subset)   | Low    | Unlocks review/curation workflows                        |
| 7        | Optimized LeRobot conversion (bypass PNG round-trip) | Medium | ~10x faster conversion for path B                        |
| 8        | Conversion cache with incremental updates            | Medium | Near-instant re-conversion for the common case           |

---

## 5. Dataset Snapshots

### 5.1. Problem Statement

A common workflow in iterative robot learning:

1. Record episodes, producing dataset **D1**
2. Train model **A1** on D1
3. Curate the dataset -- delete bad episodes, truncate others, record new
   ones -- producing **D2**
4. Train model **A2** on D2
5. A2 performs worse. Want to retrain A1 further on the **original D1**

Step 5 requires the platform to recover the exact dataset state that A1 was
trained on, even though the live dataset has since been modified. This is the
snapshot problem: associating a training run with an immutable view of the
dataset at the time training was initiated.

### 5.2. Current Approach and Its Limitations

The backend already has a snapshot mechanism. Before each training run, the
training worker performs a `shutil.copytree` of the entire dataset directory
into the model's storage directory. A `SnapshotDB` record links the copy
back to the original dataset, and the `ModelDB` record references both the
dataset and the snapshot.

This approach has the right semantics (each training run gets an immutable
copy) but creates practical problems:

| Issue                          | Impact                                                                                                                                                                                                                     |
|--------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Full physical copy**         | Every training run duplicates all video files, Parquet data, and metadata. For a 100-episode dataset with 2 cameras at 640x480, this is several GB per snapshot.                                                           |
| **Linear storage growth**      | N training runs = N full copies of the dataset, even if nothing changed between runs                                                                                                                                       |
| **Slow snapshot creation**     | Copying GB of video data takes seconds to minutes, blocking the training pipeline                                                                                                                                          |
| **No user-facing snapshots**   | Snapshots are created implicitly at training time and buried inside model directories. There is no way to manually create a snapshot ("freeze this dataset state"), list snapshots, compare them, or select one for reuse. |
| **No retraining on snapshots** | `TrainJobPayload` accepts a `dataset_id` but not a `snapshot_id`. There is no path to say "train using the dataset state from this previous snapshot."                                                                     |

### 5.3. Proposed Approach: Manifest-Based Snapshots

With episode-isolated storage, snapshots become dramatically cheaper by
separating the question "which episodes are in this dataset?" (the manifest)
from the episode data itself.

**Core idea: a snapshot is a frozen copy of the manifest, not a copy of the
data.**

Episode data directories are the shared, immutable substrate. The manifest
defines which episodes (and which versions of those episodes) constitute a
particular dataset state. Freezing the manifest freezes the dataset state.

```
dataset/
├── manifest.json                  # Current (mutable) dataset state
├── snapshots/
│   ├── snap_2026-02-10_14-30.json # Frozen manifest from before curation
│   └── snap_2026-02-18_09-15.json # Frozen manifest from before re-record
├── episodes/
│   ├── ep_000_abc1/               # Original episode (referenced by both snapshots)
│   │   ├── data.jsonl
│   │   ├── cam_overview.mp4
│   │   └── stats.json
│   ├── ep_001_def2/               # Original episode, truncated in current manifest
│   │   └── ...
│   ├── ep_001_def2_trunc_080/     # Truncated version (new directory)
│   │   └── ...
│   ├── ep_005_ghi3/               # New episode added after first snapshot
│   │   └── ...
│   └── ...
└── .cache/
    └── stats.json
```

**Snapshot creation** is O(1): copy `manifest.json` to
`snapshots/<name>.json`. No data is duplicated.

**Snapshot restoration** is O(1): copy the snapshot file back over
`manifest.json`, then re-aggregate global stats from the referenced
episodes' per-episode stats (~1ms).

**Storage overhead** per snapshot: one JSON file, typically under 10 KB.

### 5.4. Episode Mutability and Copy-on-Write

Addition and deletion of episodes are purely manifest operations -- add or
remove an entry, no episode data changes. But truncation and splitting
modify an episode's content. The manifest-snapshot model requires that
episode directories referenced by a snapshot are never modified in place.

The solution is **copy-on-write at the episode level**: any operation that
would change an episode's content instead creates a new episode directory
with the modified data, and updates the manifest to point to the new
directory. The original directory remains untouched, still referenced by
any snapshots that include it.

#### Truncation

Truncation removes frames from the start or end of an episode (e.g.,
trimming a bad approach or a post-grasp wobble).

1. Create a new episode directory (e.g., `ep_001_def2_trunc_080/`)
2. Trim the video file(s) using ffmpeg stream copy (cut on keyframe
   boundaries -- no re-encoding in most cases, or re-encode only the
   single GOP at the cut point)
3. Truncate the JSONL data file (drop the corresponding lines)
4. Recompute per-episode stats for the new, shorter episode
5. Update the manifest to reference the new directory in place of the old
6. The old directory (`ep_001_def2/`) remains on disk, referenced by any
   prior snapshots

**Cost**: one partial video remux + one stats recomputation. Seconds, not
minutes.

#### Splitting

Splitting divides one episode into two (e.g., separating a pick from a
place within a single recording).

1. Create two new episode directories (e.g., `ep_003_jkl4_part_a/`,
   `ep_003_jkl4_part_b/`)
2. Split the video file(s) at the chosen frame boundary (ffmpeg stream
   copy, same keyframe caveat as truncation)
3. Split the JSONL data file at the corresponding line
4. Compute per-episode stats for each new episode
5. Update the manifest: replace the original entry with two new entries
6. The original directory remains on disk for prior snapshots

**Cost**: two partial video remuxes + two stats computations. Comparable to
truncation.

#### Episode Directory Naming

Since modified episodes produce new directories, episode directories should
use names that are both stable and unique. A scheme like
`ep_<sequential>_<short_hash>/` works well:

- The sequential prefix keeps directory listings human-readable
- The hash suffix (e.g., first 4 chars of a content hash or a random ID)
  guarantees uniqueness when an episode is modified

The manifest references directories by name, so the naming scheme is
transparent to all other components.

### 5.5. Garbage Collection

Copy-on-write means old episode directories accumulate on disk. They must
be retained as long as any snapshot references them, but can be deleted
once no manifest (current or snapshot) points to them.

A simple garbage collection pass:

1. Collect all episode directory names referenced by the current manifest
   and all snapshot manifests
2. List all episode directories on disk
3. Delete any directory not in the referenced set

This can run on a schedule, on user request, or after snapshot deletion.
The operation is safe because it only removes directories that no manifest
references.

### 5.6. Integration with Training

To support the "retrain on a previous dataset state" workflow, the training
job payload should accept an optional snapshot reference:

- **No snapshot specified**: train on the current live manifest (the common
  case for new training runs). A snapshot is automatically created before
  training begins for reproducibility.
- **Snapshot specified**: train on the frozen manifest from that snapshot.
  No new snapshot needed -- the referenced one is already immutable.

The snapshot itself is just a manifest file, so the training adapter or
converter reads it exactly the same way it reads the live manifest. No
special code paths are needed downstream.

The `Model` record should store a reference to the snapshot used, preserving
the full provenance chain: `Model → Snapshot → list of episode directories
(with their per-episode stats)`.

### 5.7. Comparison with Full-Copy Approach

| Aspect                           | Full copy (current)                                               | Manifest snapshots (proposed)               |
|----------------------------------|-------------------------------------------------------------------|---------------------------------------------|
| **Snapshot creation time**       | Seconds to minutes (copies all video data)                        | O(1) -- copy one JSON file                  |
| **Storage per snapshot**         | Full dataset size (GB)                                            | ~10 KB (manifest file)                      |
| **10 snapshots of 5 GB dataset** | ~50 GB                                                            | ~5 GB + ~100 KB of manifests                |
| **Snapshot restoration**         | Copy files back or retarget path                                  | Replace manifest, re-aggregate stats (~1ms) |
| **Episode truncation/split**     | Modify in place (breaks other snapshots) or copy everything first | Copy-on-write one episode directory         |
| **Garbage collection**           | Manual cleanup of model dirs                                      | Automated: delete unreferenced episode dirs |
| **Retraining on old snapshot**   | Not supported in current API                                      | Pass snapshot reference to training job     |
| **Provenance**                   | Model → Snapshot (opaque copy)                                    | Model → Snapshot → exact episode list       |
