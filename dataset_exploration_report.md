# Dataset Architecture Exploration Report

This document evaluates LeRobot's v3.0 dataset format from the perspective of a
platform that needs to **acquire**, **review/manipulate**, and **train on**
teleoperation data. It identifies where the current format creates friction for
non-training workflows and proposes an architecture that addresses these gaps,
building on the raw-video dataset module already implemented in
`application/backend/src/internal_datasets/raw_video/`.

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

| Operation | Difficulty | Why |
|-----------|-----------|-----|
| **Delete an episode** | Hard | Must rewrite the consolidated Parquet + video chunk files, update all episode offset metadata, recompute global stats |
| **Truncate an episode** (remove bad frames) | Hard | Must re-encode the video chunk, rewrite the Parquet chunk, update frame counts and timestamp offsets, recompute stats |
| **Split a dataset** (train/val by episode) | Medium | Requires duplicating or rewriting consolidated files since episodes share physical files |
| **Merge datasets** from different sessions | Medium | Must re-consolidate Parquet and video files, reconcile episode indices and task tables |
| **Reorder episodes** | Hard | Consolidated files encode ordering implicitly via offsets; reordering means rewriting everything |

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

| Component | % of time | Notes |
|-----------|----------|-------|
| AV1 video encoding (SVT-AV1) | ~50% | Re-encodes every frame from PNG |
| PNG writes + reads (round-trip) | ~30% | Forced by `add_frame()` API |
| Source video decoding | ~13% | Our code, unavoidable |
| Stats computation | ~5% | Subsampled image scan + trivial numerical scan |
| Parquet + metadata | ~2% | Fast |

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
├── episode_000/
│   ├── data.jsonl                 # Per-frame state/action timeseries
│   ├── cam_overview.mp4           # One video per camera, original codec
│   ├── cam_gripper.mp4
│   └── stats.json                 # Per-episode normalization stats
├── episode_001/
│   └── ...
└── .cache/
    └── stats.json                 # Aggregated global stats (derived)
```

Each episode directory is fully self-contained. The manifest tracks dataset-level
metadata (FPS, joint dimensions, joint names, robot type, camera list) and an
ordered list of episode references.

**Manipulation becomes trivial:**

| Operation | How |
|-----------|-----|
| Delete an episode | `rm -rf episode_NNN/`, remove from manifest |
| Truncate an episode | Re-encode one small video, truncate one JSONL file, recompute one episode's stats |
| Split dataset | Copy episode directories to separate roots, write new manifests |
| Merge datasets | Copy episode directories together, merge manifests |
| Review an episode | Open `episode_NNN/cam_overview.mp4` in any media player |
| Tag/annotate | Add fields to manifest's episode entry or a per-episode metadata file |

### 3.3. Per-Episode Stats for Cheap Global Aggregation

Each episode stores its own normalization statistics immediately after recording
(or after any modification). These per-episode stats include:

- `mean`, `std`, `min`, `max` for state and action vectors (element-wise)
- `mean`, `std`, `min`, `max` for image features (per-channel)
- `count` (number of frames in the episode)

Global dataset stats are then **aggregated from per-episode stats in O(episodes)
time** without touching any raw data:

| Statistic | Aggregation method |
|-----------|-------------------|
| **mean** | Weighted average: `global_mean = sum(ep_mean * ep_count) / sum(ep_count)` |
| **std** | Parallel variance algorithm (Welford-style): combines per-episode mean, variance, and count |
| **min / max** | Element-wise min/max across all per-episode values |
| **count** | Sum of all per-episode counts |

This means:
- Adding a new episode: compute its stats (~200ms for numerical, ~2s for
  subsampled images), then re-aggregate (~1ms)
- Deleting an episode: re-aggregate from remaining episodes' stats (~1ms)
- No operation ever requires re-scanning all raw frames

### 3.4. On-Demand Conversion to LeRobot v3.0

When training is requested, the platform converts the selected episodes into
a LeRobot v3.0 dataset. Three levels of optimization are possible:

#### Level 1: Current approach (functional, slow)

Use `RawVideoToLeRobotConverter` as-is, going through LeRobot's
`add_frame()` / `save_episode()` / `finalize()` API. This works correctly
but is slow (~1 minute per episode) due to the PNG round-trip and AV1
re-encoding.

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

## 4. Proposal: Building on the Existing Raw-Video Module

The `internal_datasets/raw_video/` package already implements most of what is
needed for the episode-isolated primary storage layer. Here is a concrete
mapping of existing components to the proposed architecture, and what would
need to be added.

### 4.1. What Already Exists

| Component | File | Role in proposed architecture |
|-----------|------|-------------------------------|
| `DatasetManifest` | `manifest.py` | Dataset-level metadata schema. Already tracks FPS, dimensions, joint names, robot type, camera configs, episode list. Serves as the manifest for primary storage. |
| `EpisodeEntry` | `manifest.py` | Per-episode reference (directory, data file, video files). Already episode-isolated. |
| `CameraConfig` | `manifest.py` | Camera stream metadata. Reusable as-is. |
| `load_manifest()` | `manifest.py` | Manifest loading with on-disk validation. Reusable as-is. |
| `FrameIndex` | `frame_index.py` | O(1) global-to-episode frame lookup. Used by the adapter for training; would also support review/playback UIs. |
| `WelfordAccumulator` | `stats.py` | Numerically-stable online mean/std/min/max. Core building block for per-episode stats computation. |
| `DatasetStats` | `stats.py` | Aggregated stats container with JSON serialization. Serves as both per-episode and global stats format. |
| `compute_stats()` | `stats.py` | Full-dataset stats computation (single-pass numerical + subsampled images). Currently computes global stats in one shot. |
| `load_or_compute_stats()` | `stats.py` | Cache-aware stats loading with staleness detection via mtime. Already caches to `.cache/stats.json`. |
| `RawVideoDatasetAdapter` | `adapter.py` | Training adapter implementing `getiaction.data.Dataset`. Loads scalars into memory, decodes video lazily, provides normalization parameters from cached stats. **This is the direct-training path** -- no LeRobot conversion needed. |
| `decode_frame()` / `decode_frames()` | `video_decode.py` | Video frame decoding with torchcodec/pyav backends. Used by both the adapter and the converter. |
| `LeRobotToRawVideoConverter` | `converters.py` | Converts LeRobot v3 → raw-video format. Enables importing existing LeRobot datasets into the episode-isolated format. |
| `RawVideoToLeRobotConverter` | `converters.py` | Converts raw-video → LeRobot v3 format. The "compile for training" path when LeRobot format is required. |

### 4.2. What Would Need to Be Added

#### A. Per-episode stats computation and storage

Currently `compute_stats()` does a single pass over the entire dataset. This
should be refactored into two layers:

1. **`compute_episode_stats(episode, dataset_root)`** -- computes stats for a
   single episode. Uses `WelfordAccumulator` for numerical features, subsampled
   video decoding for image features. Writes result to
   `episode_NNN/stats.json`.

2. **`aggregate_episode_stats(episode_stats_list)`** -- combines a list of
   per-episode `DatasetStats` into a single global `DatasetStats` using the
   parallel variance algorithm. This is O(episodes) and touches no raw data.

The existing `compute_stats()` and `load_or_compute_stats()` would be updated
to use these building blocks internally. The `WelfordAccumulator` already
supports the online merge pattern needed for aggregation.

#### B. Episode manipulation operations

A small set of manifest-level operations:

- **`remove_episode(manifest, episode_idx)`** -- removes an episode entry from
  the manifest. Does not touch files (caller handles deletion). Re-aggregates
  global stats from remaining per-episode stats.
- **`add_episode(manifest, episode_entry)`** -- appends a new episode entry,
  computes its per-episode stats, re-aggregates global stats.
- **`subset_manifest(manifest, episode_indices)`** -- creates a new manifest
  containing only the selected episodes. Useful for train/val splits.

These are purely metadata operations. The episode directories and their files
remain untouched.

#### C. Optimized LeRobot conversion (Level 2)

A faster conversion path that bypasses LeRobot's `add_frame()` API:

1. Write Parquet data files directly using PyArrow, reading from JSONL
2. Remux episode videos into consolidated chunk files using ffmpeg's concat
   demuxer (lossless, no re-encoding)
3. Aggregate global stats from per-episode stats (already computed)
4. Write `meta/info.json`, `meta/stats.json`, `meta/tasks.parquet`, and
   episode metadata Parquet files following the v3.0 schema

This eliminates the PNG round-trip entirely and avoids video re-encoding
when the source codec is compatible, reducing conversion time from ~1 minute
per episode to ~5-10 seconds per episode.

#### D. Conversion cache with incremental updates (Level 3)

A cache manager that:

1. Maintains a `.cache/lerobot/` directory alongside the primary dataset
2. Tracks which episodes have been converted (by content hash or mtime)
3. On training request: converts only new/modified episodes, appends to
   existing cached LeRobot output, re-aggregates stats
4. Provides a `get_or_build_lerobot_path(manifest)` API that returns a
   ready-to-use LeRobot dataset path

### 4.3. Proposed Architecture Diagram

```
                        PRIMARY STORAGE (episode-isolated)
                        ──────────────────────────────────
                        manifest.json
                        episode_000/  ← self-contained, deletable
                          data.jsonl
                          cam_overview.mp4
                          cam_gripper.mp4
                          stats.json  ← per-episode stats [NEW]
                        episode_001/
                          ...

                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        ┌───────────┐  ┌───────────┐  ┌──────────────┐
        │  REVIEW   │  │ MANIPULATE│  │    TRAIN     │
        │           │  │           │  │              │
        │ Play .mp4 │  │ rm -rf ep │  │ Path A:      │
        │ Thumbnail │  │ Truncate  │  │ Direct via   │
        │ Annotate  │  │ Split     │  │ Adapter      │
        │ Score     │  │ Merge     │  │ (no convert) │
        └───────────┘  └─────┬─────┘  │              │
                             │        │ Path B:      │
                             ▼        │ Convert to   │
                    Re-aggregate      │ LeRobot v3   │
                    global stats      │ (cached,     │
                    from per-ep       │  incremental) │
                    stats (~1ms)      └──────┬───────┘
                                             │
                                             ▼
                                    .cache/lerobot/
                                    (derived, disposable)
```

**Training path A** (`RawVideoDatasetAdapter`) is already fully functional.
It reads directly from the episode-isolated format, decodes video on demand,
and provides normalization stats from the cached global stats file. This
requires no conversion at all.

**Training path B** (`RawVideoToLeRobotConverter` → LeRobot DataLoader) is
available for workflows that specifically need the LeRobot v3.0 format (e.g.,
using LeRobot's own training scripts or uploading to HuggingFace Hub). The
proposed optimizations (Levels 2 and 3) would make this path significantly
faster.

### 4.4. Priority Order

| Priority | Change | Effort | Impact |
|----------|--------|--------|--------|
| 1 | Add per-episode `stats.json` and refactor `compute_stats()` into per-episode + aggregate | Low | Enables O(1ms) re-aggregation after any dataset mutation |
| 2 | Add episode manipulation helpers (remove, add, subset) | Low | Unlocks review/curation workflows |
| 3 | Optimized LeRobot conversion (bypass PNG round-trip) | Medium | ~10x faster conversion for path B |
| 4 | Conversion cache with incremental updates | Medium | Near-instant re-conversion for the common case |
