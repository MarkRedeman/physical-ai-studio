---
slides:
  title: "Dataset Architecture Proposal"
---

# Dataset Architecture Proposal

### Rethinking how we store, manipulate, and train on teleoperation data

Notes:
This presentation summarizes the findings from dataset_exploration_report.md.
Goal: get alignment on the proposed architecture before implementation begins.

---

## Agenda

1. The Problem -- where LeRobot v3 creates friction
2. Proposed Architecture -- episode-isolated storage
3. Snapshots -- cheap, immutable dataset versions
4. Training Paths -- direct adapter vs. LeRobot conversion
5. Implementation Plan -- priorities and timeline
6. Open Questions

Notes:
We'll spend most of the time on sections 1-3. Section 4 is about the training
integration, and section 5 is the implementation roadmap. Section 6 is for
discussion.

---

## 1. The Problem

---

### What LeRobot v3 does well

- **Training-optimized**: pre-computed stats, consolidated files, fast sequential I/O
- **Standard format**: interop with HuggingFace Hub, community datasets
- **Proven**: used in production training pipelines today

Notes:
Credit where due -- LeRobot v3 is a good training format. The issue is that
we're also using it as the source-of-truth for acquisition and curation, where
it was never designed to excel.

---

### Where it creates friction

| Operation | Difficulty | Why |
|-----------|------------|-----|
| **Delete an episode** | Hard | Rewrite consolidated Parquet + video chunks, update all offsets, recompute stats |
| **Truncate an episode** | Hard | Re-encode video chunk, rewrite Parquet, update frame counts + timestamps, recompute stats |
| **Split a dataset** | Medium | Episodes share physical files -- must duplicate or rewrite |
| **Merge datasets** | Medium | Re-consolidate files, reconcile indices and tasks |
| **Review one episode** | Hard | Episode is spread across consolidated chunks -- no self-contained artifact |

Notes:
The key insight: LeRobot packs multiple episodes into shared physical files
(Parquet chunks, video chunks). This is great for sequential reads during
training, but it means you can never touch one episode without affecting others.

---

### The cost of "consolidated"

Episode 37 lives inside:

```
data/chunk-000/file-000.parquet    (also contains ep 0-99)
videos/cam_overview/chunk-000/file-000.mp4  (also contains ep 0-99)
videos/cam_gripper/chunk-000/file-000.mp4   (also contains ep 0-99)
```

**Deleting episode 37** requires rewriting all three files.

**Truncating episode 37** requires re-encoding the video chunk.

Notes:
This is the fundamental architectural mismatch. The consolidation that makes
training fast makes everything else expensive.

---

### Conversion is slow

Converting 100 episodes (30s each, 640x480, 2 cameras) into LeRobot v3:

| Component | % of time |
|-----------|-----------|
| AV1 video encoding | ~50% |
| PNG round-trip (add_frame API) | ~30% |
| Source video decoding | ~13% |
| Stats computation | ~5% |
| Parquet + metadata | ~2% |

**Total: ~90-100 minutes** for a modest dataset.

Every frame is decoded, written as PNG, read back from PNG, then re-encoded.

Notes:
The PNG round-trip is forced by LeRobot's add_frame() API -- it writes
every frame to disk as PNG, then reads them back during save_episode().
This accounts for ~30% of total conversion time. See converting_episodes.md
for detailed benchmarks.

---

### Summary: right tool, wrong job

LeRobot v3 is **optimized for training**.

We're using it as the **source of truth** for a platform where:
- Data acquisition
- Episode review and quality control
- Dataset curation (delete, truncate, split, merge)
- Iterative training

...are all first-class workflows.

Notes:
The proposal is not to replace LeRobot for training. It's to stop using it
as the primary storage format and instead derive it when needed.

---

## 2. Proposed Architecture

---

### Core principle

> **Store data in a manipulation-friendly, episode-isolated format.**
> **Convert to LeRobot v3 on demand for training.**

Each episode is a self-contained directory. The dataset is a manifest that
references episodes.

Notes:
This is the single most important slide. Everything else follows from this
principle.

---

### Episode-isolated storage layout

```
dataset/
  manifest.json                    # Dataset-level metadata
  snapshots/                       # Frozen manifests (see section 3)
    snap_2026-02-10_14-30.json
  episodes/
    ep_000_abc1/
      data.jsonl                   # Per-frame state/action timeseries
      cam_overview.mp4             # One video per camera, original codec
      cam_gripper.mp4
      stats.json                   # Per-episode normalization stats
    ep_001_def2/
      ...
  .cache/
    stats.json                     # Aggregated global stats (derived)
```

Notes:
Key points:
- Each episode directory is fully self-contained and independently deletable
- Videos are stored in their original codec -- no re-encoding on ingest
- Per-episode stats enable cheap global aggregation
- The .cache directory contains derived data that can be regenerated

---

### Manipulation becomes trivial

| Operation | Before (LeRobot v3) | After (episode-isolated) |
|-----------|---------------------|--------------------------|
| **Delete episode** | Rewrite chunk files (~3 min) | `rm -rf` episode dir + update manifest (~1 sec) |
| **Truncate episode** | Re-encode video chunk (~5 min) | New episode dir with trimmed video (~5 sec) |
| **Split dataset** | Duplicate consolidated files (~2 min) | New manifest referencing same episodes (~1 ms) |
| **Review one episode** | Seek into consolidated MP4 by offset | Open `ep_000/cam_overview.mp4` in any player |
| **Merge datasets** | Re-consolidate everything (~5 min) | Merge manifests, copy episode dirs (~seconds) |

Notes:
The "After" times assume the episode-isolated format. Delete is instant because
there's nothing to rewrite. Truncation uses ffmpeg stream copy (remux, not
re-encode) in most cases. Split is a manifest-only operation -- no data
duplication.

Glossary:
- remux = repackage video data without decoding/re-encoding. Changes container,
  preserves codec. Very fast.

-v-

### How stats stay cheap

Each episode stores its own normalization statistics on creation.

Global stats are **aggregated from per-episode stats** -- no raw data scan.

| Statistic | Aggregation |
|-----------|-------------|
| **mean** | Weighted average across episodes |
| **std** | Parallel variance (Welford's algorithm) |
| **min/max** | Element-wise across episodes |
| **count** | Sum |

**Adding an episode**: compute its stats (~200ms), re-aggregate (~1ms)

**Deleting an episode**: re-aggregate from remaining stats (~1ms)

Notes:
Welford's parallel variance algorithm combines per-group mean, variance,
and count into a global variance in one pass. This is a well-known technique
for distributed statistics computation.

The key benefit: no operation ever requires re-scanning all raw frames.
Stats update is always O(episodes), not O(frames).

---

## 3. Snapshots

---

### The snapshot problem

1. Record episodes -> dataset **D1**
2. Train model **A1** on D1
3. Curate: delete bad episodes, truncate others -> **D2**
4. Train model **A2** on D2
5. A2 is worse. Want to retrain on **original D1**

Step 5 requires recovering the exact dataset state that A1 was trained on.

Notes:
This is a real workflow. Every team doing iterative robot learning hits this.
Without snapshots, the only option is to keep manual backups or re-record.

---

### Current approach: full physical copy

Before each training run, `shutil.copytree` duplicates the entire dataset.

| Issue | Impact |
|-------|--------|
| **Full copy every time** | Several GB per snapshot |
| **Linear storage growth** | 10 training runs = 10 full copies |
| **Slow creation** | Seconds to minutes, blocking training |
| **Not user-facing** | Buried in model directories, no list/compare/reuse |
| **No retrain path** | `TrainJobPayload` has no `snapshot_id` field |

Notes:
The current snapshot mechanism (in snapshot_service.py) has the right semantics
-- each training run gets an immutable copy -- but the implementation is
expensive. And snapshots are only created implicitly; there's no way for a user
to manually snapshot a dataset.

---

### Proposed: manifest-based snapshots

**A snapshot is a frozen copy of the manifest, not a copy of the data.**

```
dataset/
  manifest.json                      # Current (mutable) state
  snapshots/
    snap_2026-02-10_14-30.json       # Frozen manifest (~10 KB)
    snap_2026-02-18_09-15.json       # Another frozen manifest
  episodes/
    ep_000_abc1/                     # Shared by both snapshots
    ep_001_def2/                     # Only in first snapshot
    ep_001_def2_trunc_080/           # Truncated version, in current
    ep_005_ghi3/                     # Only in current
```

**Snapshot creation**: copy `manifest.json` -> `snapshots/<name>.json` -- **O(1)**

**Storage per snapshot**: ~10 KB (one JSON file)

Notes:
Episode directories are the shared, immutable substrate. The manifest defines
which episodes constitute a particular dataset state. Freezing the manifest
freezes the dataset state. This is conceptually similar to how Git commits
reference trees of blobs.

-v-

### Copy-on-write for mutations

When truncating or splitting an episode:

1. Create a **new** episode directory with modified data
2. Update manifest to point to new directory
3. Old directory stays on disk (referenced by prior snapshots)

**Truncation example:**

```
ep_001_def2/            <- original (kept for snapshots)
ep_001_def2_trunc_080/  <- new, truncated (in current manifest)
```

Cost: one video remux + one stats recomputation. Seconds, not minutes.

Notes:
Copy-on-write (COW) means we never modify an episode directory in place.
Any mutation creates a new directory. This guarantees that snapshots always
point to valid, unchanged data.

The "one video remux" uses ffmpeg stream copy -- it repackages the video
without decoding/re-encoding. Only the GOP at the cut point may need
re-encoding (a Group of Pictures is the smallest independently decodable
unit in a video stream, typically 0.5-2 seconds).

-v-

### Garbage collection

Old episode directories accumulate. Clean up when safe:

1. Collect all episode dirs referenced by current manifest + all snapshots
2. List all episode dirs on disk
3. Delete any directory not in the referenced set

**Safe**: only removes directories no manifest points to.

Can run on schedule, on user request, or after snapshot deletion.

Notes:
This is simple reference counting. An episode directory is "alive" if any
manifest (current or snapshot) references it. Once no manifest references it,
it can be safely deleted. This is analogous to garbage collection in
managed runtimes.

-v-

### Comparison

| Aspect | Full copy (current) | Manifest snapshots |
|--------|--------------------|--------------------|
| **Creation time** | Seconds to minutes | O(1) -- copy one JSON file |
| **Storage per snapshot** | Full dataset size (GB) | ~10 KB |
| **10 snapshots of 5 GB** | ~50 GB | ~5 GB + ~100 KB |
| **Restoration** | Copy files back | Replace manifest (~1ms) |
| **Retrain on old snapshot** | Not supported | Pass snapshot ref to training job |
| **Provenance** | Model -> opaque copy | Model -> snapshot -> exact episode list |

Notes:
The storage savings alone justify this change. But the bigger win is the
provenance chain: every model links to a snapshot that lists the exact episodes
it was trained on. This is full reproducibility with zero overhead.

---

## 4. Training Paths

---

### Two paths to training

```
                   Episode-Isolated Storage
                           |
              +------------+------------+
              |                         |
         Path A: Direct             Path B: Convert
         Training Adapter           to LeRobot v3
              |                         |
         Lazy video decode         Cached, incremental
         JSONL in memory           conversion
         Stats from cache          |
              |                    .cache/lerobot/
              v                         |
         DataLoader                     v
                                   DataLoader
```

**Path A** is the target default -- no conversion overhead.

**Path B** is maintained for ecosystem interop (HuggingFace Hub, LeRobot training scripts).

Notes:
Path A reads directly from the episode-isolated format. Video frames are
decoded lazily in __getitem__. Scalar data (state, action) is loaded into
memory from JSONL at init time. Stats come from the cached global aggregation.

Path B converts to LeRobot v3 but with key optimizations: bypass the PNG
round-trip by writing Parquet directly and remuxing video; cache the output
so re-conversion after small changes is near-instant.

---

### Conversion optimization levels

| Level | Strategy | Speed |
|-------|----------|-------|
| **1. Naive** | Use LeRobot's `add_frame()` API | ~1 min/episode |
| **2. Bypass PNG** | Write Parquet directly, remux video with ffmpeg | ~10 sec/episode |
| **3. Cached incremental** | Only convert changed episodes, reuse cache | ~instant (common case) |

For 100 episodes:
- Level 1: ~100 minutes
- Level 2: ~17 minutes
- Level 3: seconds (if only a few episodes changed)

Notes:
Level 2 eliminates ~55% of conversion time by avoiding the PNG decode-write-read
round-trip. Level 3 makes the common case (re-training after small changes)
nearly free by only converting the delta.

---

## 5. Implementation Plan

---

### Priority order

| # | Component | Time Est. | Impact |
|---|-----------|-----------|--------|
| 1 | Manifest schema + episode models | ~2 days | Foundation for everything |
| 2 | Per-episode stats + global aggregation | ~2 days | O(1ms) stats after any mutation |
| 3 | Frame index + video decoding | ~2 days | Required by adapter and review |
| 4 | Training adapter (Path A) | ~1 week | Training without conversion |
| 5 | Bidirectional converters (Path B) | ~1 week | LeRobot ecosystem interop |
| 6 | Episode manipulation helpers | ~2 days | Unlocks curation workflows |
| 7 | Optimized conversion (bypass PNG) | ~1 week | ~10x faster Path B |
| 8 | Conversion cache + incremental | ~1 week | Near-instant re-conversion |

**Total estimate: ~5-6 weeks** for all components.

Items 1-4 are the critical path to a working system.

Notes:
These are rough estimates. Items 1-3 are low-risk and well-understood.
Item 4 (training adapter) needs profiling -- lazy video decode throughput
is the biggest unknown. If it's too slow for GPU utilization, we may need
a pre-decode cache or prefetch strategy.

Items 5-8 can be parallelized and are lower priority.

---

## 6. Open Questions

---

### For discussion

1. **Migration path**: How do we transition existing LeRobot datasets?
   - Auto-convert on first access? Dual-format period? Manual migration tool?

2. **Lazy decode performance**: Is frame-by-frame video decode fast enough for training?
   - Need benchmarks: lazy decode vs. pre-decoded LeRobot, GPU utilization impact

3. **Concurrent access**: What locking strategy for manifest.json?
   - Recording + training + UI all touching the same dataset simultaneously

4. **Multi-rate sensors**: Record at native rates, align at training time?
   - Valuable but a separate proposal (see appendix in full document)

5. **Long-term Path B**: Can we eventually deprecate the LeRobot conversion path?
   - Or is HuggingFace Hub interop a permanent requirement?

Notes:
These are the key decisions that need team input before implementation.
The migration path (#1) is the most critical -- it determines how disruptive
the transition is. The lazy decode question (#2) needs empirical data;
we should profile before committing to Path A as the default.

Multi-rate (#4) is deliberately scoped out of this proposal per review
feedback. It's a separate, valuable initiative but would dilute this
proposal's focus.

---

### Error recovery considerations

| Failure mode | Strategy |
|--------------|----------|
| JSONL crash mid-write | Validate last line on load; discard if malformed |
| Manifest corruption | Keep previous version as `.manifest.json.bak` |
| Disk full during COW | Detect, clean up partial new dir, leave original intact |
| Interrupted recording | Episode dir exists but is incomplete; mark in manifest |

Notes:
These came out of the review process. They don't need to be solved in the
architecture document but should be addressed during implementation.

---

## Next Steps

1. **Align on architecture** (this meeting)
2. **Profile lazy video decode** -- determine if Path A is viable as default
3. **Define migration strategy** for existing datasets
4. **Begin implementation** -- items 1-4 from priority table (~3 weeks)

---

# Questions?

Full document: `dataset_exploration_report.md`

Review synthesis: `review_report.md`

Benchmarks: `converting_episodes.md`

Notes:
The full architecture document is 790 lines with detailed rationale for every
design decision. The review report synthesizes feedback from two reviewer
personas (codebase veteran and system architect). The converting_episodes
document has the original benchmarks for LeRobot conversion performance.
