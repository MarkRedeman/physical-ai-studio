# Converting Episodes: Performance Analysis

This document describes the performance characteristics of the raw-video to LeRobot v3
converter (`RawVideoToLeRobotConverter`), where time is spent, how it scales, and
what the fundamental bottlenecks are.

---

## 1. Conversion Pipeline Overview

The converter reads a raw-video dataset (manifest + per-episode JSONL + MP4 files) and
writes a LeRobot v3 dataset using the `LeRobotDataset.create()` / `add_frame()` /
`save_episode()` / `finalize()` API.

The per-episode pipeline looks like this:

```
Source MP4 (per camera)
  → decode ALL frames to numpy         [our code, video_decode.py]
  → add_frame() x N                    [LeRobot API]
       writes each frame as a PNG to a temp directory
       (async via 4 image-writer threads, compression level 1)
       buffers scalar data (state, action, timestamps) in memory
  → save_episode()                     [LeRobot API]
       waits for async PNG writes to drain
       computes per-episode stats (subsampled image scan + full numerical scan)
       writes frame data to Parquet (incremental writer, Snappy compression)
       re-reads PNGs from disk, encodes to AV1 MP4 via SVT-AV1 (PyAV)
       concatenates episode MP4 onto chunk video file (lossless remux)
       aggregates per-episode stats into running global stats
       writes metadata parquet + info.json + stats.json
       deletes temporary PNGs
  → finalize()                         [LeRobot API, lightweight]
       closes Parquet writers (flushes file footers)
```

The critical observation is the **PNG round-trip**: every frame is decoded from our
source video, written as a PNG to disk by `add_frame()`, then read back from disk
during `save_episode()` for both stats computation and video encoding. This is
inherent to LeRobot's API design (built for real-time teleoperation where frames
arrive one at a time, not batch conversion).

---

## 2. Where Time Is Spent

Measured baseline: 2 episodes, ~30 seconds each (900 frames/episode), 640x480
resolution, 2 cameras, ~2 minutes total wall time.

### Per-Episode Breakdown

| Step | What happens | Est. time (30s episode) | Scales with |
|------|-------------|------------------------|-------------|
| Video decode | Decode all frames from source MP4 via torchcodec/pyav into `(N, H, W, 3)` uint8 array | ~8s | Frames per episode |
| `add_frame()` x N | Validate frame, buffer scalars in memory, write PNG to disk (async, 4 threads) | ~15s | Frames x cameras |
| Stats (numerical) | Compute mean/std/min/max/quantiles over state and action vectors for the episode | <0.1s | Frames (trivial) |
| Stats (image) | Load ~`N^0.75` subsampled PNGs from disk, downsample to ~150px, compute per-channel stats | ~3s | `N^0.75` (sublinear) |
| Parquet write | Write frame data to parquet | <1s | Frames |
| **Video encoding** | Read all PNGs, encode to AV1 via SVT-AV1 (CRF 30, preset 12). Parallel across cameras. | **~30s** | Frames x cameras |
| Video concatenation | Lossless remux to append episode to chunk file | <1s | Constant |
| Stats aggregation | Weighted mean/variance merge into global stats | <1ms | Constant |

### Proportion of Total Time (2 episodes, ~120s)

| Component | Est. time | % of total |
|-----------|----------|------------|
| SVT-AV1 video encoding | ~60s | **~50%** |
| PNG writes (`add_frame`) | ~30s | ~25% |
| Video decode (our code) | ~16s | ~13% |
| PNG reads (for encoding + stats) | ~8s | ~7% |
| Stats computation | ~6s | ~5% |

**Video encoding dominates at ~50% of wall time.** The PNG write/read round-trip is
the second biggest cost at ~30%.

---

## 3. Scaling to Larger Datasets

Everything scales **linearly with total frame count**, not with episode count. The
only sublinear component is image stats sampling (`N^0.75` per episode).

### Projected Times

| Component | 2 episodes (~1,800 frames) | 100 episodes (~90,000 frames) | Scale |
|-----------|---------------------------|-------------------------------|-------|
| Video decode | ~16s | ~800s (~13 min) | 50x |
| PNG writes | ~30s | ~1,500s (~25 min) | 50x |
| SVT-AV1 encoding | ~60s | ~3,000s (~50 min) | 50x |
| Stats (image sampling) | ~6s | ~150s (~2.5 min) | ~30x |
| Stats (numerical) | negligible | negligible | 50x |
| Stats aggregation | negligible | negligible | linear, tiny |
| Parquet + metadata | ~2s | ~100s (~1.5 min) | 50x |
| **Total** | **~2 min** | **~90-100 min (~1.5 hr)** | **~50x** |

The linear scaling means doubling frames roughly doubles conversion time. There are
no super-linear surprises.

---

## 4. Stats Computation Is Not a Bottleneck

A common concern is that computing statistics over the whole dataset might be
expensive. In practice, stats are ~5% of total conversion time because:

- **Numerical features** (state/action vectors): Computing mean/std/min/max/quantiles
  over a few thousand 6-element float32 vectors is trivial (microseconds).

- **Image features**: LeRobot uses aggressive subsampling. Per episode, it loads only
  ~`N^0.75` frames (e.g., ~250 of 900), downsamples them to ~150px on the larger
  axis, and computes per-channel (R, G, B) statistics. This keeps image stats
  computation fast even for long episodes.

- **Global aggregation**: Happens incrementally via the parallel variance algorithm
  (weighted mean + delta-means-squared for variance). This is O(1) per episode --
  it merges two sets of summary statistics without re-reading any data. For
  min/max, it's an element-wise comparison. For quantiles (q01, q99, etc.), it
  uses a weighted average of per-episode quantiles (approximate, not exact).

The two-tier design (per-episode stats computed in `save_episode()`, global stats
aggregated incrementally) means adding a new episode never requires re-scanning
existing data.

---

## 5. The Fundamental Bottleneck: The PNG Round-Trip

The dominant cost is the PNG round-trip forced by LeRobot's `add_frame()` API:

```
Our raw MP4 → decode to numpy → add_frame() writes PNG to disk
                                                  ↓
         encode to AV1 MP4 ← save_episode() reads PNGs back from disk
```

For every single frame, the pipeline:

1. Decodes from source video (fast, ~5ms/frame)
2. Writes as PNG to disk (~15-20ms/frame, even at compression level 1)
3. Reads PNG back from disk (~10ms/frame, for stats + encoding)
4. Encodes to AV1 (~30ms/frame with SVT-AV1)

Steps 2 and 3 are pure waste -- we already have the pixels in memory. But this is
how LeRobot's API works: `add_frame()` persists each frame as an image file, and
`save_episode()` reads them all back to encode video. The API was designed for
real-time teleoperation where frames arrive one at a time over the lifetime of a
recording session, not for batch conversion of existing video.

---

## 6. Potential Optimizations (Not Implemented)

If conversion time becomes a problem at scale, these are the highest-impact
improvements, ordered by estimated benefit:

### 6.1. Bypass the PNG round-trip (~55% time savings)

Instead of going through `add_frame()` → PNG → `save_episode()` → encode, call
LeRobot's video encoder directly with in-memory frames and write Parquet data
ourselves. This would eliminate the entire PNG write/read cycle.

**Trade-off**: Requires using LeRobot internals (`encode_video_frames`) or writing
our own Parquet + video pipeline. Couples us to LeRobot's internal format details
rather than its public API.

### 6.2. Use H.264 instead of AV1 (~2-3x encoding speedup)

SVT-AV1 at preset 12 is roughly 2-3x slower than libx264 at preset "fast" for
comparable visual quality. LeRobot defaults to AV1 for better compression ratios,
but for local datasets where disk space isn't critical, H.264 would be significantly
faster.

**Trade-off**: Larger files (~30-50% bigger at similar quality). Would require
overriding LeRobot's default encoder settings.

### 6.3. Parallel episode processing

Episodes are independent -- multiple episodes could be decoded and converted
concurrently using a process pool. Currently episodes are processed sequentially.

**Trade-off**: Higher peak memory usage (each episode's decoded frames live in
memory simultaneously). For 640x480 @ 30fps @ 30s, each episode's frames are
~900 * 640 * 480 * 3 ≈ 830 MB per camera. Parallelizing 4 episodes with 2 cameras
would need ~6.5 GB just for frame buffers.

### 6.4. Stream-copy video segments instead of re-encoding

If the source raw-video files are already in a compatible codec (AV1 or H.264),
we could potentially stream-copy the video data into the LeRobot chunk files
without decoding and re-encoding. This would reduce the video pipeline from
decode + PNG + encode (~50ms/frame) to a simple byte copy (<1ms/frame).

**Trade-off**: Only works when the source codec matches the target. Requires
bypassing LeRobot's `add_frame()` API entirely and managing video chunk files
manually. Also requires careful handling of keyframe boundaries and timestamp
remapping for the consolidated video format.
