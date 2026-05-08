# Performance Telemetry Plan

This document captures the implementation plan for integrating topswatch-like
telemetry into the FastAPI backend in standalone Python.

## Goals

- Provide resource telemetry over time for CPU, GPU, NPU, RAM.
- Prioritize Intel hardware metrics (CPU, Intel GPU, Intel NPU).
- Keep NVIDIA support as a first-class optional path.
- Persist telemetry in SQLite for history and charting.
- Expose API endpoints for latest, history, ranges, devices, and streaming.

## Phase 1 — Data model and configuration

1. Add telemetry configuration values in `settings.py`:
   - enable/disable collection
   - collection interval
   - retention
   - batch size and persistence toggles
2. Add DB schema tables:
   - `telemetry_series` (metric identity + labels)
   - `telemetry_points` (timestamped numeric values)
   - optional rollup table(s)
3. Add Alembic migration for new tables and indexes.

## Phase 2 — Collector modules (Python)

1. Implement collector interface and metric model.
2. Implement collectors:
   - CPU (`/proc/stat`, cpufreq, meminfo)
   - Intel GPU sysfs metrics (freq/temps/power where available)
   - NVIDIA via NVML (if installed and devices available)
3. Ensure collectors degrade gracefully when unavailable.

## Phase 3 — Background telemetry service

1. Add `TelemetryService` with lifecycle-managed async loop.
2. Sample at fixed interval and keep latest in memory.
3. Persist to SQLite efficiently (series cache + batched inserts).
4. Add retention pruning for old points.

## Phase 4 — API integration

1. Add `/api/system/telemetry/*` endpoints:
   - latest
   - history
   - ranges
   - devices
   - stream (SSE)
2. Integrate service dependency via FastAPI dependency injection.
3. Keep `/api/hardware` scoped to camera/robot peripherals.

## Phase 5 — Intel GPU utilization parity (perf PMU)

1. Add optional Intel GPU PMU path for better utilization parity.
2. Use Linux `perf_event_open` pathway where permissions allow.
3. Keep robust fallback to sysfs-level utilization when PMU unavailable.

## Phase 6 — Intel NPU PMT metrics

1. Add optional PMT reader for Intel NPU telemetry (`/sys/class/intel_pmt`).
2. Implement generation-aware register decode (MTL/ARL/LNL/PTL mappings).
3. Compute derived rates (power, bandwidth) from cumulative counters.

## Phase 7 — rollups, query efficiency, UI-readiness

1. Add downsampled ranges (`5min`, `1h`, `24h`) query paths.
2. Use SQL aggregation for bucketed history reads.
3. Keep payload stable for frontend charting.
4. Add operational documentation and tuning knobs.

## Incremental delivery and commits

- Implement each phase with incremental commits.
- Prefer stable partial delivery over all-at-once changes.
- Validate with static diagnostics and runtime sanity checks.
