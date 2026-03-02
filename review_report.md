# Review Synthesis Report

Review of `dataset_exploration_report.md` (790 lines, 6 sections) by two
personas representing different stakeholder perspectives. This report
synthesizes their feedback, identifies common themes, and proposes a
prioritized improvement plan.

---

## 1. Reviewers and Their Perspectives

| Reviewer       | Persona          | Lens                                                                                                                                                                                            |
|----------------|------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Reviewer 1** | Codebase Veteran | Deep knowledge of the existing codebase (`DatasetClient`, `RecordingMutation`, `SnapshotDB`, training worker). Skeptical of rewrites; asks "how does this integrate with what we already have?" |
| **Reviewer 2** | System Architect | Understands project goals, user workflows, and system-level trade-offs. No line-level code familiarity. Asks "does this tell a clear story and cover the operational realities?"                |

**Important context**: Reviewer 1's original feedback referenced the
`raw_video/` package as if it were production code. It is a
**prototype/reference implementation** built during design exploration and
has **not been upstreamed**. The feedback below is adjusted for this
clarification -- concerns about "shipping the prototype" are dropped, but
the underlying integration questions remain valid.

---

## 2. Reviewer 1 (Codebase Veteran) -- Summary

### Strongest Points Raised

1. **Migration path is the document's biggest gap.** The proposal describes
   a new episode-isolated format but does not address:
   - What happens to existing LeRobot datasets already on disk?
   - How does the `DatasetClient` ABC adapt -- does it get a new
     implementation, or is the interface itself changed?
   - Does `RecordingMutation` / `DeleteEpisodesMutation` survive, get
     replaced, or get wrapped?
   - What about existing `SnapshotDB` records and model directories that
     contain `shutil.copytree` snapshots?

2. **Snapshot copy-on-write may be over-engineered.** Counter-proposal:
   use **hardlinks** for video files instead of copying entire episode
   directories. On Linux/macOS, hardlinks are O(1) and zero-copy at the
   filesystem level. This avoids the complexity of garbage collection and
   the "old directories accumulate" problem.

3. **Manifest concurrency is unaddressed.** The platform has concurrent
   processes: recording worker, training worker, UI/API serving curation
   requests. If all read/write `manifest.json`, race conditions are
   inevitable without a locking or transactional strategy.

4. **JSONL crash recovery.** If the recording process crashes mid-write,
   the last line of a JSONL file may be truncated. The document should
   address detection and recovery (e.g., validate last line on load,
   discard if malformed).

5. **Lazy video decode during training needs profiling.** The document
   presents "direct training via adapter (no conversion)" as the happy
   path, but frame-by-frame video decoding in `__getitem__` may be too
   slow for real training throughput. Needs benchmarks comparing:
   - Lazy decode from per-episode MP4 vs. pre-decoded LeRobot Parquet
   - Impact on GPU utilization (are workers starved?)
   - torchcodec vs. pyav performance characteristics

6. **Missing benchmarks throughout.** Claims like "O(1) snapshot creation"
   and "~1ms re-aggregation" are plausible but unsubstantiated. The
   document would be stronger with measured numbers.

### Assessment

Reviewer 1's feedback is predominantly about **integration realism** --
the gap between "here's a clean-sheet design" and "here's how we get
there from where we are today." The most critical gap is the migration
plan.

---

## 3. Reviewer 2 (System Architect) -- Summary

### Strongest Points Raised

1. **Section 6 (multi-rate) should be separated.** It constitutes ~30% of
   the document and introduces a different problem domain (recording
   architecture) that dilutes the core message (storage format for
   manipulation + snapshots). Recommend extracting it into its own proposal
   or appendix.

2. **Needs concrete user workflow walkthroughs.** The friction table (2.1)
   is effective, but the proposed solutions lack corresponding before/after
   narratives with measurable improvements. Example:
   - "Delete episode: **before** = rewrite 200 MB Parquet chunk + 500 MB
     video chunk + recompute stats (~3 min). **After** = remove manifest
     entry, `rm -rf` episode dir (~1 sec)."

3. **Concurrent access scenarios.** What happens when:
   - A user deletes an episode while a training run is reading from it?
   - Two users curate the same dataset simultaneously?
   - A recording is in progress while another episode is being deleted?

4. **Migration plan for existing datasets.** Same concern as Reviewer 1
   but from a user-facing angle: will existing datasets be auto-migrated?
   Is there a dual-format transition period? What's the user experience?

5. **Priority table needs time estimates.** "Low/Medium" effort is not
   actionable for planning. Rough time ranges (e.g., "1-2 days", "1 week")
   would help stakeholders understand the commitment.

6. **Error recovery discussion needed.** Interrupted recordings, corrupted
   manifests, partial stats, disk-full during copy-on-write -- the document
   should at least acknowledge these failure modes and sketch recovery
   strategies.

7. **Disk space management.** Snapshot storage is cheap per-manifest, but
   the episode directories they keep alive can accumulate significantly.
   Needs a user-facing strategy: storage visibility, cache eviction
   policies, disk-full handling.

8. **Lead with user benefit in Section 3.3.** The Welford formulas are
   correct but intimidating for non-ML readers. Lead with "stats update
   in 1ms instead of 10 minutes" and move the math to an appendix.

9. **Define jargon on first use.** Terms like "remux", "GOP", "Welford",
   "delta_indices" appear without definition.

10. **Clarify long-term intent.** Is Path A (direct adapter) the primary
    long-term path? Can Path B (LeRobot conversion) eventually be
    deprecated? The document is ambiguous about this.

### Positive Highlights

- Friction table (2.1) is the document's strongest section
- Architecture diagram (4.2) is clear and effective
- Snapshot comparison table (5.7) makes the case compellingly

### Assessment

Reviewer 2's feedback is about **communication clarity and operational
completeness**. The technical design is sound, but the document needs to
tell a better story (before/after workflows, jargon glossary) and address
real-world operational concerns (concurrency, errors, disk space).

---

## 4. Common Themes

Both reviewers independently raised these issues:

| Theme                                    | R1                                                                  | R2                                                         | Severity     |
|------------------------------------------|---------------------------------------------------------------------|------------------------------------------------------------|--------------|
| **Migration path for existing datasets** | Primary concern -- DatasetClient, mutations, SnapshotDB integration | Primary concern -- user experience, dual-format transition | **Critical** |
| **Concurrent access / locking**          | manifest.json race conditions                                       | Multi-user curation, record-while-train                    | **High**     |
| **Section 6 scope**                      | Implicit (focused review time on other sections)                    | Explicit recommendation to extract                         | **Medium**   |
| **Error recovery**                       | JSONL crash recovery specifically                                   | Broader: corrupted manifests, disk-full, partial writes    | **Medium**   |
| **Benchmarks / evidence**                | Lazy decode throughput, snapshot timing                             | Before/after metrics for all operations                    | **Medium**   |

Neither reviewer challenged the **core architecture** (episode-isolated
storage with manifest-based snapshots). The design is accepted as sound;
the gaps are about integration path, communication, and operational edge
cases.

---

## 5. Prioritized Improvement Plan

Changes to `dataset_exploration_report.md` ranked by impact. Each item
is tagged with a disposition:

- **Accept** -- incorporate into the document
- **Defer** -- valid but belongs in a follow-up document or implementation spec
- **Reject** -- disagreed with, explanation provided

### P0: Critical (must address before sharing)

| # | Change                                                                                                                                                                            | Source | Disposition | Notes                                                                                       |
|---|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------|-------------|---------------------------------------------------------------------------------------------|
| 1 | **Add migration section**: how existing LeRobot datasets transition to the new format, dual-format period, DatasetClient integration, what happens to existing SnapshotDB records | R1, R2 | **Accept**  | New section 4.4 or standalone section 7. This is the #1 gap.                                |
| 2 | **Add before/after user workflow walkthroughs** to sections 3.2, 5.3, and 5.4 with concrete time/storage estimates                                                                | R2     | **Accept**  | Strengthens the pitch significantly. Use measured or well-estimated numbers.                |
| 3 | **Extract Section 6 (multi-rate) to a separate document** or reduce to a 1-page "Future Work" appendix                                                                            | R2     | **Accept**  | Keeps the document focused. Multi-rate is a different proposal with different stakeholders. |

### P1: High (should address before sharing)

| # | Change                                                                                                                                                            | Source | Disposition | Notes                                                                      |
|---|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------|-------------|----------------------------------------------------------------------------|
| 4 | **Add concurrency discussion**: manifest locking strategy (file locks, atomic rename, or DB-backed), record-while-train isolation, multi-user curation            | R1, R2 | **Accept**  | Add as subsection under section 4 or section 5.                            |
| 5 | **Add error recovery discussion**: JSONL truncation detection, manifest corruption recovery (keep previous version), disk-full during COW, interrupted recordings | R1, R2 | **Accept**  | Brief subsection; doesn't need to be exhaustive, but must be acknowledged. |
| 6 | **Add rough time estimates to priority table (4.3)**                                                                                                              | R2     | **Accept**  | Replace "Low/Medium" with "~2 days", "~1 week" etc.                        |
| 7 | **Restructure Section 3.3**: lead with the user benefit ("stats in 1ms, not 10 min"), move Welford details to an appendix or footnote                             | R2     | **Accept**  | Makes the section accessible to non-ML readers.                            |

### P2: Medium (nice to have, can iterate)

| #  | Change                                                                                          | Source | Disposition | Notes                                                                                                            |
|----|-------------------------------------------------------------------------------------------------|--------|-------------|------------------------------------------------------------------------------------------------------------------|
| 8  | **Add jargon glossary** or define terms on first use (remux, GOP, Welford, delta_indices, COW)  | R2     | **Accept**  | Simple to add, helps accessibility.                                                                              |
| 9  | **Add lazy-decode benchmark plan** or placeholder for results                                   | R1     | **Accept**  | At minimum, state the assumption and commit to profiling before choosing Path A as default.                      |
| 10 | **Clarify Path A vs Path B long-term intent**                                                   | R2     | **Accept**  | Add a paragraph to section 4.2 stating Path A is the target default; Path B is maintained for ecosystem interop. |
| 11 | **Disk space management**: storage visibility for snapshots, GC triggers, cache eviction policy | R2     | **Accept**  | Extend section 5.5 with a brief user-facing strategy.                                                            |

### P3: Deferred

| #  | Change                                                                                                      | Source | Disposition | Notes                                                                                                                                                                                                   |
|----|-------------------------------------------------------------------------------------------------------------|--------|-------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 12 | **Hardlinks as alternative to COW episode directories**                                                     | R1     | **Defer**   | Valid optimization but adds filesystem-specific complexity (hardlinks don't work across filesystems, don't work on all NFS setups). Worth exploring during implementation, not in the architecture doc. |
| 13 | **Detailed RecordingMutation / DeleteEpisodesMutation integration plan**                                    | R1     | **Defer**   | Belongs in the implementation spec, not the architecture proposal. The migration section (P0 #1) should acknowledge this exists.                                                                        |
| 14 | **Full benchmark suite** (snapshot creation, stats aggregation, lazy decode throughput, conversion speedup) | R1     | **Defer**   | Needs implementation first. Document should commit to benchmarking, not contain premature numbers.                                                                                                      |

### Rejected

| #  | Proposal | Source | Reason                                                                         |
|----|----------|--------|--------------------------------------------------------------------------------|
| -- | None     | --     | No feedback was rejected outright. All points are either accepted or deferred. |

---

## 6. Summary

The document's **core architecture is endorsed by both reviewers**. The
episode-isolated format with manifest-based snapshots is a clear
improvement over the current full-copy approach.

The three critical gaps are:

1. **Migration path** -- how we get from here to there
2. **User-facing storytelling** -- before/after workflows with numbers
3. **Document scope** -- Section 6 dilutes the message

Addressing the P0 and P1 items above would make the document ready for
a team-wide architecture review. P2 items can be iterated on after initial
feedback. P3 items are implementation-time concerns.
