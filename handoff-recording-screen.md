# Recording / runtime session screen handoff

## Direction

Continue refining the unified runtime workspace: live robot/camera views on the left, session setup on the right. Recording and inference cards always remain visible, but **one mode is selected** using radios in their top-left headers. Entry points set the initial selection and section order; switching does not reorder or collapse cards.

This is still a UX experiment. Prioritize visual iteration and small signed commits; do the full technical cleanup before treating it as release-ready. Do not revert unrelated work or include the existing untracked files in commits.

## What changed

- Removed the separate top page header. The episode toolbar contains back navigation, recording actions, and timer; dataset counts and guidance now live in the Recording card.
- Consolidated recording into one provider so task edits, buttons, and hotkeys share state. Removed duplicate hidden recording controls and their hotkey registrations.
- Added distinct Environment, Recording, and Inference cards. Resource actions live in each card's top-right corner, not in `PartRow`.
- Replaced routine reload/unload menus with “Change…” and contextual recovery: “Reconnect devices” or “Try again.” Backend command names remain unchanged.
- Removed normal green health lights. Failures highlight the affected card header; device errors automatically open details. The Environment header chevron toggles inline details, **not a disclosure heading**.
- Adapted [Dawid's PR #1343](https://github.com/open-edge-platform/physical-ai-studio/pull/1343): compact device cards show names, roles, and activity. Follower tiles are purple; leader tiles are blue with mirrored thumbnails; camera tiles are teal with camera icons and resolution.
- Distinguished readiness from activity: robot control produces “Active”; healthy cameras show “Streaming”; normal readiness stays muted. Errors override activity.
- Consolidated episode counts into a dataset-progress block **above the dataset name** in Recording. Show the total, this session's contribution only when nonzero, progress toward the suggested 50 episodes, visible guidance, and a compact labeled **Recording tips** action. Removed the separate “No episodes saved…” message. The recommendation is not a training prerequisite. A fixed recording-rate note sits directly below the dataset name.
- Renamed Task to **Current task** and kept a multiline textarea—**do not replace it with a combobox**.
- Added a compact task-history icon action beside **Current task**, with a tooltip and accessible “Use previous task…” label. It opens a popover with search and wrapping task buttons. Suggestions combine previous episode tasks and successfully saved tasks from this session, trim blanks, and deduplicate.
- Put editable recording rate **before Current task** for new datasets. Explain that the first saved episode locks it; show only a muted fixed-rate summary afterward.
- Replaced empty inference's “No model selected” and header chooser with a clickable, left-aligned text resource selector and a right-hand chevron. Do not add a play icon here: it suggests playing a movie. An earlier large centered/dashed tile felt inconsistent with the rest of the screen. Projects without runnable exports retain a quiet training/export placeholder.
- Kept inference execution settings behind a disclosure, with execution-mode radios and the chunk threshold shown only for asynchronous execution.
- Mode selection adds a border glow even when idle; “Recording”/“Running” text describes actual execution separately. The toolbar shows the selected workflow's actions. Selecting a mode never starts motion; switching first holds any existing robot control and waits for its acknowledgement. Switching is disabled during episodes and shared recording/model commands.

## Code map

Under `application/ui/src/features/runtime-session/workspace/`:

- `session-workspace.tsx` / `.module.css`: layout, cards, empty inference well, header actions, device-details toggle.
- `parts-panel.tsx` / `.module.css`: resource metadata, action menus, environment details.
- `environment-devices.tsx` / `.module.css`: role-specific robot/camera visuals and status mapping. Replaced the earlier `environment-robots.*` files.
- `record-mode.tsx`: shared recording state, commands, hotkeys, toolbar, recording rate.
- `session-mode.tsx`: selected mode, safe switching, and shared recording/model command lock.
- `current-task-picker.tsx` / `.module.css`: multiline editor and searchable task-history popover.
- `recording-summary.tsx` / `.module.css`: dataset-progress block, persisted + session totals, and recording guidance popover.
- `back-to-dataset-button.tsx`: waits for dataset finalization before leaving; stays on the page if it fails.
- `inference-mode.tsx`: shared inference provider, model settings, and inference Run/Stop toolbar.

Entry points: `application/ui/src/routes/datasets/record/index.tsx` and `application/ui/src/features/models/inference/inference-page.tsx`. Shared hotkey handling is in `application/ui/src/features/hotkeys/use-hotkey.ts`; it now ignores handled events and dialog/menu/listbox interactions.

## Pay attention when continuing

1. **Session lifetime and hardware:** keep suspension inside `RuntimeSessionProvider`; remounting it ends the session. Use mocked sockets for automation. A second browser opening the live recording URL cannot acquire hardware already owned by the user's session. Never stop their session or change robot control just to obtain a screenshot.
2. **Saving is not finalization:** saved episodes stay session-local until the dataset is closed/copied back. Keep totals and task history consistent with that contract; avoid double-counting after finalization or stale episode caches when switching datasets.
3. **Leaving and command races:** the dataset back button awaits finalization and is disabled during recording. The generic inference back button does not use the same finalization flow. Recording/model commands and mode switching now share a lock, but teleoperation's own mutation and dataset back navigation still need a pending-race review. Also check browser Back/tab-close behavior.
4. **Task state:** test dataset switching, failed saves, discard, and reuse of tasks saved during this session. The default-task effect can overwrite edits if the server's default changes during a refetch; revisit this during cleanup.
5. **Rate locking:** respect `fps_fixed` and `episodes_recorded > 0`. A discarded episode is not a saved first episode. Verify failures while changing rate and the slider-to-summary transition.
6. **Status semantics:** don't infer activity just from a connected device. Preserve failure priority and automatic expansion when a new error arrives, while allowing the user to collapse details. Camera statuses use `environment/{camera.key}`, not camera IDs.
7. **Visual/accessibility follow-up:** check long names/tasks, many cameras, narrow screens, light theme, popover placement, keyboard focus/return, and disabled-state explanations. Current camera/robot previews use mocked devices, not real feeds.

## Verification state

Before the latest experimental pass, the full unit suite passed (559 passed, one skipped), the production build passed, and targeted browser checks passed. Full type-check had unrelated existing model-renaming API and settings-fixture errors.

The device/task-history pass checked four mocked browser preview flows: device overview, task-history search/selection, first-episode rate setup, and model selection. The mode-selector follow-up checked radio switching, toolbar changes, stopping existing model control, keyboard switching, and disabled switching during an episode. The progress/help follow-up previewed an empty dataset, persisted + session episode totals, recording tips, and the compact model selector. **Do not assume current HEAD passes the full suite, lint, type-check, or build.** The latest shared mode/command providers and task-history/camera changes still need regression coverage and verification.

Run UI commands from `application/ui/` when ready: `npm run lint`, `npm run format:check`, `npm run type-check`, `npm run test:unit`, `npm run build`, and `npm run test:component`. Browser previews used installed Chrome via a temporary Playwright config because the bundled Chromium executable was missing.

## Commit checkpoints

Started from WIP `f3c03701b`; all subsequent commits were signed:

- `bb1a7434b`: unified workspace and recording controls.
- `c0743c8a7`: friendlier actions and contextual errors.
- `951857b0f`: streamlined navigation and device-details chevron.
- `9944f3d35`: initial robot activity cards.
- `3080e91ee`: differentiated robot roles, camera visuals, resource typography.
- `875873a12`: recording setup, task reuse, and actionable model selection.
- `775f26cf3`: mode radios, active glow, and mode-specific toolbars.
