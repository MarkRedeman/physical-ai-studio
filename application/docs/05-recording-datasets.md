# Recording Datasets

This guide covers what users do in the UI to collect demonstration episodes.

## 1) Create a dataset
[//]: # (Screenshot suggestion: New Dataset dialog showing Environment picker, Dataset name, Task, and Save button.)

In the Datasets section:

- Click New Dataset.
- Select an environment.
- Enter Dataset name.
- Optionally enter Task.
- Click Save.

## 2) Open recording mode
[//]: # (Screenshot suggestion: recording page header with environment/task context and readiness/loading indicators.)

From your dataset page, start recording.

In recording view, you will see:

- Environment and task context.
- Robot and camera panels.
- Episode controls.

If initialization is still running, the UI shows loading status for dataset/environment readiness.

## 3) Record episodes
[//]: # (Screenshot suggestion: active recording state with Accept and Discard buttons visible; include hotkey hints if shown.)

Typical loop:

1. Reset your physical scene.
2. Enter or confirm the task.
3. Click Start episode.
4. Perform the demonstration.
5. Click Accept or Discard.

Keyboard shortcuts in recording view:

- Right arrow: Start episode or Accept.
- Left arrow: Discard.

## 4) Build quality early
[//]: # (Screenshot suggestion: recording workspace view with camera feeds and robot panel used for quality checks.)

From a user perspective in the UI:

- Keep tasks consistent when building one dataset.
- Discard failed or noisy attempts.
- Watch camera framing and lighting before each episode.

The UI also surfaces a recommendation to collect enough episodes before first training.

## Next
[//]: # (Screenshot suggestion: optional docs links screenshot to Dataset Management and Training Policies chapters.)

- Manage and curate datasets in `application/docs/06-dataset-management.md`.
- Start training in `application/docs/07-training-policies.md`.
