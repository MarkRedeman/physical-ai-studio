# Plan: Datasets without a required environment

## Goal

Allow users to create and import datasets without assigning an environment. An
environment is only needed when the user wants to record new episodes. A
dataset may then be associated with zero, one, or many environments configured
for recording.

This addresses [issue #849](https://github.com/open-edge-platform/physical-ai-studio/issues/849): imported datasets that are only needed for viewing or training should not require physical hardware setup, and users must be able to choose or change the environment used for recording.

The same decoupling pattern may later be applied to models: models imported
from outside Studio should not be required to reference a Studio dataset.

## Terminology

- **Dataset**: the persisted LeRobot dataset and its Studio metadata.
- **Environment**: a project-owned physical hardware setup of robots and
  cameras.
- **Recording environment**: the configured association of one dataset and one
  environment. It describes how this environment appends episodes to this
  dataset, including source-to-dataset stream mappings.

Although a recording environment is implemented as an association, it should
be a first-class API resource rather than a hidden pivot row.

## Data model

Replace `datasets.environment_id` with a new table:

```text
dataset_recording_environments
├── id              UUID primary key
├── dataset_id      FK datasets.id, ON DELETE CASCADE
├── environment_id  FK project_environments.id, ON DELETE CASCADE
├── mappings         JSON / JSONB, initially optional
├── created_at
└── updated_at

UNIQUE(dataset_id, environment_id)
```

Use a stable `id` instead of only a composite primary key. This gives the API
and runtime a durable identifier for the chosen configuration and provides a
natural parent for future mapping/configuration tables.

The backend must verify that the referenced dataset and environment belong to
the same project. The independent existing foreign keys do not enforce that
cross-project invariant by themselves.

### Migration

1. Create `dataset_recording_environments`.
2. Backfill one row for every existing `datasets.environment_id` value.
3. Drop `datasets.environment_id` and its ORM/Pydantic representation.
4. Update tests and fixture data that construct a dataset with an environment.

Deleting a dataset or environment should delete only its association rows. It
should not delete the other resource, and environment deletion should no
longer be blocked merely because datasets were once recorded with it.

## API

Dataset create and import-finalization requests no longer include or require
an environment. Dataset update should support both `name` and `default_task`.

Expose recording environments as explicit nested resources, for example:

```text
GET    /api/datasets/{dataset_id}/recording-environments
POST   /api/datasets/{dataset_id}/recording-environments
GET    /api/datasets/{dataset_id}/recording-environments/{recording_environment_id}
PATCH  /api/datasets/{dataset_id}/recording-environments/{recording_environment_id}
DELETE /api/datasets/{dataset_id}/recording-environments/{recording_environment_id}
```

The exact API prefix should follow established backend naming conventions when
implemented. The resource must return the selected environment and the stored
configuration, so the UI does not need to infer the association from a dataset
field.

Initial creation payload:

```json
{
  "environment_id": "<uuid>",
  "camera_mappings": [
    {
      "dataset_camera_key": "wrist",
      "environment_camera_id": "<uuid>"
    }
  ]
}
```

Persist physical camera IDs rather than environment camera display names.
Names are editable; IDs keep a saved mapping stable.

## Mapping configuration

### Camera mapping: first supported mapping

A dataset can use a camera key such as `wrist`, while the selected environment
calls the physical camera `gripper`. The recording environment maps the
dataset key to the physical environment camera. During recording, frames from
the mapped source are written under the dataset's established feature key, for
example `observation.images.wrist`.

Mapping creation may suggest exact name matches. It may also suggest a sole
unassigned source for a sole unassigned dataset key. Ambiguous and non-name
matching mappings require explicit user confirmation: do not infer semantic
equivalence such as `gripper` == `wrist` automatically.

### Future mappings

Design the `mappings` field as a versioned, structured configuration rather
than a camera-only unstructured list. Future supported mappings may include:

- dataset robot role / slot to environment robot ID, for bimanual or
  multi-robot recording;
- dataset feature name to environment feature name, covering joint positions,
  velocities, gripper values, force/torque, IMU, depth, tactile, and other
  observation streams;
- action-source or control-role configuration, such as the source of recorded
  follower actions.

Do **not** initially enable arbitrary robot joint/feature remapping. A name
alias is only safe when values have identical meaning and representation.
Degrees versus radians, sign inversions, offsets, limits, continuous versus
binary grippers, or differing degrees of freedom require explicit transforms
and validation; they are not simple mappings.

Similarly, camera resolution/FPS, synchronization, intrinsics/extrinsics, and
image conversion are constraints or transforms, not aliases. The first
implementation must not silently resize, resample, or numerically transform
recorded data.

## UI and recording flow

1. Remove the required environment picker from the new-dataset dialog.
2. Remove the required recording-environment picker from import finalization.
3. Add dataset-detail editing for the default task and recording environments.
4. When recording from a dataset:
   - use its only recording environment directly;
   - otherwise prompt the user to choose one or create a new association;
   - show camera mapping configuration when needed;
   - enter recording with the selected `recording_environment_id`.
5. When recording begins from an environment, implement the symmetric dataset
   selection flow later.

Replace the existing per-project last-used-dataset local-storage preference
with a last-used recording pair:

```ts
type LastUsedRecordingPair = {
    datasetId: string;
    recordingEnvironmentId: string;
};
```

This is browser/user preference only, not a persisted global database default.
It must be cleared or ignored if either referenced resource was deleted.

The recording route and runtime setup should identify the selected recording
environment explicitly, rather than retrieving `dataset.environment_id`.

## Runtime integration

The selected recording environment must be passed into runtime setup. Its
camera mappings determine which active physical camera provides each
dataset-side image feature. Runtime currently derives recording features from
the active hardware session, so this work must ensure the mapped dataset keys
are used when composing those features.

The existing WebSocket handshake currently provides robot/camera IDs and the
`load_dataset` command only provides a dataset ID. A later hardening step
should also include the recording-environment identity in the session contract,
resolve the environment server-side, and verify the requested devices match
that environment. This prevents browser state from claiming one environment
while a reused runtime worker controls another.

## Deferred: manifests and compatibility

Commit `1fae0b72d5598358e69b0c7884757344bb92657e` introduced independent
dataset, environment, and model manifest endpoints intended for future
compatibility checks. That work is deliberately out of scope for the first
recording-environment implementation.

When compatibility is revisited, it belongs on the recording-environment
boundary:

1. A shared backend compatibility service compares the current dataset and
   environment manifests.
2. Compatibility is evaluated whenever creating/updating a recording
   environment and immediately before recording, not cached permanently in
   the association. Environments can change after an association is created.
3. Store an optional compatibility snapshot/report on the recording environment
   only for UI/audit purposes, including manifest versions or fingerprints and
   diagnostics. The current live comparison remains authoritative.
4. Evaluate mappings as part of comparison: a `wrist` dataset stream mapped to
   the environment's `gripper` camera is valid when the mapping resolves its
   schema requirements.
5. Return explainable incompatibility reasons to the UI, rather than only a
   boolean, so users can correct mappings or hardware configuration.

Before adopting the referenced manifest implementation, reconcile its robot
identity and feature semantics with runtime recording. In particular, dataset
recording metadata currently derives robot type from the runtime follower name,
while the environment manifest derives type from the robot catalog; those are
not necessarily comparable identifiers.

## Delivery order

1. **Decouple datasets from environments**: migration, backend schemas,
   creation/import flows, and tests.
2. **Recording-environment CRUD**: association resource, project ownership
   validation, dataset-detail management, and tests.
3. **Recording selection and camera mapping**: selection dialog, persisted
   mappings, last-used pair, mapped runtime feature generation, and tests.
4. **Runtime hardening**: bind runtime sessions to recording-environment IDs
   and validate devices server-side.
5. **Compatibility follow-up**: manifests, comparison service, diagnostics,
   and enforcement once the required semantics are agreed.

## Verification

Backend coverage should include migration/backfill, environment-less dataset
create/import, cross-project association rejection, association deletion and
cascades, mapping serialization, and mapped runtime recording behavior.

UI coverage should include environment-less dataset creation and import,
recording-environment selection, saved camera mappings, last-used pair
fallback/cleanup, and handling deleted associations.
