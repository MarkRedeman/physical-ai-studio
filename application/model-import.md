# Model Import Feature Design

## Summary

Add the ability for users to import a previously exported model zip file into
the application. The user selects a zip file and provides a model name; the
backend extracts the model artifacts, reads `metadata.yaml` to determine the
policy type, and creates a new database record. A `manifest.json` file embedded
in the export zip preserves the original model's identifiers so that
re-importing into the same application can reconnect relationships (dataset,
snapshot).

## Nullable Foreign Keys: Trade-offs

The `models` table currently has `NOT NULL` foreign keys to `datasets` and
`snapshots`. An imported model may not have a corresponding local dataset or
snapshot, so these columns must become nullable.

### Pros

- **Correct data modeling**: An imported model genuinely does not have a local
  dataset or snapshot. NULL accurately represents "no relationship" rather than
  pointing to a fake sentinel record.
- **No sentinel complexity**: No need to create and manage placeholder
  Dataset/Snapshot rows that would appear in UI lists, need special deletion
  protection, and complicate queries.
- **Cascade safety**: When a Dataset or Snapshot is deleted, its cascade
  (`delete-orphan`) only affects models that reference it. Imported models with
  NULL foreign keys are unaffected -- they won't be accidentally deleted.
- **Clean queries**: `WHERE dataset_id IS NOT NULL` cleanly separates trained
  models from imported models if needed.
- **Future-proof**: If more import sources are added (e.g. Hugging Face hub),
  they also won't have local datasets/snapshots.

### Cons

- **Schema migration required**: Existing databases need an Alembic migration.
  Since SQLite doesn't natively support `ALTER COLUMN`, the migration uses
  batch mode (table rebuild). This is safe but slightly more complex than
  additive migrations.
- **NULL handling in application code**: Any code that accesses `model.dataset_id`
  or `model.snapshot_id` must now handle `None`. This includes:
  - The `GET /api/models/{model_id}/tasks` endpoint (which fetches the dataset
    to get task names) -- must return an empty list or 404 for imported models.
  - Any future UI that displays dataset/snapshot info for a model.
- **Weaker referential integrity**: The database no longer guarantees that
  every model has a dataset and snapshot. This is intentional for imported
  models but means application code must be more defensive.
- **OpenAPI schema change**: `dataset_id` and `snapshot_id` become optional in
  the API schema, which is a breaking change for any external API consumers
  (though none currently exist).

### Alternatives considered

**Sentinel records**: Create a well-known "Imported Models" Dataset and
Snapshot per project. Rejected because:
- Sentinel records appear in dataset/snapshot lists in the UI
- Deleting a sentinel cascades to all imported models
- Requires special logic to prevent user deletion of sentinels
- Adds conceptual complexity ("this dataset isn't really a dataset")

### Mitigation

To support re-import round-trips (export -> delete -> re-import on the same
application), the export zip includes a `manifest.json` with the original
`dataset_id` and `snapshot_id`. On import, the backend checks whether these
resources still exist in the database and reconnects them if so. This means:

- Export from App A, import into App A: dataset/snapshot are reconnected
- Export from App A, import into App B: dataset/snapshot are NULL (expected)
- Export, delete model, delete dataset, re-import: dataset/snapshot are NULL

## Architecture

### Export Enhancement

The `GET /api/models/{model_id}:export` endpoint is updated to include a
`manifest.json` in the zip root:

```
{model_name}/
  manifest.json           <-- NEW: model metadata for re-import
  exports/
    torch/
      act.pt
      metadata.yaml
```

`manifest.json` contents:
```json
{
  "name": "My Model",
  "policy": "act",
  "properties": {},
  "original_model_id": "8313591e-...",
  "original_project_id": "...",
  "original_dataset_id": "...",
  "original_snapshot_id": "...",
  "exported_at": "2026-02-27T15:30:00",
  "physicalai_version": "0.1.0"
}
```

### Import Endpoint

`POST /api/models:import` accepts multipart form data:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | UploadFile | Yes | The zip archive |
| `name` | string (Form) | Yes | Model name |
| `project_id` | UUID (Form) | Yes | Target project |

Flow:
1. Validate the zip (must contain `exports/{backend}/metadata.yaml`)
2. Generate a new model UUID and create the model directory
3. Extract `exports/` into the new model directory
4. Read `manifest.json` (if present) for original dataset/snapshot IDs
5. Check if referenced dataset/snapshot exist locally; link if so
6. Parse `metadata.yaml` to extract the policy type
7. Create the Model DB record
8. Return the created Model

### Database Migration

New Alembic migration making `dataset_id` and `snapshot_id` nullable on the
`models` table using batch mode (required for SQLite).

### Frontend

An "Import model" button is added next to "Train model" on the models page.
It opens a modal with:
- A text field for the model name
- A file picker (using React Spectrum's `FileTrigger`) for the zip file
- Import and Cancel buttons

## Files Modified

| File | Change |
|------|--------|
| `backend/src/api/models.py` | Update export; add import endpoint |
| `backend/src/db/schema.py` | Make `dataset_id`, `snapshot_id` nullable |
| `backend/src/schemas/model.py` | Make `dataset_id`, `snapshot_id` optional |
| `backend/src/alembic/versions/<new>` | Nullable FK migration |
| `ui/src/routes/models/import-model.tsx` | New ImportModelModal |
| `ui/src/routes/models/index.tsx` | Add "Import model" button |
| `ui/src/api/openapi-spec.json` | Regenerated |
| `ui/src/api/openapi-spec.d.ts` | Regenerated |
