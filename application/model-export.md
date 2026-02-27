# Model Export Feature Design

## Summary

Add a "Download" action to the model context menu that exports the model's
`exports/{backend}/` directory (`.pt` file + `metadata.yaml`) as a streamed zip
file. The zip structure is designed to be import-friendly for a future import
feature.

## Scope

- **Exported content**: Only the `exports/` subtree of a model directory
  (e.g. `exports/torch/act.pt` + `exports/torch/metadata.yaml`, ~198 MB).
- **Delivery mechanism**: Zip archive built in memory and returned via
  `StreamingResponse` with `Content-Disposition: attachment`.
- **Import support**: Not included in this iteration, but the zip layout is
  designed so a future `POST /api/models:import` can extract and register the
  model with minimal work.

## Architecture

### Backend

One new endpoint added to the existing models router.

#### `GET /api/models/{model_id}:export`

| Aspect | Detail |
|---|---|
| Router file | `backend/src/api/models.py` |
| Auth | None (matches existing endpoints) |
| Response | `StreamingResponse`, `application/zip` |
| Zip root | `{model_name}/exports/{backend}/{files}` |
| Error cases | 404 if model or exports dir not found |
| Path safety | Resolved paths validated against `models_dir` |

Flow:
1. Look up model by ID via `ModelService`.
2. Resolve `Path(model.path).expanduser() / "exports"`.
3. Validate the directory exists (404 if not).
4. Build a zip in memory containing all files under `exports/`, with archive
   paths rooted at `{model.name}/exports/...`.
5. Return as `StreamingResponse` with
   `Content-Disposition: attachment; filename="{model.name}.zip"`.

No changes to `ModelService`, `ModelRepository`, schemas, or the database.

### Frontend

#### Model context menu

| Aspect | Detail |
|---|---|
| File | `ui/src/routes/models/model-table.component.tsx` |
| Trigger | New "Download" `<Item>` in the existing `<Menu>` |
| Mechanism | Programmatically create an `<a>` element pointing to the export endpoint and click it |

No new API client hooks are needed since the browser handles the download
natively via the `<a>` element.

### OpenAPI spec

After the backend change, regenerate types:
```
npm run build:api:download && npm run build:api
```

The frontend download does not use `$api` hooks (it's a direct URL), so the
type regeneration is for spec-consistency only.

## Files modified

| File | Change |
|---|---|
| `backend/src/api/models.py` | Add `GET /{model_id}:export` endpoint |
| `ui/src/routes/models/model-table.component.tsx` | Add "Download" menu item |
| `ui/src/api/openapi-spec.json` | Regenerated (auto) |
| `ui/src/api/openapi-spec.d.ts` | Regenerated (auto) |

## Files NOT modified

- No schema changes
- No service-layer changes
- No database migrations
- No new dependencies

## Import-friendliness

Design decisions that make a future import straightforward:

1. **Zip structure** mirrors on-disk layout (`{name}/exports/{backend}/`).
   Import can unzip into a fresh UUID directory.
2. **`metadata.yaml`** contains everything needed to reconstruct a DB record
   (policy class, features, normalization, hyperparameters).
3. **No database IDs in the zip** -- uses model name, not UUID. Import assigns
   a new UUID.
4. **Pure file export** -- no DB dump, so import only needs to extract files
   and read `metadata.yaml`.

## Future considerations

- **Format selection**: Accept a `format` query parameter to serve
  ONNX / OpenVINO exports from `exports/{format}/` or trigger a conversion.
- **Quantization**: Add a `quantize` parameter that triggers a background job.
- **Import endpoint**: `POST /api/models:import` accepting multipart upload,
  extracting the zip, reading `metadata.yaml`, creating the DB record.
- **Larger exports**: If checkpoints are included (~600 MB+), switch from
  in-memory zip to chunked streaming with a pipe buffer.
