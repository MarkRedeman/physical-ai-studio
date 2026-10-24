# Hugging Face Integration

Several policies download assets from Hugging Face Hub (for example, SmolVLA,
Pi0.5, and other Hub-backed models). Studio also provides standalone CLI commands
to inspect, import, and export Studio models and LeRobot v3 datasets.

If no Hugging Face token is configured, the backend logs a warning and Hub
access is unauthenticated.

Configure a token for any workflow that depends on Hugging Face-hosted assets.
Without a token, model downloads may fail (for example, due to anonymous rate
limits or access restrictions on gated/private repositories).

## Required token permissions

For model download in Physical AI Studio, use a token with **read-only** Hub
access:

- **Required:** `Read` permission for model repositories.
- **Not required:** `Write` or admin permissions.
- **If using gated/private models:** the token must belong to a Hugging Face
  account that has been granted access to those repositories.

If you use a fine-grained token, grant read access to the specific model repos
you plan to train from.

For the CLI below, grant **read** access to the model or dataset repositories you
import. Exports require **write** access to the destination and permission to
create repositories when the destination does not exist. A read-only training
token is not sufficient for export.

## Create a Hugging Face token

1. Sign in to [huggingface.co](https://huggingface.co/).
2. Open **Settings** -> **Access Tokens**.
3. Create a new token.
4. Grant read access for downloads, and write/create-repository permissions if
   you intend to export (see required permissions above).
5. Copy the token value.

## Configure the token

### Native backend

Set the token from the UI: **Settings > General > Hugging Face**. This persists
the token to the backend's `settings.json` file.

Alternatively, edit `settings.json` directly (see
[`get_settings_file_path`](../src/settings.py) for its location) and set
`huggingface.hf_token`.

### Docker deployment

Add the token to `application/docker/.env`:

```env
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

docker-compose injects this as a real container environment variable. Then run
Docker Compose as usual:

```bash
cd application/docker
docker compose up
```

## Verify setup

- For CLI use, inspect a repository that your account should be able to access:
  `physicalai-studio hf model inspect OWNER/PRIVATE_MODEL --json`. This reads
  metadata without downloading weights. Success on a public repo alone does not
  prove that credentials are configured or that private/write access is granted.
- For training, confirm there is no missing-token warning when starting an
  otherwise intended Hub-backed training job. Do not start a large training job
  solely to test credentials.

## Studio Hub CLI

These commands operate directly on the **local** Studio database and storage.
Run them in the backend environment, using the same `STORAGE_DIR`, `SETTINGS_FILE`,
and database configuration as the running application. For Docker deployments,
run them inside the backend container. They do not call a remote Studio server.
Imports/exports require an initialized database; metadata inspection does not.
If a project is unexpectedly missing, verify the installation configuration before
initializing anything. See the [Studio CLI guide](studio_cli.md) for command
discovery, local-folder imports, and database maintenance.

Authentication uses the saved Studio token first, then `HF_TOKEN`. Without either,
access is explicitly anonymous; the CLI does not silently use another account's
cached `hf auth login` credentials. Tokens are never saved in transfer records.

### Inspect before importing

```bash
physicalai-studio hf model inspect MarkRedeman/dice-cleanup-pi05-snapflow-200 --json
physicalai-studio hf dataset inspect MarkRedeman/websummit-computex --json
```

Inspection fetches repository listings and bounded JSON metadata, not weights,
checkpoints, Parquet shards, or videos. The structured result includes:

- `schema_version`, `repo_id`, and the resolved immutable `revision` (commit SHA).
- The detected `subfolder` and `detected_format`.
- Model policies, backend manifests, and checkpoint names; or dataset features,
  cameras, frame/episode counts, and FPS.
- `importable`, `errors`, `warnings`, and `required_import_options`.
- Selected `files`, `size_bytes`, and whether the size estimate is complete.

`importable: true` means **metadata validation passed**, not that the model was
loaded or every video decoded. An inspected but unsupported/incomplete resource
returns exit code 0 with `importable: false`; network/authentication failures
return a nonzero exit code.

Use `--revision BRANCH_TAG_OR_SHA` to choose a version. If more than one asset is
present, choose it with `--subfolder PATH`. Inspection output supplies these
arguments directly; imports run the same inspection again at the pinned revision.

### Import

```bash
physicalai-studio hf model import OWNER/MODEL \
  --project-id PROJECT_UUID --dataset-id LOCAL_DATASET_UUID \
  --revision COMMIT_SHA --dry-run --json

physicalai-studio hf dataset import OWNER/DATASET \
  --project-id PROJECT_UUID --revision COMMIT_SHA --dry-run --json
```

Replace `COMMIT_SHA` with the `revision` returned by inspection and UUID placeholders
with existing Studio record IDs. Add `--subfolder PATH` if inspection selected a
nested root. Review the dry-run result, then repeat without `--dry-run` to transfer
and register the asset. These commands are distinct from local-folder imports:
`datasets import-dir` requires `--environment-id`, while Hub dataset import does not.

`--name NAME` overrides the display name. Otherwise the name comes from Studio
metadata, or the repository name for legacy repositories. Imports always generate
new local UUIDs; foreign project/job/snapshot IDs are not restored.

Model imports require a dataset from the target project for Studio's local
dataset/environment association. The command does not guess a dataset or rewrite
camera/joint names. Check that the dataset's feature schema is appropriate for
the model before using it for inference or further training.

Dataset imports currently create an **empty placeholder environment** in the same
transaction as the dataset record. No robots or cameras are configured. This is
temporary until Studio allows datasets without environments.

Supported layouts are Studio policy packages (`exports/<backend>/manifest.json`)
and LeRobot v3.0 datasets. An exports-only model tree (`torch/manifest.json`, for
example) is normalized under `exports/` locally. Models without checkpoints can
be imported for deployment but cannot resume training. Arbitrary Transformers
layouts, checkpoint-only repositories, and LeRobot v2 conversion are not supported.

Dataset validation also reads task/episode Parquet metadata and checks referenced
data/video shard paths. It does not decode every video or validate every data row.
Model validation inspects manifests and referenced files without deserializing
checkpoints or instantiating classes specified by the repository.

### Export

```bash
physicalai-studio hf model export OWNER/MODEL --model-id MODEL_UUID --dry-run --json
physicalai-studio hf dataset export OWNER/DATASET --dataset-id DATASET_UUID --dry-run --json
```

After reviewing the selected files and destination, repeat without `--dry-run` to
publish. Export's `REPO_ID` is the destination; its UUID option identifies the local
asset. Import/export commands omit `REPO_ID` only when resuming a saved transfer.

The local record determines the project. New repositories are **private** unless
`--public` is supplied. That flag does not change an existing repository's visibility.
Nonempty destinations require `--update`. Updates that would leave stale payload
files fail rather than delete or silently merge them; use a fresh repository.
Existing `README.md` and license files are preserved. Local cards/licenses are
included when the destination does not already contain them.

Exports upload native files, not ZIP archives. A versioned `studio.json` preserves
the display name, model policy/version and LoRA/DoRA/SnapFlow flags, or the dataset
default task. Source UUIDs are provenance only. Snapshots, hidden cache files, and
unrecognized top-level model files are not exported. Supported checkpoints,
backend artifacts, and Lightning `version_0/hparams.yaml` and `metrics.csv` are
included. Review metadata and hyperparameters for sensitive information before
publishing; the CLI does not scrub arbitrary training configuration contents.

Do not modify an asset or record into a dataset during export. Known active model
training jobs are rejected, and changes to selected source files or their inventory
invalidate a transfer. Dataset recording is not locked by this standalone CLI.

### Resume interrupted transfers

Every new transfer prints a transfer UUID and a ready-to-run command on stderr
**before transferring payload data**:

```bash
physicalai-studio hf model import --resume TRANSFER_UUID
physicalai-studio hf dataset export --resume TRANSFER_UUID
```

Use the same resource kind, operation, `STORAGE_DIR`, and database configuration.
Do not supply new transfer options alongside `--resume`; `--json` is allowed.
Resolve authentication again through Studio settings or `HF_TOKEN` if needed.

Downloads retain complete files, partial downloads, and Hub metadata in hidden
staging directories under the destination's models/datasets directory. They
resume from the original pinned commit, not the latest branch contents. An
already-committed import is detected by its saved UUID, so recovery does not
create duplicate resources or placeholder environments.

Uploads retain Hub upload bookkeeping under
`<storage_dir>/cache/hf/transfers/<transfer-id>/payload/`. They use hard links where
possible, falling back to staged copies across filesystems. Completed uploads
are reused; the SDK may repeat hashing, remote checks, or an incomplete chunk.
Byte-perfect continuation of every interrupted request is not guaranteed.
The SDK handles network retries; Ctrl-C stops its upload worker and retains state.
Deleting staging/cache files sacrifices resumability.

Large uploads can create multiple commits. Their `studio.json` declares
`status: uploading` until the payload completes; Studio refuses to import an
incomplete export. Do not concurrently edit the remote repository. Another
Studio transfer cannot take over an in-progress export. A changed local source
requires a new transfer, normally to a new repository.

Completed exports remove their staged payload and retain a small transfer receipt.
Interrupted transfers intentionally retain data until resumed or manually removed.

### Automation and dry runs

All commands support `--json`; progress stays on stderr and stdout contains one
JSON object. Operational errors return nonzero and an `error` object with a
`code` and `message`. Errors during a retained transfer also include `transfer_id`
and `resume_command`. Invalid CLI syntax follows standard Click usage errors.

Import/export `--dry-run` validates the plan without downloading the payload,
creating a Hub repository, or registering a local asset. Import dry runs still
read local associations and remote metadata; export dry runs validate the local
payload and destination. Disk checks honor Studio's import size/headroom settings.
Commands never prompt interactively for missing information or credentials.

### Troubleshooting

| Symptom | Next action |
| --- | --- |
| Inspection exits 0 but `importable` is false | Read `errors`; select an ambiguous root with `--subfolder`, or finish an incomplete export before importing. |
| Project/model/dataset UUID is not found | Verify local storage/database configuration and the record's UUID. Do not create a fresh database or substitute a Hub repo name. |
| Access denied or repository not found | Verify spelling and token access; a private repo can appear missing. Use Studio settings or `HF_TOKEN`, not an assumed cached login. |
| Export fails with a read-only token | Grant write permission for that destination; successful inspection/dry-run does not establish write access. |
| Network interruption | Run the reported `resume_command` with the same storage/configuration; do not start another transfer or delete staging files. |
| Source changed or stale remote shards detected | Plan a new export, normally to a fresh repo; do not edit transfer records or silently merge dataset shards. |
| CLI option is unknown | Check that command's `--help` and the installed version; local-folder and Hub imports have different flags. |

## Security notes

- Never commit real tokens to source control.
- Store tokens in `settings.json`, the Docker `.env` file, or your secret manager.
- Rotate the token immediately if it is exposed.
