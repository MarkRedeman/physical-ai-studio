---
name: pas-hf-cli
description: Inspects, imports, and exports Physical AI Studio models and LeRobot datasets on Hugging Face Hub using physicalai-studio hf. Use for Hub repository inspection, project-local model/dataset registration, publishing Studio assets, and resuming interrupted uploads or downloads. DO NOT USE FOR converting a checkpoint to ONNX/OpenVINO/Torch/ExecuTorch (use pai-export-policy), training a policy, or changing Studio source.
license: Apache-2.0
---

# Transfer Studio Assets with Hugging Face Hub

Use the installed `physicalai-studio` application CLI, not `physicalai export` or
the separate `hf upload/download` commands. Studio's commands handle local database
registration, native artifact layouts, portable metadata, and resumable transfers.
No Studio source checkout is required. The command reference is the
[Studio Hub CLI guide](https://github.com/open-edge-platform/physical-ai-studio/blob/main/application/backend/docs/huggingface_integration.md#studio-hub-cli).

## Establish the target and permission

1. Check `physicalai-studio hf --help` and the selected subcommand's `--help` in
   the user's Studio environment. If `hf` is missing, report the installed version
   and request an upgrade; do not patch installed packages or Studio source.
2. Identify the operation, `OWNER/REPO`, and local project/resource UUIDs. Use the
   same `STORAGE_DIR`, `SETTINGS_FILE`, and database configuration as Studio. The
   CLI operates on local storage, **not** a remote Studio HTTP server; for a Docker
   deployment, run it inside the appropriate backend container. Do not initialize
   a new database to work around an unexpected missing project.
3. Authentication uses Studio's saved Hugging Face token, then `HF_TOKEN`, otherwise
   anonymous access. Cached `hf auth login` credentials are not used automatically.
   Imports need read access; exports need write/create-repository permission.
   Ask the user to configure credentials through Studio settings or their secret
   manager. Never request, echo, or paste a token into a command, report, or file.
4. Explain remote metadata reads and estimated payload size. Get authorization
   before a large download or upload, and explicit permission before public
   publishing or overwriting remote files. Do not add `--public` or `--update`
   merely to work around a failure.

Done when the intended local Studio installation, resource, destination, and
permitted operation are unambiguous.

## Inspect before importing

```bash
physicalai-studio hf model inspect OWNER/MODEL --json
physicalai-studio hf dataset inspect OWNER/DATASET --json
```

Inspection downloads only listings and bounded JSON metadata, not weights,
checkpoints, Parquet shards, or videos. Check `importable`, `errors`, `warnings`,
`detected_format`, `required_import_options`, `files`, `size_bytes`, and
`size_complete`. Pin subsequent imports to the returned `revision` commit SHA;
carry over the returned `subfolder` with `--subfolder PATH` if nonempty.

`importable: true` confirms metadata compatibility, not successful inference or
complete video decoding. Exit code 0 alone is not enough: an accessible but
unsupported/incomplete resource has `importable: false`. Multiple asset roots
require another inspection with `--subfolder PATH`.

Supported inputs are Studio model packages with backend manifests and LeRobot
v3.0 datasets. Do not promise arbitrary Transformers models, checkpoint-only
repositories, or LeRobot v2 conversion. A SnapFlow checkpoint's presence and the
original logger hparams can describe different training phases; do not infer
model flags from the repository name or hparams alone.

## Import a model

Obtain an existing project UUID and a local dataset UUID **in that project**.
Compare model input/output features with the intended dataset/camera/joint schema;
do not guess a dataset association or silently remap features.

```bash
physicalai-studio hf model import OWNER/MODEL \
  --project-id PROJECT_UUID --dataset-id DATASET_UUID \
  --revision COMMIT_SHA --dry-run --json

# After approval, repeat without --dry-run:
physicalai-studio hf model import OWNER/MODEL \
  --project-id PROJECT_UUID --dataset-id DATASET_UUID \
  --revision COMMIT_SHA --json
```

Use `--name NAME` for an explicit display name. Imports generate a new local UUID;
foreign jobs, snapshots, and parent-model links are not restored. Exports-only
trees are normalized under `exports/`. Without a checkpoint the model is
deployment-only, not resumable for training.

Done when the result supplies the new local model `id`, `project_id`, `path`, Hub
`repo_id`, and pinned `revision`. Do not claim inference was tested.

## Import a dataset

Only the project association is required; there is **no `--environment-id`**.

```bash
physicalai-studio hf dataset import OWNER/DATASET \
  --project-id PROJECT_UUID --revision COMMIT_SHA --dry-run --json

# After approval, repeat without --dry-run:
physicalai-studio hf dataset import OWNER/DATASET \
  --project-id PROJECT_UUID --revision COMMIT_SHA --json
```

Studio temporarily creates an empty environment in the same transaction as the
dataset. It contains no robots/cameras and is not evidence of configured hardware.
Do not ask for a model-style `--dataset-id` or create hardware configuration to
make the import pass. `--name NAME` overrides the display name.

Done when the result supplies a new dataset `id`, `environment_id`, `project_id`,
`path`, and pinned source `revision`. Validation checks task/episode metadata and
referenced data/video shards, not every data row or decoded video.

## Export a model or dataset

Get the local resource UUID and destination `OWNER/REPO`. Exports infer the project
from the local record; they do not accept `--project-id`.

```bash
physicalai-studio hf model export OWNER/MODEL --model-id MODEL_UUID --dry-run --json
physicalai-studio hf dataset export OWNER/DATASET --dataset-id DATASET_UUID --dry-run --json
```

Review the selected files and sensitive metadata before publishing. Model exports
can include checkpoints, backend packages, and training hparams/metrics; snapshots
and hidden files are excluded, but arbitrary hparams are not scrubbed for secrets.
Stop dataset recording and avoid modifying the asset throughout the transfer.
Known active model training jobs are rejected; dataset recording is not locked.

After approval, run the selected command without `--dry-run`. New repositories
default to private. Use `--public` only with explicit approval; it does not change
an existing repository's visibility. Nonempty destinations require authorized
`--update`. If stale files would remain, choose a new repository rather than
deleting files or force-merging shards. Existing cards/licenses are preserved.

Done when the result includes the repository `url`, completion `revision`, and
local resource `id`. A completed `studio.json` marks a Studio-compatible upload;
it does not convert the policy to another inference backend or prove deployment.

## Resume instead of starting over

Transfers print a transfer UUID and exact recovery command on stderr before
payload transfer. Save that command. On failure, prefer `error.resume_command`
from JSON output and append `--json` for machine-readable recovery:

```bash
physicalai-studio hf model import --resume TRANSFER_UUID --json
physicalai-studio hf model export --resume TRANSFER_UUID --json
physicalai-studio hf dataset import --resume TRANSFER_UUID --json
physicalai-studio hf dataset export --resume TRANSFER_UUID --json
```

- Resume with the same Studio installation/storage/database, resource type, and
  operation. Do not supply a repo, IDs, revision, name, or other transfer options
  alongside `--resume`; `--json` is allowed.
- Preserve staging and transfer caches. Completed files/uploads are reused;
  hashing, remote checks, or incomplete chunks can repeat. Do not promise zero
  retransmission of every interrupted request.
- Ctrl-C stops the upload worker while retaining state. Do not run two resumes
  concurrently or edit the remote repository during upload.
- An incomplete `studio.json` blocks import until export finishes. If the local
  source changed, report that the transfer cannot resume safely and plan a new
  export, normally to a fresh repository. Do not edit transfer records to bypass
  the check or take over another transfer's incomplete export.
- Repeating a completed import does not register duplicate models/datasets or
  placeholder environments. Do not start a new import merely because a network
  failure occurred after registration.

All commands support `--json`: stdout contains one object and progress is on
stderr. Check the exit code and structured `error`, not progress text. CLI syntax
errors follow Click's standard stderr usage format. End with the resource IDs,
pinned revision, warnings, and either the completed result or exact resume command;
never include credentials.
