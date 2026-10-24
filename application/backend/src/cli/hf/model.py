# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Explicit model commands, independent of dataset command options."""

from uuid import UUID

import click

from cli.hf.output import invoke


@click.group()
def model() -> None:
    """Inspect, import, and publish Studio models on the Hub.

    REPO_ID is OWNER/REPO. Import registers a new local model in an existing
    project and requires a suitable dataset in that project. Export publishes
    existing files; it does not convert a checkpoint to another inference format.
    """


@model.command("inspect")
@click.argument("repo_id")
@click.option("--revision", default="main", show_default=True, help="Source branch, tag, or commit SHA.")
@click.option("--subfolder", help="Model root within the repository.")
@click.option("--json", "as_json", is_flag=True, help="Emit one JSON object on stdout; diagnostics go to stderr.")
def inspect_model(repo_id: str, revision: str, subfolder: str | None, as_json: bool) -> None:
    """Inspect OWNER/REPO without downloading weights or checkpoints.

    Checks Studio manifests, backends, and selected files. JSON includes the
    resolved revision SHA, subfolder, size estimate, warnings, and importable.
    Exit code 0 means inspection succeeded, not necessarily importable=true.
    Metadata compatibility does not prove successful inference.

    \b
    Example:
      physicalai-studio hf model inspect OWNER/MODEL --json
    """
    from cli.hf.inspection import inspect_remote_model

    invoke(inspect_remote_model, as_json=as_json, repo_id=repo_id, revision=revision, subfolder=subfolder)


@model.command("import")
@click.argument("repo_id", required=False)
@click.option("--project-id", type=click.UUID, help="Target project (required for a new import).")
@click.option("--dataset-id", type=click.UUID, help="Local dataset in the target project (required for a new import).")
@click.option("--name", help="Override the model display name.")
@click.option("--revision", help="Source branch, tag, or commit (default: main).")
@click.option("--subfolder", help="Model root within the repository.")
@click.option("--resume", type=click.UUID, help="Resume a transfer UUID; use only this option and optionally --json.")
@click.option(
    "--dry-run", is_flag=True, help="Read metadata and check local associations/disk space; do not download payload."
)
@click.option("--json", "as_json", is_flag=True, help="Emit one JSON object on stdout; diagnostics go to stderr.")
def import_model(  # noqa: PLR0913, PLR0917
    repo_id: str | None,
    project_id: UUID | None,
    dataset_id: UUID | None,
    name: str | None,
    revision: str | None,
    subfolder: str | None,
    resume: UUID | None,
    dry_run: bool,
    as_json: bool,
) -> None:
    """Download OWNER/REPO and register a new model in local Studio storage.

    A new import requires REPO_ID, --project-id, and --dataset-id. The dataset must
    belong to the target project and match the model's intended environment.
    Supports Studio policy packages, not arbitrary Transformers/checkpoint-only
    repositories. Inspect first and pin its returned commit with --revision.

    Resume uses the saved project, dataset, revision, and name; omit REPO_ID and
    all other transfer options. Preserve staging/cache files between attempts.

    \b
    Examples (replace UUID placeholders):
      physicalai-studio hf model import OWNER/MODEL \\
        --project-id PROJECT_UUID --dataset-id DATASET_UUID --dry-run --json
      physicalai-studio hf model import --resume TRANSFER_UUID --json
    """
    from cli.hf.transfer import import_model as run_import

    invoke(
        run_import,
        as_json=as_json,
        repo_id=repo_id,
        project_id=project_id,
        dataset_id=dataset_id,
        name=name,
        revision=revision,
        subfolder=subfolder,
        resume=resume,
        dry_run=dry_run,
    )


@model.command("export")
@click.argument("repo_id", required=False)
@click.option("--model-id", type=click.UUID, help="Local Studio model UUID (required for a new export).")
@click.option(
    "--public", is_flag=True, help="Create a public repo instead of private; existing visibility is unchanged."
)
@click.option(
    "--update", is_flag=True, help="Authorize overwriting a nonempty repo; stale remote files still cause failure."
)
@click.option("--resume", type=click.UUID, help="Resume a transfer UUID; use only this option and optionally --json.")
@click.option(
    "--dry-run", is_flag=True, help="Check local files and remote destination without creating a repo or uploading."
)
@click.option("--json", "as_json", is_flag=True, help="Emit one JSON object on stdout; diagnostics go to stderr.")
def export_model(
    repo_id: str | None,
    model_id: UUID | None,
    public: bool,
    update: bool,
    resume: UUID | None,
    dry_run: bool,
    as_json: bool,
) -> None:
    """Publish an existing local model to OWNER/REPO without format conversion.

    A new export requires REPO_ID and --model-id; the local record supplies the
    project. New repositories default to private. Existing payloads require
    --update, but stale-file merges are rejected. Review hparams for secrets.

    Do not train or modify the model during upload. Resume uses retained upload
    state and requires unchanged source files. Omit REPO_ID and other options
    when using --resume (except --json).

    \b
    Examples (replace UUID placeholders):
      physicalai-studio hf model export OWNER/MODEL --model-id MODEL_UUID --dry-run --json
      physicalai-studio hf model export --resume TRANSFER_UUID --json
    """
    from cli.hf.transfer import export_model as run_export

    invoke(
        run_export,
        as_json=as_json,
        repo_id=repo_id,
        model_id=model_id,
        public=public,
        update=update,
        resume=resume,
        dry_run=dry_run,
    )
