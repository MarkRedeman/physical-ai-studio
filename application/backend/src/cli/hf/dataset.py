# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Explicit dataset commands, independent of model command options."""

from uuid import UUID

import click

from cli.hf.output import invoke


@click.group()
def dataset() -> None:
    """Inspect, import, and publish LeRobot v3 datasets on the Hub.

    REPO_ID is OWNER/REPO. Import needs an existing local project, not an environment
    UUID: it creates a temporary empty environment. For local folders, use the
    separate datasets import-dir command, which has different requirements.
    """


@dataset.command("inspect")
@click.argument("repo_id")
@click.option("--revision", default="main", show_default=True, help="Source branch, tag, or commit SHA.")
@click.option("--subfolder", help="Dataset root within the repository.")
@click.option("--json", "as_json", is_flag=True, help="Emit one JSON object on stdout; diagnostics go to stderr.")
def inspect_dataset(repo_id: str, revision: str, subfolder: str | None, as_json: bool) -> None:
    """Inspect OWNER/REPO without downloading Parquet shards or videos.

    Reads bounded JSON metadata and file listings. JSON includes features,
    cameras, size estimate, resolved revision SHA, warnings, and importable.
    Exit code 0 means inspection succeeded, not necessarily importable=true.
    Metadata compatibility does not verify every data row or decode videos.

    \b
    Example:
      physicalai-studio hf dataset inspect OWNER/DATASET --json
    """
    from cli.hf.inspection import inspect_remote_dataset

    invoke(inspect_remote_dataset, as_json=as_json, repo_id=repo_id, revision=revision, subfolder=subfolder)


@dataset.command("import")
@click.argument("repo_id", required=False)
@click.option("--project-id", type=click.UUID, help="Target project (required for a new import).")
@click.option("--name", help="Override the dataset display name.")
@click.option("--revision", help="Source branch, tag, or commit (default: main).")
@click.option("--subfolder", help="Dataset root within the repository.")
@click.option("--resume", type=click.UUID, help="Resume a transfer UUID; use only this option and optionally --json.")
@click.option("--dry-run", is_flag=True, help="Read metadata and check project/disk space; do not download payload.")
@click.option("--json", "as_json", is_flag=True, help="Emit one JSON object on stdout; diagnostics go to stderr.")
def import_dataset(  # noqa: PLR0913, PLR0917
    repo_id: str | None,
    project_id: UUID | None,
    name: str | None,
    revision: str | None,
    subfolder: str | None,
    resume: UUID | None,
    dry_run: bool,
    as_json: bool,
) -> None:
    """Download a LeRobot v3.0 dataset from OWNER/REPO into local Studio storage.

    A new import requires REPO_ID and --project-id. Creates a new dataset and
    temporary empty environment atomically; no robots or cameras are configured.
    No environment or existing dataset UUID is needed. LeRobot v2 conversion is not
    supported. Inspect first and pin its returned commit with --revision.

    Resume uses saved options; omit REPO_ID and other transfer options except
    --json. Preserve staging/cache files to reuse downloaded data.

    \b
    Examples (replace UUID placeholders):
      physicalai-studio hf dataset import OWNER/DATASET --project-id PROJECT_UUID --dry-run --json
      physicalai-studio hf dataset import --resume TRANSFER_UUID --json
    """
    from cli.hf.transfer import import_dataset as run_import

    invoke(
        run_import,
        as_json=as_json,
        repo_id=repo_id,
        project_id=project_id,
        name=name,
        revision=revision,
        subfolder=subfolder,
        resume=resume,
        dry_run=dry_run,
    )


@dataset.command("export")
@click.argument("repo_id", required=False)
@click.option("--dataset-id", type=click.UUID, help="Local Studio dataset UUID (required for a new export).")
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
def export_dataset(
    repo_id: str | None,
    dataset_id: UUID | None,
    public: bool,
    update: bool,
    resume: UUID | None,
    dry_run: bool,
    as_json: bool,
) -> None:
    """Publish an existing local dataset to OWNER/REPO in native LeRobot layout.

    A new export requires REPO_ID and --dataset-id; the local record supplies the
    project. New repositories default to private. Existing payloads require
    --update, but stale shards are not silently merged or deleted.

    Stop dataset recording before export; this CLI does not lock recording.
    Resume requires unchanged source files. Omit REPO_ID and other transfer
    options when using --resume (except --json).

    \b
    Examples (replace UUID placeholders):
      physicalai-studio hf dataset export OWNER/DATASET --dataset-id DATASET_UUID --dry-run --json
      physicalai-studio hf dataset export --resume TRANSFER_UUID --json
    """
    from cli.hf.transfer import export_dataset as run_export

    invoke(
        run_export,
        as_json=as_json,
        repo_id=repo_id,
        dataset_id=dataset_id,
        public=public,
        update=update,
        resume=resume,
        dry_run=dry_run,
    )
