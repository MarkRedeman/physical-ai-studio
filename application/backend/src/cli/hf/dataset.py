# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Explicit dataset commands, independent of model command options."""

from uuid import UUID

import click

from cli.hf.output import invoke


@click.group()
def dataset() -> None:
    """Inspect, import, and export a LeRobot dataset on Hugging Face Hub."""


@dataset.command("inspect")
@click.argument("repo_id")
@click.option("--revision", default="main", show_default=True)
@click.option("--subfolder", help="Dataset root within the repository.")
@click.option("--json", "as_json", is_flag=True)
def inspect_dataset(repo_id: str, revision: str, subfolder: str | None, as_json: bool) -> None:
    """Inspect dataset metadata without downloading Parquet shards or videos."""
    from cli.hf.inspection import inspect_remote_dataset

    invoke(inspect_remote_dataset, as_json=as_json, repo_id=repo_id, revision=revision, subfolder=subfolder)


@dataset.command("import")
@click.argument("repo_id", required=False)
@click.option("--project-id", type=click.UUID, help="Target project (required for a new import).")
@click.option("--name", help="Override the dataset display name.")
@click.option("--revision", help="Source branch, tag, or commit (default: main).")
@click.option("--subfolder", help="Dataset root within the repository.")
@click.option("--resume", type=click.UUID, help="Resume a retained dataset import; omit other transfer options.")
@click.option("--dry-run", is_flag=True)
@click.option("--json", "as_json", is_flag=True)
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
    """Download and register a dataset with an empty placeholder environment."""
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
@click.option("--public", is_flag=True, help="Create a public repository; new repositories default to private.")
@click.option("--update", is_flag=True, help="Allow overwriting dataset files in a nonempty destination.")
@click.option("--resume", type=click.UUID, help="Resume a retained dataset export; omit other transfer options.")
@click.option("--dry-run", is_flag=True)
@click.option("--json", "as_json", is_flag=True)
def export_dataset(
    repo_id: str | None,
    dataset_id: UUID | None,
    public: bool,
    update: bool,
    resume: UUID | None,
    dry_run: bool,
    as_json: bool,
) -> None:
    """Publish native dataset files to the Hub, without a ZIP wrapper."""
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
