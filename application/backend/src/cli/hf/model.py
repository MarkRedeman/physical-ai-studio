# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Explicit model commands, independent of dataset command options."""

from uuid import UUID

import click

from cli.hf.output import invoke


@click.group()
def model() -> None:
    """Inspect, import, and export a Studio model on Hugging Face Hub."""


@model.command("inspect")
@click.argument("repo_id")
@click.option("--revision", default="main", show_default=True)
@click.option("--subfolder", help="Model root within the repository.")
@click.option("--json", "as_json", is_flag=True)
def inspect_model(repo_id: str, revision: str, subfolder: str | None, as_json: bool) -> None:
    """Inspect model metadata without downloading weights or checkpoints."""
    from cli.hf.inspection import inspect_remote_model

    invoke(inspect_remote_model, as_json=as_json, repo_id=repo_id, revision=revision, subfolder=subfolder)


@model.command("import")
@click.argument("repo_id", required=False)
@click.option("--project-id", type=click.UUID, help="Target project (required for a new import).")
@click.option("--dataset-id", type=click.UUID, help="Local dataset in the target project (required for a new import).")
@click.option("--name", help="Override the model display name.")
@click.option("--revision", help="Source branch, tag, or commit (default: main).")
@click.option("--subfolder", help="Model root within the repository.")
@click.option("--resume", type=click.UUID, help="Resume a retained model import; omit other transfer options.")
@click.option("--dry-run", is_flag=True)
@click.option("--json", "as_json", is_flag=True)
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
    """Download and register a model in the local Studio database."""
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
@click.option("--public", is_flag=True, help="Create a public repository; new repositories default to private.")
@click.option("--update", is_flag=True, help="Allow overwriting model files in a nonempty destination.")
@click.option("--resume", type=click.UUID, help="Resume a retained model export; omit other transfer options.")
@click.option("--dry-run", is_flag=True)
@click.option("--json", "as_json", is_flag=True)
def export_model(
    repo_id: str | None,
    model_id: UUID | None,
    public: bool,
    update: bool,
    resume: UUID | None,
    dry_run: bool,
    as_json: bool,
) -> None:
    """Publish native model files to the Hub, without a ZIP wrapper."""
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
