# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""CLI-local Hugging Face inspection and resumable transfers."""

import json
from typing import Any

import click


def _output(value: dict[str, Any], *, as_json: bool) -> None:
    if as_json:
        click.echo(json.dumps(value, indent=2))
        return
    for key, item in value.items():
        click.echo(f"{key}: {json.dumps(item) if isinstance(item, (dict, list)) else item}")


def _invoke(callback: Any, *, as_json: bool, **kwargs: Any) -> None:
    try:
        result = callback(**kwargs)
    except (Exception, KeyboardInterrupt) as exc:
        # SDK exceptions can contain request details. Never echo credentials.
        from cli.hf.inspection import redact

        message = redact(str(exc) or "Transfer interrupted")
        if as_json:
            error = {"code": "hf_operation_failed", "message": message}
            if hasattr(exc, "resume_command"):
                error.update(resume_command=exc.resume_command, transfer_id=getattr(exc, "transfer_id", ""))
            click.echo(json.dumps({"error": error}))
            raise click.exceptions.Exit(1) from None
        raise click.ClickException(message) from None
    _output(result, as_json=as_json)


def _resource_group(kind: str) -> click.Group:
    @click.group(name=kind)
    def resource() -> None:
        """Inspect, import, and export a single Hub resource."""

    @resource.command("inspect")
    @click.argument("repo_id")
    @click.option("--revision", default="main", show_default=True)
    @click.option("--subfolder", default=None, help="Asset root within the repository.")
    @click.option("--json", "as_json", is_flag=True, help="Emit structured JSON on stdout.")
    def inspect_resource(repo_id: str, revision: str, subfolder: str | None, as_json: bool) -> None:
        """Inspect metadata without downloading weights or videos."""
        from cli.hf.inspection import inspect_remote

        _invoke(inspect_remote, as_json=as_json, kind=kind, repo_id=repo_id, revision=revision, subfolder=subfolder)

    @resource.command("import")
    @click.argument("repo_id", required=False)
    @click.option("--project-id", type=click.UUID, help="Target Studio project (required for a new transfer).")
    @click.option("--dataset-id", type=click.UUID, hidden=kind == "dataset", help="Local dataset association.")
    @click.option("--name", help="Override the imported display name.")
    @click.option("--revision", default=None, help="Source branch, tag, or commit (default: main).")
    @click.option("--subfolder", default=None)
    @click.option("--resume", type=click.UUID, help="Resume a retained transfer; omit other transfer options.")
    @click.option("--dry-run", is_flag=True, help="Inspect and validate without transferring or registering an asset.")
    @click.option("--json", "as_json", is_flag=True)
    def import_resource(as_json: bool, **options: Any) -> None:
        """Download and register an asset in the local Studio storage/database."""
        from cli.hf.transfer import import_asset

        _invoke(import_asset, as_json=as_json, kind=kind, **options)

    @resource.command("export")
    @click.argument("repo_id", required=False)
    @click.option(f"--{kind}-id", "asset_id", type=click.UUID, help="Local Studio resource UUID.")
    @click.option("--public", is_flag=True, help="Create a public repository (new repositories default to private).")
    @click.option("--update", is_flag=True, help="Allow overwriting files in a nonempty destination.")
    @click.option("--resume", type=click.UUID, help="Resume a retained transfer; omit other transfer options.")
    @click.option("--dry-run", is_flag=True)
    @click.option("--json", "as_json", is_flag=True)
    def export_resource(as_json: bool, **options: Any) -> None:
        """Publish native Studio files to a Hub repository, without a ZIP wrapper."""
        from cli.hf.transfer import export_asset

        _invoke(export_asset, as_json=as_json, kind=kind, **options)

    return resource


@click.group()
def hf() -> None:
    """Inspect and transfer Studio assets on Hugging Face Hub.

    Commands operate on local Studio storage, not a remote Studio HTTP server.
    Use the same STORAGE_DIR and SETTINGS_FILE as the backend.
    """


hf.add_command(_resource_group("model"))
hf.add_command(_resource_group("dataset"))
