"""Command line interface for interacting with the Physical AI Studio application."""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import click

from cli.database import database
from cli.models import models
from cli.serve import serve


@click.group()
def cli() -> None:
    """Physical AI Studio CLI."""


@cli.command()
@click.option("--target-path", default="docs/openapi.json")
def gen_api(target_path: str) -> None:
    """Generate OpenAPI specification JSON file."""
    # Importing create_openapi imports threading which is slow. Importing here to not slow down other cli commands.
    from create_openapi import create_openapi

    try:
        create_openapi(target_path=target_path)
        click.echo("✓ OpenAPI specification generated successfully!")
    except Exception as e:
        click.echo(f"✗ Failed to generate OpenAPI specification: {e}")
        sys.exit(1)
    click.echo("Waiting for threading to finish...")


@cli.command()
@click.option(
    "--target-dir",
    default=str(Path(__file__).resolve().parents[1] / "static" / "robot-assets"),
    show_default=True,
)
def sync_robot_assets(target_dir: str) -> None:
    """Sync SO101 and WidowX assets into backend static storage."""

    def run_git(args: list[str], cwd: Path | None = None) -> None:
        subprocess.run(["git", *args], cwd=cwd, check=True)

    target_root = Path(target_dir)
    target_root.mkdir(parents=True, exist_ok=True)

    so101_repo_url = "https://github.com/TheRobotStudio/SO-ARM100.git"
    widowx_repo_url = "https://github.com/TrossenRobotics/trossen_arm_description.git"

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)

            so101_repo_path = tmp_root / "so101-repo"
            run_git(
                [
                    "clone",
                    "--depth",
                    "1",
                    "--branch",
                    "main",
                    "--filter=blob:none",
                    "--sparse",
                    so101_repo_url,
                    str(so101_repo_path),
                ]
            )
            run_git(["sparse-checkout", "set", "--no-cone", "Simulation/SO101"], cwd=so101_repo_path)

            so101_target = target_root / "SO101"
            if so101_target.exists():
                shutil.rmtree(so101_target)
            shutil.copytree(so101_repo_path / "Simulation" / "SO101", so101_target)

            widowx_repo_path = tmp_root / "widowx-repo"
            run_git(["clone", "--depth", "1", "--branch", "main", widowx_repo_url, str(widowx_repo_path)])

            widowx_target = target_root / "widowx"
            if widowx_target.exists():
                shutil.rmtree(widowx_target)
            shutil.copytree(widowx_repo_path, widowx_target, ignore=shutil.ignore_patterns(".git"))

        click.echo("✓ Robot assets synced successfully")
    except (subprocess.CalledProcessError, OSError) as error:
        click.echo(f"✗ Failed to sync robot assets: {error}")
        sys.exit(1)


cli.add_command(database)
cli.add_command(models)
cli.add_command(serve)
