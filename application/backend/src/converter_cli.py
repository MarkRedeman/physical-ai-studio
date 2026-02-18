# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""CLI tools for converting between LeRobot v3 and raw-video dataset formats.

Usage::

    # LeRobot v3 -> Raw Video
    uv run src/converter_cli.py convert-to-raw --from ./lerobot_dataset --to ./raw_dataset

    # Raw Video -> LeRobot v3
    uv run src/converter_cli.py convert-to-lerobot --from ./raw_dataset --to ./lerobot_dataset

    # With custom robot type
    uv run src/converter_cli.py convert-to-lerobot --from ./raw_dataset --to ./lerobot_dataset --robot-type so100
"""

import sys
from pathlib import Path

import click


@click.group()
def cli() -> None:
    """Dataset format conversion tools for Geti Action."""


@cli.command("convert-to-raw")
@click.option(
    "--from",
    "from_path",
    required=True,
    type=click.Path(exists=True, file_okay=False, resolve_path=True, path_type=Path),
    help="Path to the source LeRobot v3 dataset.",
)
@click.option(
    "--to",
    "to_path",
    required=True,
    type=click.Path(resolve_path=True, path_type=Path),
    help="Path to write the raw-video dataset (must not exist).",
)
def convert_to_raw(from_path: Path, to_path: Path) -> None:
    """Convert a LeRobot v3 dataset to raw-video format."""
    from internal_datasets.raw_video.converters import LeRobotToRawVideoConverter

    try:
        converter = LeRobotToRawVideoConverter(source=from_path, dest=to_path)
        converter.convert()
    except (FileNotFoundError, FileExistsError) as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    except Exception as exc:
        click.echo(f"Conversion failed: {exc}", err=True)
        sys.exit(1)


@cli.command("convert-to-lerobot")
@click.option(
    "--from",
    "from_path",
    required=True,
    type=click.Path(exists=True, file_okay=False, resolve_path=True, path_type=Path),
    help="Path to the source raw-video dataset.",
)
@click.option(
    "--to",
    "to_path",
    required=True,
    type=click.Path(resolve_path=True, path_type=Path),
    help="Path to write the LeRobot v3 dataset (must not exist).",
)
@click.option(
    "--robot-type",
    default="unknown",
    show_default=True,
    help="Robot type for the LeRobot dataset metadata.",
)
def convert_to_lerobot(from_path: Path, to_path: Path, robot_type: str) -> None:
    """Convert a raw-video dataset to LeRobot v3 format."""
    from internal_datasets.raw_video.converters import RawVideoToLeRobotConverter

    try:
        converter = RawVideoToLeRobotConverter(
            source=from_path,
            dest=to_path,
            robot_type=robot_type,
        )
        converter.convert()
    except (FileNotFoundError, FileExistsError) as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    except Exception as exc:
        click.echo(f"Conversion failed: {exc}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
