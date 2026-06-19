"""Command line interface for interacting with the Physical AI Studio application."""

import sys

import click

from cli.database import check_db, clean_db, init_db, migrate
from cli.models import models


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


@click.group(name="db")
def database() -> None:
    """Database management commands."""


database.add_command(init_db, name="init")
database.add_command(clean_db, name="clean")
database.add_command(check_db, name="check")
database.add_command(migrate)

cli.add_command(database)
cli.add_command(models)
