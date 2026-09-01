"""Plugin maintenance CLI commands."""

import asyncio

import click

from plugins.plugin_manager import PluginManager
from settings import get_settings


@click.group()
def plugins() -> None:
    """Manage installed robot plugins."""


@plugins.command()
def restore() -> None:
    """Restore recorded plugins that are missing from the active environment."""
    settings = get_settings()
    manager = PluginManager(record_path=settings.storage_dir / "installed-plugins.json")

    try:
        restored = asyncio.run(manager.restore_installed())
    except Exception:
        # Restoration is best effort. Do not prevent callers such as Docker's
        # startup entrypoint from continuing to start Studio.
        click.echo("Warning: could not restore all recorded plugins; continuing.")
        return

    if restored:
        click.echo(f"Restored plugins: {', '.join(restored)}")
    else:
        click.echo("All recorded plugins are already installed.")
