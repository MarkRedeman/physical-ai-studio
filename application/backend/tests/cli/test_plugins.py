import importlib
from unittest.mock import AsyncMock, MagicMock

from click.testing import CliRunner

from cli import cli

plugins_command = importlib.import_module("cli.plugins")


def test_restore_uses_configured_storage_and_reports_restored_plugins(monkeypatch, tmp_path) -> None:
    plugin_manager = MagicMock()
    plugin_manager.restore_installed = AsyncMock(return_value=["demo-plugin"])
    manager_factory = MagicMock(return_value=plugin_manager)
    settings = MagicMock(storage_dir=tmp_path)
    monkeypatch.setattr(plugins_command, "PluginManager", manager_factory)
    monkeypatch.setattr(plugins_command, "get_settings", lambda: settings)

    result = CliRunner().invoke(cli, ["plugins", "restore"])

    assert result.exit_code == 0
    assert "Restored plugins: demo-plugin" in result.output
    manager_factory.assert_called_once_with(record_path=tmp_path / "installed-plugins.json")
    plugin_manager.restore_installed.assert_awaited_once_with()


def test_restore_reports_when_no_plugins_need_restoring(monkeypatch, tmp_path) -> None:
    plugin_manager = MagicMock()
    plugin_manager.restore_installed = AsyncMock(return_value=[])
    monkeypatch.setattr(plugins_command, "PluginManager", lambda **_kwargs: plugin_manager)
    monkeypatch.setattr(plugins_command, "get_settings", lambda: MagicMock(storage_dir=tmp_path))

    result = CliRunner().invoke(cli, ["plugins", "restore"])

    assert result.exit_code == 0
    assert "already installed" in result.output


def test_restore_warns_and_continues_on_unexpected_failure(monkeypatch, tmp_path) -> None:
    plugin_manager = MagicMock()
    plugin_manager.restore_installed = AsyncMock(side_effect=RuntimeError("offline"))
    monkeypatch.setattr(plugins_command, "PluginManager", lambda **_kwargs: plugin_manager)
    monkeypatch.setattr(plugins_command, "get_settings", lambda: MagicMock(storage_dir=tmp_path))

    result = CliRunner().invoke(cli, ["plugins", "restore"])

    assert result.exit_code == 0
    assert "could not restore" in result.output
