import json
from pathlib import Path

import settings as settings_module
from settings import (
    Settings,
    get_default_storage_dir,
    get_settings,
    get_settings_file_path,
    load_user_settings_file,
    merge_user_settings,
    write_user_settings,
)


def test_default_storage_dir_uses_xdg_data_home(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(settings_module.sys, "platform", "linux")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg-data"))

    assert get_default_storage_dir() == tmp_path / "xdg-data" / "physicalai"


def test_default_storage_dir_ignores_relative_xdg_data_home(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(settings_module.sys, "platform", "linux")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_DATA_HOME", "relative/path")

    assert get_default_storage_dir() == tmp_path / ".local" / "share" / "physicalai"


def test_default_storage_dir_uses_macos_application_support(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(settings_module.sys, "platform", "darwin")
    monkeypatch.setenv("HOME", str(tmp_path))

    assert get_default_storage_dir() == tmp_path / "Library" / "Application Support" / "physicalai"


def test_storage_dir_override_expands_user(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    settings = Settings(STORAGE_DIR="~/custom-storage")

    assert settings.storage_dir == tmp_path / "custom-storage"


def test_data_dir_is_storage_backed_even_with_data_dir_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    custom_data_dir = tmp_path / "custom-data"
    monkeypatch.setenv("DATA_DIR", str(custom_data_dir))

    settings = Settings(STORAGE_DIR="~/custom-storage")

    assert settings.data_dir == tmp_path / "custom-storage" / "data"


def test_settings_file_path_defaults_under_storage_dir(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("SETTINGS_FILE", raising=False)

    assert get_settings_file_path() == get_default_storage_dir() / "settings.json"


def test_json_settings_file_sets_group_values(monkeypatch, tmp_path: Path) -> None:
    settings_file = tmp_path / "settings.json"
    settings_file.write_text(
        json.dumps(
            {
                "streaming": {"vcodec": "libsvtav1", "crf": 23},
                "trainer": {"request_timeout_s": 45.0},
                "huggingface": {"hf_token": "secret-token"},
                "logger": {"providers": ["csv", "wandb"], "wandb_project": "studio"},
            }
        )
    )
    monkeypatch.setenv("SETTINGS_FILE", str(settings_file))

    settings = Settings()

    assert settings.streaming.vcodec == "libsvtav1"
    assert settings.streaming.crf == 23
    assert settings.trainer.request_timeout_s == 45.0
    assert settings.huggingface.hf_token is not None
    assert settings.huggingface.hf_token.get_secret_value() == "secret-token"
    assert settings.logger.providers == ["csv", "wandb"]
    assert settings.logger.wandb_project == "studio"


def test_group_values_are_not_read_from_env(monkeypatch, tmp_path: Path) -> None:
    """Grouped settings are configured only via settings.json / the API."""
    monkeypatch.setenv("SETTINGS_FILE", str(tmp_path / "settings.json"))
    monkeypatch.setenv("STREAMING_VCODEC", "libx264")
    monkeypatch.setenv("HF_TOKEN", "secret-token")
    monkeypatch.setenv("LOGGER_PROVIDER", "wandb")

    settings = Settings()

    assert settings.streaming.vcodec == "auto"
    assert settings.huggingface.hf_token is None
    assert settings.logger.providers == ["csv"]


def test_json_settings_partial_group_merges_with_defaults(monkeypatch, tmp_path: Path) -> None:
    settings_file = tmp_path / "settings.json"
    settings_file.write_text('{"streaming": {"vcodec": "libx264"}}')
    monkeypatch.setenv("SETTINGS_FILE", str(settings_file))

    settings = Settings()

    assert settings.streaming.vcodec == "libx264"
    assert settings.streaming.encoder_queue_maxsize == 60
    assert settings.streaming.crf is None


def test_missing_settings_file_is_a_no_op(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SETTINGS_FILE", str(tmp_path / "does-not-exist.json"))

    settings = Settings()

    assert settings.streaming.vcodec == "auto"
    assert settings.logger.providers == ["csv"]


def test_write_user_settings_persists_groups_and_drops_unknown_keys(monkeypatch, tmp_path: Path) -> None:
    settings_file = tmp_path / "settings.json"
    monkeypatch.setenv("SETTINGS_FILE", str(settings_file))

    write_user_settings(
        {
            "streaming": {"vcodec": "libx264", "crf": 23},
            "logger": {"providers": ["csv", "wandb"], "wandb_project": "prod", "wandb_api_key": "key-xyz"},
            "host": "should-be-dropped",
            "huggingface": None,
        }
    )

    assert settings_file.exists()
    settings = Settings()
    assert settings.streaming.vcodec == "libx264"
    assert settings.streaming.crf == 23
    assert settings.logger.providers == ["csv", "wandb"]
    assert settings.logger.wandb_api_key is not None
    assert settings.logger.wandb_api_key.get_secret_value() == "key-xyz"
    # Env-only fields are not writable via the settings file.
    assert settings.host == "0.0.0.0"
    assert settings.huggingface.hf_token is None


def test_get_settings_is_fresh_after_write(monkeypatch, tmp_path: Path) -> None:
    settings_file = tmp_path / "settings.json"
    monkeypatch.setenv("SETTINGS_FILE", str(settings_file))

    before = get_settings()
    assert before.streaming.vcodec == "auto"

    write_user_settings({"streaming": {"vcodec": "libx264"}})

    after = get_settings()
    assert after.streaming.vcodec == "libx264"


def test_write_user_settings_is_atomic(monkeypatch, tmp_path: Path) -> None:
    settings_file = tmp_path / "settings.json"
    settings_file.write_text('{"trainer": {"request_timeout_s": 30.0}}')
    monkeypatch.setenv("SETTINGS_FILE", str(settings_file))

    write_user_settings({"trainer": {"request_timeout_s": 55.0}})

    # Only a single file remains (no leftover temp files).
    assert sorted(p.name for p in tmp_path.iterdir()) == ["settings.json"]
    assert Settings().trainer.request_timeout_s == 55.0


def test_merge_user_settings_keeps_untouched_groups_and_fields(monkeypatch, tmp_path: Path) -> None:
    settings_file = tmp_path / "settings.json"
    monkeypatch.setenv("SETTINGS_FILE", str(settings_file))
    write_user_settings(
        {
            "streaming": {"vcodec": "libx264", "crf": 23},
            "logger": {"providers": ["wandb"], "wandb_api_key": "key-xyz"},
        }
    )

    merge_user_settings({"streaming": {"encoder_threads": 4}})

    settings = Settings()
    assert settings.streaming.encoder_threads == 4
    # Untouched fields/groups are preserved.
    assert settings.streaming.vcodec == "libx264"
    assert settings.streaming.crf == 23
    assert settings.logger.providers == ["wandb"]
    assert settings.logger.wandb_api_key is not None


def test_merge_user_settings_clears_field_and_removes_group_with_null(monkeypatch, tmp_path: Path) -> None:
    settings_file = tmp_path / "settings.json"
    monkeypatch.setenv("SETTINGS_FILE", str(settings_file))
    write_user_settings({"huggingface": {"hf_token": "secret"}})

    # Explicit null clears a field within a group.
    merge_user_settings({"huggingface": {"hf_token": None}})
    assert Settings().huggingface.hf_token is None

    # A null group removes that group entirely (falls back to defaults).
    merge_user_settings({"huggingface": None})
    assert "huggingface" not in load_user_settings_file()


def test_load_user_settings_file_handles_missing_and_corrupt(monkeypatch, tmp_path: Path) -> None:
    settings_file = tmp_path / "settings.json"
    monkeypatch.setenv("SETTINGS_FILE", str(settings_file))

    assert load_user_settings_file() == {}

    settings_file.write_text("{not valid json")
    assert load_user_settings_file() == {}

    write_user_settings({"streaming": {"vcodec": "libx264"}})
    assert load_user_settings_file() == {"streaming": {"vcodec": "libx264"}}
