# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Application configuration management.

Settings come from (highest precedence first):

1. ``settings.json`` — a JSON config file under the storage directory (or
   ``SETTINGS_FILE``), editable by hand or through the settings API/UI.
2. Environment variables / ``.env`` — for the flat, environment-only fields
   below (server, storage, database, ...).
3. Field defaults.

The user-configurable groups (``streaming``, ``trainer``, ``huggingface``,
``logger``) are configured exclusively through ``settings.json`` (and the
settings API/UI); they are not read from environment variables. There is no
migration of legacy flat settings formats.
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, JsonConfigSettingsSource, PydanticBaseSettingsSource, SettingsConfigDict


def get_default_storage_dir() -> Path:
    """Return the platform-appropriate directory for persistent app data."""
    if sys.platform == "darwin":
        return Path("~/Library/Application Support/physicalai").expanduser()

    if sys.platform == "win32":
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            return Path(local_app_data).expanduser() / "physicalai"
        return Path("~/AppData/Local/physicalai").expanduser()

    xdg_data_home = os.environ.get("XDG_DATA_HOME")
    if xdg_data_home:
        xdg_data_home_path = Path(xdg_data_home).expanduser()
        if xdg_data_home_path.is_absolute():
            return xdg_data_home_path / "physicalai"

    return Path("~/.local/share/physicalai").expanduser()


def get_settings_file_path() -> Path:
    """Return the path to the user-configurable ``settings.json`` file.

    Overridable with the ``SETTINGS_FILE`` environment variable; defaults to a
    ``settings.json`` next to the application's persistent data.
    """
    override = os.environ.get("SETTINGS_FILE")
    if override:
        return Path(override).expanduser()
    return get_default_storage_dir() / "settings.json"


class StreamingSettings(BaseModel):
    """Streaming video-encoding settings for dataset recordings."""

    vcodec: str = Field(default="auto", description="FFmpeg video codec, or 'auto' to probe for one.")
    pix_fmt: str | None = Field(default=None, description="Pixel format; None lets the encoder pick.")
    crf: int | float | None = Field(default=None, description="Constant-rate-factor (quality).")
    preset: int | str | None = Field(default=None, description="Encoder preset.")
    extra_options: dict[str, Any] | None = Field(default=None, description="Extra FFmpeg encoder options.")
    encoder_threads: int | None = Field(default=None, description="Threads for the video encoder.")
    encoder_queue_maxsize: int = Field(default=60, description="Max queued frames for the encoder.")


class TrainerClientSettings(BaseModel):
    """Client-side timeouts for talking to a remote trainer service."""

    # Seconds to wait for trainer HTTP requests (excludes long-poll/SSE streams).
    request_timeout_s: float = Field(default=30.0)
    # Seconds to wait between chunks while streaming the model artifact. A stalled
    # transfer (e.g. a proxy holding the connection open) must fail instead of
    # hanging the job forever; this is a per-read gap, not a total transfer cap.
    download_read_timeout_s: float = Field(default=120.0)
    # Stop reconnecting after this continuous trainer outage.
    stream_reconnect_max_s: float = Field(default=900.0)
    # Upper bound on the exponential backoff between event-stream reconnect attempts.
    stream_reconnect_backoff_max_s: float = Field(default=30.0)


class HuggingFaceSettings(BaseModel):
    """Hugging Face credentials used when training pulls pretrained assets."""

    hf_token: SecretStr | None = Field(default=None, description="Hugging Face token for authenticated downloads.")


class LoggerSettings(BaseModel):
    """Training-run logging (Lightning logger) configuration."""

    providers: list[Literal["csv", "tensorboard", "wandb"]] = Field(
        default=["csv"],
        description="Loggers to enable for training runs. Multiple providers run simultaneously.",
    )
    wandb_project: str | None = Field(default=None, description="W&B project name when 'wandb' is in providers.")
    wandb_entity: str | None = Field(default=None, description="W&B entity/team when 'wandb' is in providers.")
    wandb_api_key: SecretStr | None = Field(default=None, description="W&B API key when 'wandb' is in providers.")

    @field_validator("providers", mode="after")
    @classmethod
    def dedupe_providers(cls, value: list[str]) -> list[str]:
        """Remove duplicates, preserving order, and treat an empty list as CSV."""
        deduped = list(dict.fromkeys(value))
        if not deduped:
            deduped = ["csv"]
        return deduped


#: Nested settings groups that are user-configurable via settings.json / the API.
_USER_CONFIG_GROUPS: tuple[str, ...] = ("streaming", "trainer", "huggingface", "logger")


class UserConfigSettingsSource(JsonConfigSettingsSource):
    """JSON settings source restricted to the user-configurable groups."""

    def __call__(self) -> dict[str, Any]:
        data: dict[str, Any] = super().__call__()
        return {key: value for key, value in data.items() if key in _USER_CONFIG_GROUPS}


class Settings(BaseSettings):
    """Application settings with environment variable support"""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore")

    # Application
    app_name: str = "Physical AI Studio"
    version: str = "0.1.0"
    summary: str = "Physical AI Studio server"
    description: str = (
        "Physical AI Studio is a framework to train robots. It allows the user to create datasets, "
        "models and the run inference."
    )
    openapi_url: str = "/api/openapi.json"
    debug: bool = Field(default=False, alias="DEBUG")
    environment: Literal["dev", "prod"] = "dev"
    storage_dir: Path = Field(default_factory=get_default_storage_dir, alias="STORAGE_DIR")
    static_files_dir: str | None = Field(default=None, alias="STATIC_FILES_DIR")

    @field_validator("storage_dir", mode="before")
    @classmethod
    def expand_storage_dir(cls, value: Path | str) -> Path:
        """Expand user-provided storage directories like ~/.local/share."""
        return Path(value).expanduser()

    # Data import/upload safety (shared for dataset/model/project imports)
    data_import_max_uncompressed_bytes: int = Field(
        default=200 * 1024 * 1024 * 1024,
        alias="DATA_IMPORT_MAX_UNCOMPRESSED_BYTES",
    )
    # Maximum raw upload size (Content-Length) accepted before any processing.
    # Default: 100 GiB - supports large dataset imports while still guarding abuse.
    data_import_max_upload_bytes: int = Field(
        default=100 * 1024 * 1024 * 1024,
        alias="DATA_IMPORT_MAX_UPLOAD_BYTES",
    )
    # Minimum free bytes that must remain on the target filesystem after the
    # upload / extraction lands.  Default: 1 GiB headroom.
    data_import_min_free_bytes: int = Field(
        default=1 * 1024 * 1024 * 1024,
        alias="DATA_IMPORT_MIN_FREE_BYTES",
    )

    @property
    def datasets_dir(self) -> Path:
        """Storage directory for datasets."""
        return self.storage_dir / "datasets"

    @property
    def data_dir(self) -> Path:
        """Storage directory for application data."""
        return self.storage_dir / "data"

    @property
    def snapshot_dir(self) -> Path:
        """Storage directory for snapshots."""
        return self.storage_dir / "snapshots"

    @property
    def cache_dir(self) -> Path:
        """Storage directory for cache."""
        return self.storage_dir / "cache"

    @property
    def models_dir(self) -> Path:
        """Storage directory for models."""
        return self.storage_dir / "models"

    @property
    def robots_dir(self) -> Path:
        """Storage directory for robots."""
        return self.storage_dir / "robots"

    @property
    def log_dir(self) -> Path:
        """Storage directory for logs."""
        return self.storage_dir / "logs"

    # User-configurable groups (editable via settings.json / the settings API).
    # Streaming video encoding.
    streaming: StreamingSettings = StreamingSettings()
    # Client-side timeouts for the remote trainer connection.
    trainer: TrainerClientSettings = TrainerClientSettings()
    # Hugging Face credentials.
    huggingface: HuggingFaceSettings = HuggingFaceSettings()
    # Training-run logging.
    logger: LoggerSettings = LoggerSettings()

    # Server
    host: str = Field(default="0.0.0.0", alias="HOST")  # noqa: S104 # nosec B104
    port: int = Field(default=7860, alias="PORT")

    # Database
    database_file: str = Field(default="physicalai.db", alias="DATABASE_FILE", description="Database filename")
    db_echo: bool = Field(default=False, alias="DB_ECHO")
    allow_unknown_db_revision: bool = Field(default=False, alias="ALLOW_UNKNOWN_DB_REVISION")

    # Alembic
    alembic_config_path: str = Field(default="src/alembic.ini", alias="ALEMBIC_CONFIG_PATH")
    alembic_script_location: str = Field(default="src/alembic", alias="ALEMBIC_SCRIPT_LOCATION")

    # Proxy settings
    no_proxy: str = Field(default="localhost,127.0.0.1,::1", alias="no_proxy")

    @property
    def database_url(self) -> str:
        """Get database URL"""
        return f"sqlite+aiosqlite:///{self.data_dir / self.database_file}"

    @property
    def database_url_sync(self) -> str:
        """Get synchronous database URL"""
        return f"sqlite:///{self.data_dir / self.database_file}"

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Wire the settings sources: init kwargs > settings.json > env > .env.

        The JSON config file is read on every construction so UI changes are
        picked up at point of use without a restart.
        """
        return (
            init_settings,
            UserConfigSettingsSource(settings_cls, json_file=get_settings_file_path()),
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )


def write_user_settings(data: dict[str, Any]) -> None:
    """Atomically persist user-configurable settings to ``settings.json``.

    Only the user-configurable groups are written; unknown keys are dropped.
    The write is atomic (temp file + rename) so a crash never leaves a
    truncated file behind.
    """
    path = get_settings_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    filtered = {key: value for key, value in data.items() if key in _USER_CONFIG_GROUPS and value is not None}
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix="settings.", suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
            json.dump(filtered, tmp_file, indent=2, default=_plain_secret_str)
            tmp_file.write("\n")
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _plain_secret_str(value: Any) -> Any:
    """Serialize ``SecretStr`` values as their plaintext for the config file."""
    if isinstance(value, SecretStr):
        return value.get_secret_value()
    return value


def load_user_settings_file() -> dict[str, Any]:
    """Return the raw ``settings.json`` contents as a dict of groups.

    Returns ``{}`` when the file is missing or unreadable, so merge operations
    degrade gracefully instead of failing on a corrupt file.
    """
    path = get_settings_file_path()
    if not path.exists():
        return {}
    try:
        with path.open(encoding="utf-8") as settings_file:
            data = json.load(settings_file)
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def merge_user_settings(patch: dict[str, Any]) -> None:
    """Apply a partial patch to ``settings.json``, keeping unspecified values.

    Only the groups and fields present in ``patch`` are touched:
    * a missing group keeps its current value;
    * a group mapped to ``None`` removes that group entirely;
    * a field mapped to ``None`` clears it;
    * anything else overwrites the field.
    Unknown keys are dropped.
    """
    current = load_user_settings_file()
    for group in _USER_CONFIG_GROUPS:
        if group not in patch:
            continue
        value = patch[group]
        if value is None:
            current.pop(group, None)
        elif isinstance(value, dict):
            current.setdefault(group, {}).update(value)
    write_user_settings(current)


def get_settings() -> Settings:
    """Return the application settings, freshly resolved from all sources.

    Not cached: the settings file is small, and consumers call this at point of
    use so that changes made through the settings API take effect without a
    restart (including in worker processes).
    """
    return Settings()
