from fastapi import APIRouter
from pydantic import BaseModel

from settings import (
    HuggingFaceSettings,
    LoggerSettings,
    Settings,
    StreamingSettings,
    TrainerClientSettings,
    get_settings,
    merge_user_settings,
)

router = APIRouter(prefix="/api/settings", tags=["Settings"])


class UserSettingsResponse(BaseModel):
    """Effective user-configurable settings.

    Secrets (HF token, W&B API key) are serialized masked — the plaintext is
    never returned by the API.
    """

    geti_action_dataset_path: str
    streaming: StreamingSettings
    trainer: TrainerClientSettings
    huggingface: HuggingFaceSettings
    logger: LoggerSettings

    @classmethod
    def from_settings(cls, settings: Settings) -> "UserSettingsResponse":
        return cls(
            geti_action_dataset_path=str(settings.datasets_dir),
            streaming=settings.streaming,
            trainer=settings.trainer,
            huggingface=settings.huggingface,
            logger=settings.logger,
        )


class SettingsUpdate(BaseModel):
    """The user-configurable subset of application settings.

    A partial update: only the groups and fields present are applied to
    ``settings.json``; everything omitted keeps its current value. Within a
    group, an explicit ``null`` clears the field (e.g. revoke a token).
    Secrets sent here are persisted to ``settings.json`` in plaintext.
    """

    streaming: StreamingSettings | None = None
    trainer: TrainerClientSettings | None = None
    huggingface: HuggingFaceSettings | None = None
    logger: LoggerSettings | None = None


@router.get("")
async def get_user_settings() -> UserSettingsResponse:
    """Get the effective user-configurable settings."""
    return UserSettingsResponse.from_settings(get_settings())


@router.patch("")
async def update_user_settings(update: SettingsUpdate) -> UserSettingsResponse:
    """Persist only the provided settings and return the effective values."""
    merge_user_settings(update.model_dump(exclude_unset=True))
    return UserSettingsResponse.from_settings(get_settings())
