# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""User-configurable application settings API."""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from api.dependencies import SystemServiceDep
from settings import (
    HotkeySettings,
    HuggingFaceSettings,
    Settings,
    SshProvisioningSettings,
    StreamingSettings,
    TrainerClientSettings,
    get_settings,
    merge_user_settings,
    ssh_patch_to_flat,
)

router = APIRouter(prefix="/api/settings", tags=["Settings"])


class UserSettingsResponse(BaseModel):
    """Effective user-configurable settings, with secrets masked."""

    streaming: StreamingSettings
    trainer: TrainerClientSettings
    huggingface: HuggingFaceSettings
    ssh: SshProvisioningSettings
    hotkeys: HotkeySettings

    @classmethod
    def from_settings(cls, settings: Settings) -> "UserSettingsResponse":
        return cls(
            streaming=settings.streaming,
            trainer=settings.trainer,
            huggingface=settings.huggingface,
            ssh=settings.ssh,
            hotkeys=settings.hotkeys,
        )


class SettingsUpdate(BaseModel):
    """Partial update for user-configurable application settings."""

    streaming: StreamingSettings | None = None
    trainer: TrainerClientSettings | None = None
    huggingface: HuggingFaceSettings | None = None
    ssh: SshProvisioningSettings | None = None
    hotkeys: HotkeySettings | None = None


@router.get("")
async def get_user_settings() -> UserSettingsResponse:
    """Get the effective user-configurable settings."""
    return UserSettingsResponse.from_settings(get_settings())


@router.patch("")
async def update_user_settings(update: SettingsUpdate, system_service: SystemServiceDep) -> UserSettingsResponse:
    """Persist the supplied fields and return effective settings."""
    patch = update.model_dump(exclude_unset=True)
    if streaming := patch.get("streaming"):
        profile_id = streaming.get("profile")
        available_profile_ids = {profile.id for profile in system_service.get_streaming_profiles().profiles}
        if profile_id is not None and profile_id not in available_profile_ids:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"Streaming profile '{profile_id}' is unavailable on this backend.",
            )
    if "ssh" in patch:
        # The SSH knobs are grouped for the API and the UI but stored flat,
        # because the services read them off `Settings` directly and each one
        # keeps its own documented `SSH_*` environment alias.
        patch.update(ssh_patch_to_flat(patch.pop("ssh")))
    merge_user_settings(patch)
    return UserSettingsResponse.from_settings(get_settings())
