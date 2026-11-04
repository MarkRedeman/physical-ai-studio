"""Validated home poses and the short commands controlling a supervised return."""

from __future__ import annotations

from datetime import datetime  # noqa: TC003 - Pydantic resolves annotations at runtime
from typing import Literal
from uuid import UUID  # noqa: TC003 - Pydantic resolves annotations at runtime

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat


class HomeJointLimits(BaseModel):
    minimum: FiniteFloat
    maximum: FiniteFloat
    velocity: FiniteFloat = Field(gt=0)
    acceleration: FiniteFloat = Field(gt=0)
    tolerance: FiniteFloat = Field(gt=0)
    tracking_error: FiniteFloat = Field(gt=0)


class HomeCapability(BaseModel):
    robot_id: UUID
    fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    unit: Literal["normalized", "degrees"]
    joints: dict[str, HomeJointLimits]
    preserved_joints: list[str] = Field(default_factory=lambda: ["gripper.pos"])
    lease_timeout_s: float = Field(default=1.5, gt=0, le=5)


class HomePose(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    version: Literal[1] = 1
    robot_id: UUID
    fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    unit: Literal["normalized", "degrees"]
    positions: dict[str, FiniteFloat] = Field(min_length=1, max_length=32)
    captured_at: datetime


class HomeMotion(BaseModel):
    operation_id: UUID
    robot_id: UUID
    state: Literal["moving", "reached", "cancelled", "failed"]
    progress: float = Field(default=0, ge=0, le=1)
    message: str | None = None
    sequence: int = Field(default=0, ge=0)


class CaptureHomePositionCommand(BaseModel):
    command: Literal["capture_home_position"] = "capture_home_position"
    request_id: str
    robot_id: UUID


class ReturnHomeCommand(BaseModel):
    command: Literal["return_home"] = "return_home"
    request_id: str
    operation_id: UUID
    home: HomePose
    path_confirmed: Literal[True]


class RenewHomeReturnCommand(BaseModel):
    command: Literal["renew_home_return"] = "renew_home_return"
    request_id: str
    operation_id: UUID


class CancelHomeReturnCommand(BaseModel):
    command: Literal["cancel_home_return"] = "cancel_home_return"
    request_id: str
    operation_id: UUID
