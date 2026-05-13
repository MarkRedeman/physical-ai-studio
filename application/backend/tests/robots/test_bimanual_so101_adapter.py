# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from robots.so101.bimanual_adapter import BimanualSO101Adapter
from schemas.robot import RobotType


def _make_arm(features: list[str]) -> MagicMock:
    arm = MagicMock()
    arm.features.return_value = features
    arm.is_connected = True
    arm.is_controlled = True
    arm.connect = AsyncMock()
    arm.disconnect = AsyncMock()
    arm.enable_torque = AsyncMock(return_value={"event": "torque_was_enabled"})
    arm.disable_torque = AsyncMock(return_value={"event": "torque_was_disabled"})
    arm.set_joints_state = AsyncMock(return_value={"event": "joints_state_was_set"})
    arm.set_forces = AsyncMock(return_value={})
    arm.read_forces = AsyncMock(return_value={"event": "force_was_updated", "state": None})
    arm.read_state = AsyncMock(
        return_value={
            "event": "state_was_updated",
            "state": {"shoulder_pan.pos": 1.0, "gripper.pos": 2.0},
            "is_controlled": True,
        }
    )
    return arm


def _make_adapter(mode: str = "follower") -> BimanualSO101Adapter:
    features = ["shoulder_pan.pos", "gripper.pos"]
    left = _make_arm(features)
    right = _make_arm(features)
    return BimanualSO101Adapter(left=left, right=right, mode=mode)


class TestProperties:
    def test_robot_type_follower(self):
        adapter = _make_adapter("follower")
        assert adapter.robot_type == RobotType.SO101_BIMANUAL_FOLLOWER

    def test_robot_type_teleoperator(self):
        adapter = _make_adapter("teleoperator")
        assert adapter.robot_type == RobotType.SO101_BIMANUAL_LEADER

    def test_features_prefixed(self):
        adapter = _make_adapter()
        assert adapter.features() == [
            "left_shoulder_pan.pos",
            "left_gripper.pos",
            "right_shoulder_pan.pos",
            "right_gripper.pos",
        ]


class TestStateAndForces:
    def test_read_state_merges_with_prefix(self):
        adapter = _make_adapter()
        result = asyncio.run(adapter.read_state())
        assert result["event"] == "state_was_updated"
        assert "left_shoulder_pan.pos" in result["state"]
        assert "right_gripper.pos" in result["state"]

    def test_read_forces_returns_event_when_none(self):
        adapter = _make_adapter()
        result = asyncio.run(adapter.read_forces())
        assert result is not None
        assert result["event"] == "force_was_updated"
        assert result["state"] is None


class TestCommands:
    def test_set_joints_state_splits_by_prefix(self):
        adapter = _make_adapter()
        joints = {
            "left_shoulder_pan.pos": 1.0,
            "left_gripper.pos": 2.0,
            "right_shoulder_pan.pos": 3.0,
            "right_gripper.pos": 4.0,
        }
        asyncio.run(adapter.set_joints_state(joints, goal_time=0.1))

        adapter._left.set_joints_state.assert_called_once_with(
            {"shoulder_pan.pos": 1.0, "gripper.pos": 2.0},
            0.1,
        )
        adapter._right.set_joints_state.assert_called_once_with(
            {"shoulder_pan.pos": 3.0, "gripper.pos": 4.0},
            0.1,
        )

    def test_connect_rolls_back_left_on_right_failure(self):
        adapter = _make_adapter()
        adapter._right.connect = AsyncMock(side_effect=RuntimeError("right failed"))

        with pytest.raises(RuntimeError, match="right failed"):
            asyncio.run(adapter.connect())

        adapter._left.connect.assert_called_once()
        adapter._left.disconnect.assert_called_once()
