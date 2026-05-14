# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import asyncio
from unittest.mock import MagicMock, PropertyMock

import pytest
from physicalai.robot.trossen.constants import WIDOWXAI_JOINT_ORDER
from physicalai.robot.trossen.widowxai import WidowXAI

from robots.physicalai_adapter import PhysicalAIRobotAdapter, PhysicalAIRobotAdapterConfig
from schemas.robot import RobotType


def _make_mock_robot(role="follower"):
    robot = MagicMock(spec=WidowXAI)
    robot.is_connected.return_value = False
    robot.joint_names = list(WIDOWXAI_JOINT_ORDER)
    obs = MagicMock()
    obs.joint_positions = [0.0] * len(WIDOWXAI_JOINT_ORDER)
    obs.sensor_data = {
        "velocities": [0.0] * len(WIDOWXAI_JOINT_ORDER),
        "efforts": [0.0] * len(WIDOWXAI_JOINT_ORDER),
    }
    robot.get_observation.return_value = obs
    type(robot).ip = PropertyMock(return_value="192.168.1.2")
    return robot


def _make_adapter(mode="follower"):
    robot = _make_mock_robot(role=mode)
    robot_type = RobotType.TROSSEN_WIDOWXAI_FOLLOWER if mode == "follower" else RobotType.TROSSEN_WIDOWXAI_LEADER
    adapter = PhysicalAIRobotAdapter(
        robot=robot,
        robot_type=robot_type,
        config=PhysicalAIRobotAdapterConfig(
            include_velocities=True,
            pass_goal_time=True,
            goal_time_scale=1.0,
            emit_force_event_when_none=False,
            external_effort_gain=0.1,
        ),
    )
    return adapter, robot


class TestProperties:
    def test_name(self):
        adapter, _ = _make_adapter()
        assert adapter.name == "PhysicalAIRobot"

    def test_robot_type_follower(self):
        adapter, _ = _make_adapter(mode="follower")
        assert adapter.robot_type == RobotType.TROSSEN_WIDOWXAI_FOLLOWER

    def test_robot_type_leader(self):
        adapter, _ = _make_adapter(mode="leader")
        assert adapter.robot_type == RobotType.TROSSEN_WIDOWXAI_LEADER

    def test_is_connected_delegates(self):
        adapter, robot = _make_adapter()
        robot.is_connected.return_value = True
        assert adapter.is_connected is True
        robot.is_connected.return_value = False
        assert adapter.is_connected is False

    def test_features_includes_pos_and_vel(self):
        adapter, _ = _make_adapter()
        expected = [f"{n}.pos" for n in WIDOWXAI_JOINT_ORDER] + [f"{n}.vel" for n in WIDOWXAI_JOINT_ORDER]
        assert adapter.features() == expected


class TestDegreeRadianConversion:
    def test_read_state_delegates_to_driver_state_dict(self):
        adapter, robot = _make_adapter()
        obs = MagicMock()
        obs.joint_positions = [57.2958, 28.6479, -28.6479, 85.9437, -57.2958, 17.1887, 0.02]
        obs.sensor_data = {
            "velocities": [0.1] * len(WIDOWXAI_JOINT_ORDER),
            "efforts": [0.0] * len(WIDOWXAI_JOINT_ORDER),
        }
        robot.get_observation.return_value = obs

        result = asyncio.run(adapter.read_state())

        robot.get_observation.assert_called_once()
        assert result["state"]["shoulder_pan.pos"] == pytest.approx(57.2958, abs=0.01)
        assert result["state"]["wrist_roll.pos"] == pytest.approx(17.1887, abs=0.01)
        assert result["state"]["gripper.pos"] == pytest.approx(0.02, abs=0.001)
        assert result["state"]["gripper.vel"] == pytest.approx(0.1, abs=1e-6)

    def test_set_joints_state_delegates_to_driver_state_send(self):
        adapter, robot = _make_adapter()

        joints = {
            "shoulder_pan.pos": 57.2958,
            "shoulder_lift.pos": 28.6479,
            "elbow_flex.pos": -28.6479,
            "wrist_flex.pos": 85.9437,
            "wrist_yaw.pos": -57.2958,
            "wrist_roll.pos": 17.1887,
            "gripper.pos": 0.05,
            "shoulder_pan.vel": 0.0,
            "shoulder_lift.vel": 0.0,
            "elbow_flex.vel": 0.0,
            "wrist_flex.vel": 0.0,
            "wrist_yaw.vel": 0.0,
            "wrist_roll.vel": 0.0,
            "gripper.vel": 0.0,
        }

        asyncio.run(adapter.set_joints_state(joints, goal_time=0.1))

        robot.send_action.assert_called_once()
        call_args = robot.send_action.call_args
        sent = call_args.args[0]
        assert sent[0] == pytest.approx(57.2958, abs=0.01)
        assert sent[1] == pytest.approx(28.6479, abs=0.01)
        assert sent[6] == pytest.approx(0.05, abs=1e-6)
        assert call_args.kwargs["goal_time"] == pytest.approx(0.1, abs=1e-6)

    def test_roundtrip_conversion(self):
        adapter, robot = _make_adapter()
        obs = MagicMock()
        original_radians = [0.5, -0.3, 1.0, -1.0, 0.2, 0.8, 0.05]
        obs.joint_positions = original_radians
        obs.sensor_data = {
            "velocities": [0.0] * len(WIDOWXAI_JOINT_ORDER),
            "efforts": [0.0] * len(WIDOWXAI_JOINT_ORDER),
        }
        robot.get_observation.return_value = obs
        result = asyncio.run(adapter.read_state())

        joints = {}
        for name in WIDOWXAI_JOINT_ORDER:
            joints[f"{name}.pos"] = result["state"][f"{name}.pos"]
            joints[f"{name}.vel"] = 0.0

        asyncio.run(adapter.set_joints_state(joints, goal_time=0.1))

        robot.send_action.assert_called_once()
        sent = robot.send_action.call_args.args[0]
        for i, expected in enumerate(original_radians):
            assert sent[i] == pytest.approx(expected, abs=0.01)


class TestConnect:
    def test_connect_calls_driver(self):
        adapter, robot = _make_adapter()
        asyncio.run(adapter.connect())
        robot.connect.assert_called_once()

    def test_connect_follower_sets_controlled(self):
        adapter, robot = _make_adapter(mode="follower")
        asyncio.run(adapter.connect())
        assert adapter.is_controlled is True

    def test_connect_leader_unsets_controlled(self):
        adapter, robot = _make_adapter(mode="leader")
        asyncio.run(adapter.connect())
        assert adapter.is_controlled is False


class TestDisconnect:
    def test_disconnect_calls_driver(self):
        adapter, robot = _make_adapter()
        asyncio.run(adapter.disconnect())
        robot.disconnect.assert_called_once()


class TestPing:
    def test_ping_returns_pong(self):
        adapter, _ = _make_adapter()
        result = asyncio.run(adapter.ping())
        assert result["event"] == "pong"
        assert "timestamp" in result


class TestSetJointsState:
    def test_calls_send_state_dict(self):
        adapter, robot = _make_adapter()
        joints = {}
        for name in WIDOWXAI_JOINT_ORDER:
            joints[f"{name}.pos"] = 0.0
            joints[f"{name}.vel"] = 0.1
        joints["shoulder_pan.pos"] = 57.2958
        joints["shoulder_lift.pos"] = 28.6479
        joints["gripper.pos"] = 0.05

        asyncio.run(adapter.set_joints_state(joints, goal_time=0.1))

        robot.send_action.assert_called_once()
        call_args = robot.send_action.call_args
        sent = call_args.args[0]
        assert sent[0] == pytest.approx(57.2958, abs=0.01)
        assert sent[1] == pytest.approx(28.6479, abs=0.01)
        assert sent[6] == pytest.approx(0.05, abs=1e-6)
        assert robot.send_action.call_args.kwargs["goal_time"] == pytest.approx(0.1, abs=1e-6)

    def test_leader_forwards_to_driver(self):
        adapter, robot = _make_adapter(mode="leader")
        joints = {f"{name}.pos": 0.0 for name in WIDOWXAI_JOINT_ORDER}
        joints.update({f"{name}.vel": 0.0 for name in WIDOWXAI_JOINT_ORDER})
        asyncio.run(adapter.set_joints_state(joints, goal_time=0.1))
        robot.send_action.assert_called_once()


class TestReadState:
    def test_returns_state_event(self):
        adapter, robot = _make_adapter()
        obs = MagicMock()
        obs.joint_positions = [0.0] * len(WIDOWXAI_JOINT_ORDER)
        obs.sensor_data = {
            "velocities": [0.0] * len(WIDOWXAI_JOINT_ORDER),
            "efforts": [0.0] * len(WIDOWXAI_JOINT_ORDER),
        }
        robot.get_observation.return_value = obs
        result = asyncio.run(adapter.read_state())
        assert result["event"] == "state_was_updated"
        assert "state" in result
        assert "is_controlled" in result

    def test_state_has_pos_and_vel_keys(self):
        adapter, robot = _make_adapter()
        obs = MagicMock()
        obs.joint_positions = [0.0] * len(WIDOWXAI_JOINT_ORDER)
        obs.sensor_data = {
            "velocities": [0.0] * len(WIDOWXAI_JOINT_ORDER),
            "efforts": [0.0] * len(WIDOWXAI_JOINT_ORDER),
        }
        robot.get_observation.return_value = obs
        result = asyncio.run(adapter.read_state())
        state = result["state"]
        for name in WIDOWXAI_JOINT_ORDER:
            assert f"{name}.pos" in state
            assert f"{name}.vel" in state


class TestReadForces:
    def test_follower_returns_forces(self):
        adapter, robot = _make_adapter(mode="follower")
        obs = MagicMock()
        obs.joint_positions = [0.0] * len(WIDOWXAI_JOINT_ORDER)
        obs.sensor_data = {
            "velocities": [0.0] * len(WIDOWXAI_JOINT_ORDER),
            "efforts": [float(i) for i in range(len(WIDOWXAI_JOINT_ORDER))],
        }
        robot.get_observation.return_value = obs
        result = asyncio.run(adapter.read_forces())
        assert result is not None
        state = result["state"]
        for name in WIDOWXAI_JOINT_ORDER:
            assert f"{name}.eff" in state

    def test_leader_returns_none(self):
        adapter, robot = _make_adapter(mode="leader")
        obs = MagicMock()
        obs.joint_positions = [0.0] * len(WIDOWXAI_JOINT_ORDER)
        obs.sensor_data = {"velocities": [0.0] * len(WIDOWXAI_JOINT_ORDER)}
        robot.get_observation.return_value = obs
        result = asyncio.run(adapter.read_forces())
        assert result is None


class TestSetForces:
    def test_leader_calls_set_external_efforts_when_available(self):
        adapter, robot = _make_adapter(mode="leader")
        forces = {f"{name}.eff": float(i) * 0.1 for i, name in enumerate(WIDOWXAI_JOINT_ORDER)}
        robot.set_external_efforts = MagicMock()

        result = asyncio.run(adapter.set_forces(forces))

        assert result == forces
        robot.set_external_efforts.assert_called_once()
        call_args = robot.set_external_efforts.call_args
        assert call_args.kwargs["gain"] == pytest.approx(0.1, abs=1e-6)

    def test_set_forces_noops_when_external_efforts_unavailable(self):
        adapter, robot = _make_adapter(mode="leader")
        forces = {f"{name}.eff": 0.1 for name in WIDOWXAI_JOINT_ORDER}
        if hasattr(robot, "set_external_efforts"):
            delattr(robot, "set_external_efforts")

        result = asyncio.run(adapter.set_forces(forces))
        assert result == forces

    def test_follower_returns_forces_unchanged(self):
        adapter, robot = _make_adapter(mode="follower")
        forces = {f"{name}.eff": 0.5 for name in WIDOWXAI_JOINT_ORDER}
        robot.set_external_efforts = MagicMock()
        result = asyncio.run(adapter.set_forces(forces))
        assert result == forces
        robot.set_external_efforts.assert_not_called()


class TestTorque:
    def test_enable_torque_event(self):
        adapter, _ = _make_adapter()
        result = asyncio.run(adapter.enable_torque())
        assert result["event"] == "torque_was_enabled"
        assert adapter.is_controlled is True

    def test_disable_torque_event(self):
        adapter, _ = _make_adapter()
        result = asyncio.run(adapter.disable_torque())
        assert result["event"] == "torque_was_disabled"
        assert adapter.is_controlled is False
