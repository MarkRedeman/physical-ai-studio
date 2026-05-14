import asyncio
from unittest.mock import MagicMock

import numpy as np
import pytest
from physicalai.robot.so101.constants import SO101_JOINT_ORDER

from robots.so101.adapter import SO101Adapter, _clamp, _clamp_joints


def _make_mock_robot() -> MagicMock:
    robot = MagicMock()
    robot.port = "/dev/ttyUSB0"
    robot.joint_names = list(SO101_JOINT_ORDER)
    robot.max_speed = 270.0
    robot.is_connected.return_value = False
    return robot


def _make_adapter(
    mode: str = "follower",
) -> tuple[SO101Adapter, MagicMock]:
    robot = _make_mock_robot()
    adapter = SO101Adapter(robot=robot, mode=mode)
    return adapter, robot


class TestClamp:
    def test_within_range(self):
        assert _clamp(5.0, 10.0) == 5.0

    def test_above_range(self):
        assert _clamp(15.0, 10.0) == 10.0

    def test_below_range(self):
        assert _clamp(-15.0, 10.0) == -10.0

    def test_zero(self):
        assert _clamp(0.0, 10.0) == 0.0


class TestClampJoints:
    def test_clamps_towards_target(self):
        current = {"a.pos": 0.0, "b.pos": 0.0}
        target = {"a.pos": 100.0, "b.pos": -100.0}
        result = _clamp_joints(current, target, 50.0)
        assert result["a.pos"] == 50.0
        assert result["b.pos"] == -50.0

    def test_no_clamp_when_within_range(self):
        current = {"a.pos": 90.0}
        target = {"a.pos": 100.0}
        result = _clamp_joints(current, target, 50.0)
        assert result["a.pos"] == 100.0


class TestProperties:
    def test_name(self):
        adapter, _ = _make_adapter()
        assert adapter.name == "So101"

    def test_robot_type_follower(self):
        from schemas.robot import RobotType

        adapter, _ = _make_adapter(mode="follower")
        assert adapter.robot_type == RobotType.SO101_FOLLOWER

    def test_robot_type_teleoperator(self):
        from schemas.robot import RobotType

        adapter, _ = _make_adapter(mode="teleoperator")
        assert adapter.robot_type == RobotType.SO101_LEADER

    def test_is_connected_delegates_to_robot(self):
        adapter, robot = _make_adapter()
        robot.is_connected.return_value = True
        assert adapter.is_connected is True
        robot.is_connected.return_value = False
        assert adapter.is_connected is False

    def test_features(self):
        adapter, _ = _make_adapter()
        expected = [f"{name}.pos" for name in SO101_JOINT_ORDER]
        assert adapter.features() == expected


class TestConnect:
    def test_connect_calls_robot_connect(self):
        adapter, robot = _make_adapter()
        robot.connect = MagicMock()
        asyncio.run(adapter.connect())
        robot.connect.assert_called_once()

    def test_connect_sets_is_controlled_for_follower(self):
        adapter, robot = _make_adapter(mode="follower")
        robot.connect = MagicMock()
        asyncio.run(adapter.connect())
        assert adapter.is_controlled is True

    def test_connect_does_not_set_controlled_for_teleoperator(self):
        adapter, robot = _make_adapter(mode="teleoperator")
        robot.connect = MagicMock()
        asyncio.run(adapter.connect())
        assert adapter.is_controlled is False


class TestDisconnect:
    def test_disconnect_calls_robot_disconnect(self):
        adapter, robot = _make_adapter()
        robot.disconnect = MagicMock()
        asyncio.run(adapter.disconnect())
        robot.disconnect.assert_called_once()


class TestReadState:
    def test_returns_normalized_state_dict(self):
        adapter, robot = _make_adapter()
        obs_mock = MagicMock()
        obs_mock.joint_positions = np.zeros(6, dtype=np.float32)
        robot.get_observation.return_value = obs_mock

        result = asyncio.run(adapter.read_state())

        assert result["event"] == "state_was_updated"
        assert "state" in result
        assert "timestamp" in result
        state = result["state"]
        assert len(state) == 6
        for name in SO101_JOINT_ORDER:
            assert f"{name}.pos" in state


class TestSetJointsState:
    def test_sends_action_to_robot(self):
        adapter, robot = _make_adapter()

        obs_mock = MagicMock()
        obs_mock.joint_positions = np.zeros(6, dtype=np.float32)
        robot.get_observation.return_value = obs_mock
        robot.send_action = MagicMock()

        joints = {f"{name}.pos": 0.0 for name in SO101_JOINT_ORDER}
        result = asyncio.run(adapter.set_joints_state(joints, goal_time=0.033))

        assert result["event"] == "joints_state_was_set"
        robot.send_action.assert_called_once()
        action_array = robot.send_action.call_args[0][0]
        assert action_array.shape == (6,)

    def test_velocity_clamping_limits_movement(self):
        adapter, robot = _make_adapter()

        obs_mock = MagicMock()
        obs_mock.joint_positions = np.zeros(6, dtype=np.float32)
        robot.get_observation.return_value = obs_mock
        robot.send_action = MagicMock()

        far_joints = {f"{name}.pos": 1000.0 for name in SO101_JOINT_ORDER}
        goal_time = 0.033
        asyncio.run(adapter.set_joints_state(far_joints, goal_time=goal_time))

        # Verify clamping happened in SO101 driver units.
        max_delta = robot.max_speed * goal_time
        action_sent = robot.send_action.call_args[0][0]
        for i in range(len(action_sent)):
            assert abs(action_sent[i]) <= max_delta + 1e-6


class TestTorque:
    def test_enable_torque(self):
        adapter, robot = _make_adapter()
        robot._set_torque = MagicMock()
        result = asyncio.run(adapter.enable_torque())
        robot._set_torque.assert_called_once_with(enabled=True)
        assert result["event"] == "torque_was_enabled"
        assert adapter.is_controlled is True

    def test_disable_torque(self):
        adapter, robot = _make_adapter()
        robot._set_torque = MagicMock()
        result = asyncio.run(adapter.disable_torque())
        robot._set_torque.assert_called_once_with(enabled=False)
        assert result["event"] == "torque_was_disabled"
        assert adapter.is_controlled is False


class TestPing:
    def test_ping_returns_pong(self):
        adapter, _ = _make_adapter()
        result = asyncio.run(adapter.ping())
        assert result["event"] == "pong"
        assert "timestamp" in result


class TestReadForces:
    def test_returns_none_state(self):
        adapter, _ = _make_adapter()
        result = asyncio.run(adapter.read_forces())
        assert result["event"] == "force_was_updated"
        assert result["state"] is None


class TestSetForces:
    def test_raises_not_implemented(self):
        adapter, _ = _make_adapter()
        with pytest.raises(NotImplementedError):
            asyncio.run(adapter.set_forces({}))
