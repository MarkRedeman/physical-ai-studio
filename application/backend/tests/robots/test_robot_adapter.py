import asyncio
from unittest.mock import MagicMock

import pytest
from physicalai.robot.so101.constants import SO101_JOINT_ORDER

from robots.physicalai_adapter import PhysicalAIRobotAdapter
from schemas.robot import RobotType


def _make_mock_robot() -> MagicMock:
    robot = MagicMock()
    robot.port = "/dev/ttyUSB0"
    robot.joint_names = list(SO101_JOINT_ORDER)
    robot.feature_names.return_value = [f"{name}.pos" for name in SO101_JOINT_ORDER]
    robot.read_state_dict.return_value = {f"{name}.pos": 0.0 for name in SO101_JOINT_ORDER}
    robot.read_force_dict.return_value = None
    robot.set_torque = MagicMock()
    robot.is_connected.return_value = False
    return robot


def _make_adapter(
    mode: str = "follower",
) -> tuple[PhysicalAIRobotAdapter, MagicMock]:
    robot = _make_mock_robot()
    adapter = PhysicalAIRobotAdapter(
        robot=robot,
        mode=mode,
        follower_type=RobotType.SO101_FOLLOWER,
        leader_type=RobotType.SO101_LEADER,
        emit_force_event_when_none=True,
        delegate_torque=True,
    )
    return adapter, robot


class TestProperties:
    def test_name(self):
        adapter, _ = _make_adapter()
        assert adapter.name == "PhysicalAIRobot"

    def test_robot_type_follower(self):
        adapter, _ = _make_adapter(mode="follower")
        assert adapter.robot_type == RobotType.SO101_FOLLOWER

    def test_robot_type_teleoperator(self):
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
        robot.read_state_dict.return_value = {f"{name}.pos": 0.0 for name in SO101_JOINT_ORDER}

        result = asyncio.run(adapter.read_state())

        assert result["event"] == "state_was_updated"
        assert "state" in result
        assert "timestamp" in result
        state = result["state"]
        assert len(state) == 6
        for name in SO101_JOINT_ORDER:
            assert f"{name}.pos" in state
        robot.read_state_dict.assert_called_once()


class TestSetJointsState:
    def test_sends_action_to_robot(self):
        adapter, robot = _make_adapter()

        joints = {f"{name}.pos": 0.0 for name in SO101_JOINT_ORDER}
        result = asyncio.run(adapter.set_joints_state(joints, goal_time=0.033))

        assert result["event"] == "joints_state_was_set"
        robot.send_state_dict.assert_called_once_with(joints, 0.033)

    def test_delegates_state_send_to_driver(self):
        adapter, robot = _make_adapter()

        far_joints = {f"{name}.pos": 1000.0 for name in SO101_JOINT_ORDER}
        goal_time = 0.033
        asyncio.run(adapter.set_joints_state(far_joints, goal_time=goal_time))
        robot.send_state_dict.assert_called_once_with(far_joints, goal_time)


class TestTorque:
    def test_enable_torque(self):
        adapter, robot = _make_adapter()
        result = asyncio.run(adapter.enable_torque())
        robot.set_torque.assert_called_once_with(enabled=True)
        assert result["event"] == "torque_was_enabled"
        assert adapter.is_controlled is True

    def test_disable_torque(self):
        adapter, robot = _make_adapter()
        result = asyncio.run(adapter.disable_torque())
        robot.set_torque.assert_called_once_with(enabled=False)
        assert result["event"] == "torque_was_disabled"
        assert adapter.is_controlled is False


class TestPing:
    def test_ping_returns_pong(self):
        adapter, _ = _make_adapter()
        result = asyncio.run(adapter.ping())
        assert result["event"] == "pong"
        assert "timestamp" in result


class TestReadForces:
    def test_returns_force_event_with_none_state(self):
        adapter, robot = _make_adapter()
        robot.read_force_dict.return_value = None
        result = asyncio.run(adapter.read_forces())
        assert result is not None
        assert result["event"] == "force_was_updated"
        assert result["state"] is None


class TestSetForces:
    def test_delegates_to_driver_force_write(self):
        adapter, robot = _make_adapter()
        robot.set_force_dict.side_effect = NotImplementedError("Force control is not implemented for SO101")

        with pytest.raises(NotImplementedError, match="not implemented"):
            asyncio.run(adapter.set_forces({}))
