"""Tests for ReBot robot schema mapping.

Verifies that ReBot payloads are correctly serialized/deserialized
through the discriminated union schema.
"""

from uuid import uuid4

from schemas.robot import (
    ReBotB601DMRobot,
    ReBotB601DMRobotWithConnectionState,
    ReBotArm102LeaderRobot,
    ReBotArm102LeaderRobotWithConnectionState,
    RobotAdapter,
    RobotWithConnectionStateAdapter,
)


class TestReBotB601DMMapping:
    def test_rebot_b601_dm_round_trip(self):
        robot_id = uuid4()
        robot = ReBotB601DMRobot(
            id=robot_id,
            name="ReBot B601",
            type="ReBot_B601_DM_Follower",
            payload={
                "connection_string": "/dev/ttyUSB0",
                "serial_number": "REBOT-601-001",
                "can_adapter": "damiao",
                "dm_serial_baud": 921600,
                "disable_torque_on_disconnect": True,
                "force_pos_torque_ratio": 0.1,
            },
        )
        dumped = robot.model_dump(mode="json")
        loaded = RobotAdapter.validate_python(dumped)
        assert loaded.type == robot.type
        assert loaded.payload.serial_number == "REBOT-601-001"
        assert loaded.payload.can_adapter == "damiao"

    def test_rebot_b601_dm_with_connection_state(self):
        robot_id = uuid4()
        robot = ReBotB601DMRobotWithConnectionState(
            id=robot_id,
            name="ReBot B601",
            type="ReBot_B601_DM_Follower",
            payload={
                "connection_string": "/dev/ttyUSB0",
                "serial_number": "REBOT-601-001",
                "can_adapter": "socketcan",
                "dm_serial_baud": 115200,
                "disable_torque_on_disconnect": False,
                "force_pos_torque_ratio": 0.2,
            },
            connection_status="online",
        )
        dumped = robot.model_dump(mode="json")
        loaded = RobotWithConnectionStateAdapter.validate_python(dumped)
        assert loaded.connection_status == "online"
        assert loaded.payload.can_adapter == "socketcan"
        assert loaded.payload.dm_serial_baud == 115200

    def test_rebot_b601_dm_defaults(self):
        robot_id = uuid4()
        robot = ReBotB601DMRobot(
            id=robot_id,
            name="ReBot Defaults",
            type="ReBot_B601_DM_Follower",
            payload={
                "serial_number": "REBOT-DEF-001",
            },
        )
        assert robot.payload.can_adapter == "damiao"
        assert robot.payload.dm_serial_baud == 921600
        assert robot.payload.disable_torque_on_disconnect is True
        assert robot.payload.force_pos_torque_ratio == 0.1


class TestReBotArm102LeaderMapping:
    def test_rebot_arm102_leader_round_trip(self):
        robot_id = uuid4()
        robot = ReBotArm102LeaderRobot(
            id=robot_id,
            name="ReBot Arm 102",
            type="ReBot_Arm102_Leader",
            payload={
                "connection_string": "/dev/ttyACM0",
                "serial_number": "REBOT-102-001",
                "baudrate": 1000000,
                "unlock_on_connect": True,
                "reset_multi_turn_on_connect": True,
                "zero_on_connect": False,
            },
        )
        dumped = robot.model_dump(mode="json")
        loaded = RobotAdapter.validate_python(dumped)
        assert loaded.type == robot.type
        assert loaded.payload.serial_number == "REBOT-102-001"
        assert loaded.payload.baudrate == 1000000

    def test_rebot_arm102_leader_with_connection_state(self):
        robot_id = uuid4()
        robot = ReBotArm102LeaderRobotWithConnectionState(
            id=robot_id,
            name="ReBot Arm 102",
            type="ReBot_Arm102_Leader",
            payload={
                "connection_string": "/dev/ttyACM0",
                "serial_number": "REBOT-102-002",
                "baudrate": 921600,
                "unlock_on_connect": False,
                "reset_multi_turn_on_connect": False,
                "zero_on_connect": True,
            },
            connection_status="offline",
        )
        dumped = robot.model_dump(mode="json")
        loaded = RobotWithConnectionStateAdapter.validate_python(dumped)
        assert loaded.connection_status == "offline"
        assert loaded.payload.baudrate == 921600
        assert loaded.payload.zero_on_connect is True

    def test_rebot_arm102_leader_defaults(self):
        robot_id = uuid4()
        robot = ReBotArm102LeaderRobot(
            id=robot_id,
            name="ReBot Defaults",
            type="ReBot_Arm102_Leader",
            payload={
                "serial_number": "REBOT-102-DEF",
            },
        )
        assert robot.payload.baudrate == 1000000
        assert robot.payload.unlock_on_connect is True
        assert robot.payload.reset_multi_turn_on_connect is True
        assert robot.payload.zero_on_connect is False


class TestReBotInDiscriminatedUnion:
    def test_rebot_b601_dm_in_robot_union(self):
        robot_id = uuid4()
        robot = ReBotB601DMRobot(
            id=robot_id,
            name="ReBot B601",
            type="ReBot_B601_DM_Follower",
            payload={"serial_number": "REBOT-601-003"},
        )
        dumped = robot.model_dump(mode="json")
        loaded = RobotAdapter.validate_python(dumped)
        assert isinstance(loaded, ReBotB601DMRobot)

    def test_rebot_arm102_leader_in_robot_union(self):
        robot_id = uuid4()
        robot = ReBotArm102LeaderRobot(
            id=robot_id,
            name="ReBot Arm 102",
            type="ReBot_Arm102_Leader",
            payload={"serial_number": "REBOT-102-003"},
        )
        dumped = robot.model_dump(mode="json")
        loaded = RobotAdapter.validate_python(dumped)
        assert isinstance(loaded, ReBotArm102LeaderRobot)
