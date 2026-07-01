# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from repositories.mappers.project_robot_mapper import ProjectRobotMapper
from schemas.robot import ReBotArm102LeaderRobot, ReBotB601DMRobot, RobotType


def _make_rebot_db_model(robot_type: RobotType, payload: dict):
    model = MagicMock()
    model.id = str(uuid4())
    model.name = "ReBot Test Robot"
    model.type = str(robot_type)
    model.payload = payload
    model.active_calibration_id = None
    model.created_at = datetime(2026, 1, 1)
    model.updated_at = datetime(2026, 1, 1)
    return model


class TestProjectRobotMapperReBot:
    def test_from_schema_returns_rebot_b601_dm(self):
        db_model = _make_rebot_db_model(
            RobotType.REBOT_B601_DM_FOLLOWER,
            {
                "connection_string": "",
                "serial_number": "REBOT-DM-001",
                "can_adapter": "damiao",
                "dm_serial_baud": 921600,
                "disable_torque_on_disconnect": True,
                "force_pos_torque_ratio": 0.1,
            },
        )

        result = ProjectRobotMapper.from_schema(db_model)

        assert isinstance(result, ReBotB601DMRobot)
        assert result.type == RobotType.REBOT_B601_DM_FOLLOWER
        assert result.payload.serial_number == "REBOT-DM-001"

    def test_from_schema_returns_rebot_arm102_leader(self):
        db_model = _make_rebot_db_model(
            RobotType.REBOT_ARM102_LEADER,
            {
                "connection_string": "",
                "serial_number": "REBOT-LDR-001",
                "baudrate": 1000000,
                "unlock_on_connect": True,
                "reset_multi_turn_on_connect": True,
                "zero_on_connect": False,
            },
        )

        result = ProjectRobotMapper.from_schema(db_model)

        assert isinstance(result, ReBotArm102LeaderRobot)
        assert result.type == RobotType.REBOT_ARM102_LEADER
        assert result.payload.serial_number == "REBOT-LDR-001"

    @pytest.mark.parametrize(
        "robot",
        [
            ReBotB601DMRobot.model_validate(
                {
                    "id": str(uuid4()),
                    "name": "ReBot B601 DM",
                    "type": RobotType.REBOT_B601_DM_FOLLOWER,
                    "payload": {
                        "connection_string": "/dev/ttyACM0",
                        "serial_number": "REBOT-DM-001",
                        "can_adapter": "damiao",
                        "dm_serial_baud": 921600,
                        "disable_torque_on_disconnect": True,
                        "force_pos_torque_ratio": 0.1,
                    },
                }
            ),
            ReBotArm102LeaderRobot.model_validate(
                {
                    "id": str(uuid4()),
                    "name": "ReBot Arm102",
                    "type": RobotType.REBOT_ARM102_LEADER,
                    "payload": {
                        "connection_string": "/dev/ttyUSB0",
                        "serial_number": "REBOT-LDR-001",
                        "baudrate": 1000000,
                        "unlock_on_connect": True,
                        "reset_multi_turn_on_connect": True,
                        "zero_on_connect": False,
                    },
                }
            ),
        ],
    )
    def test_to_schema_serializes_payload(self, robot):
        schema = ProjectRobotMapper.to_schema(robot)
        assert isinstance(schema.payload, dict)
        assert schema.payload["serial_number"].startswith("REBOT-")
