# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from repositories.mappers.project_robot_mapper import ProjectRobotMapper
from schemas.robot import RobotType, SO101BimanualPayload, SO101BimanualRobot


def _make_bimanual_so101_db_model(robot_type: RobotType):
    model = MagicMock()
    model.id = str(uuid4())
    model.name = "SO101 Bimanual Test Robot"
    model.type = str(robot_type)
    model.payload = {
        "connection_string_left": "",
        "connection_string_right": "",
        "serial_number_left": "SN-L",
        "serial_number_right": "SN-R",
        "active_calibration_id_left": str(uuid4()),
        "active_calibration_id_right": str(uuid4()),
    }
    model.active_calibration_id = None
    model.created_at = datetime(2026, 1, 1)
    model.updated_at = datetime(2026, 1, 1)
    return model


class TestProjectRobotMapperBimanualSO101:
    @pytest.mark.parametrize(
        "robot_type",
        [
            RobotType.SO101_BIMANUAL_FOLLOWER,
            RobotType.SO101_BIMANUAL_LEADER,
        ],
    )
    def test_from_schema_returns_bimanual_so101_robot(self, robot_type):
        db_model = _make_bimanual_so101_db_model(robot_type)
        result = ProjectRobotMapper.from_schema(db_model)

        assert isinstance(result, SO101BimanualRobot)
        assert result.type == robot_type
        assert isinstance(result.payload, SO101BimanualPayload)
        assert result.payload.serial_number_left == "SN-L"
        assert result.payload.serial_number_right == "SN-R"
