# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from robots.so101.bimanual_adapter import BimanualSO101Adapter
from schemas.robot import RobotType, SO101BimanualPayload, SO101BimanualRobot


def _make_bimanual_robot(mode: str) -> SO101BimanualRobot:
    robot_type = RobotType.SO101_BIMANUAL_FOLLOWER if mode == "follower" else RobotType.SO101_BIMANUAL_LEADER
    return SO101BimanualRobot(
        id=uuid4(),
        name="SO101 Bimanual Test",
        type=robot_type,
        payload=SO101BimanualPayload(
            connection_string_left="",
            connection_string_right="",
            serial_number_left="SN-L",
            serial_number_right="SN-R",
            active_calibration_id_left=uuid4(),
            active_calibration_id_right=uuid4(),
        ),
    )


class TestRobotClientFactoryBimanualSO101:
    @pytest.fixture()
    def factory(self):
        from robots.robot_client_factory import RobotClientFactory

        manager = MagicMock()
        manager.robots = [
            MagicMock(serial_number="SN-L", connection_string="/dev/ttyUSB0"),
            MagicMock(serial_number="SN-R", connection_string="/dev/ttyUSB1"),
        ]
        cal_service = MagicMock()
        cal_service.get_calibration = AsyncMock(side_effect=[MagicMock(values={}), MagicMock(values={})])
        return RobotClientFactory(robot_manager=manager, calibration_service=cal_service)

    @pytest.mark.parametrize("mode", ["follower", "leader"])
    def test_builds_bimanual_so101_adapter(self, factory, mode):
        robot = _make_bimanual_robot(mode)

        with (
            patch("robots.robot_client_factory.SO101Calibration") as mock_so_cal,
            patch("robots.robot_client_factory.SO101") as mock_so,
            patch("robots.robot_client_factory.SO101Adapter") as mock_adapter,
        ):
            mock_so_cal.from_dict.return_value = MagicMock()
            mock_so.return_value = MagicMock()
            left_adapter = MagicMock()
            right_adapter = MagicMock()
            mock_adapter.side_effect = [left_adapter, right_adapter]

            import asyncio

            result = asyncio.run(factory.build(robot))

        assert isinstance(result, BimanualSO101Adapter)
        assert mock_so.call_count == 2

    def test_requires_both_calibrations(self, factory):
        robot = _make_bimanual_robot("follower")
        factory.calibration_service.get_calibration = AsyncMock(side_effect=[None, MagicMock(values={})])

        with pytest.raises(Exception):
            import asyncio

            asyncio.run(factory.build(robot))
