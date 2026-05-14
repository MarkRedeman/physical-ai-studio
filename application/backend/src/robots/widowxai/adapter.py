# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Literal

from physicalai.robot.trossen import WidowXAI

from robots.physicalai_adapter import PhysicalAIRobotAdapter
from schemas.robot import RobotType


class WidowXAIAdapter(PhysicalAIRobotAdapter):
    name = "WidowXAI"

    def __init__(self, robot: WidowXAI, mode: Literal["follower", "leader"]) -> None:
        super().__init__(
            robot=robot,
            mode=mode,
            follower_type=RobotType.TROSSEN_WIDOWXAI_FOLLOWER,
            leader_type=RobotType.TROSSEN_WIDOWXAI_LEADER,
            emit_force_event_when_none=False,
            delegate_torque=False,
        )
