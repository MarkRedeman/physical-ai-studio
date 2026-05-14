# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""WidowXAI protocol adapter.

Wraps physicalai's ``WidowXAI`` driver behind the backend's ``RobotClient``
interface with async-safe hardware access.
"""

import asyncio
from typing import Literal

from loguru import logger
from physicalai.robot.trossen import WidowXAI

from robots.robot_client import RobotClient
from schemas.robot import RobotType

HARDWARE_TIMEOUT_CONNECT = 10.0
HARDWARE_TIMEOUT_COMMAND = 5.0


class WidowXAIAdapter(RobotClient):
    """Adapt physicalai's :class:`WidowXAI` to the backend's RobotClient API."""

    name = "WidowXAI"

    def __init__(self, robot: WidowXAI, mode: Literal["follower", "leader"]) -> None:
        self._robot = robot
        self._mode = mode
        self._bus_lock = asyncio.Lock()
        self.is_controlled: bool = False

    @property
    def robot_type(self) -> RobotType:
        if self._mode == "follower":
            return RobotType.TROSSEN_WIDOWXAI_FOLLOWER
        return RobotType.TROSSEN_WIDOWXAI_LEADER

    @property
    def is_connected(self) -> bool:
        return self._robot.is_connected()

    async def connect(self) -> None:
        logger.info(f"Connecting to WidowXAI {self._mode} at {self._robot.ip}")
        try:
            async with self._bus_lock, asyncio.timeout(HARDWARE_TIMEOUT_CONNECT):
                await asyncio.to_thread(self._robot.connect)
            if self._mode == "follower":
                self.is_controlled = True
            else:
                self.is_controlled = False
        except TimeoutError:
            logger.error("Timeout connecting to robot")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to robot: {e}")
            raise

    async def disconnect(self) -> None:
        logger.info(f"Disconnecting WidowXAI {self._mode} at {self._robot.ip}")
        try:
            async with self._bus_lock, asyncio.timeout(HARDWARE_TIMEOUT_COMMAND):
                await asyncio.to_thread(self._robot.disconnect)
            logger.info("Robot disconnected")
        except TimeoutError:
            logger.warning("Timeout during robot disconnect - forcing cleanup")
        except Exception as e:
            logger.error(f"Error during robot disconnect: {e}")

    async def ping(self) -> dict:
        return self._create_event("pong")

    async def set_joints_state(self, joints: dict, goal_time: float) -> dict:
        if self._mode == "leader":
            raise RuntimeError("Cannot send actions to a leader arm")

        async with self._bus_lock, asyncio.timeout(HARDWARE_TIMEOUT_COMMAND):
            await asyncio.to_thread(self._robot.send_state_dict, joints, goal_time)

        return self._create_event("joints_state_was_set", joints=joints)

    async def enable_torque(self) -> dict:
        self.is_controlled = True
        return self._create_event("torque_was_enabled")

    async def disable_torque(self) -> dict:
        self.is_controlled = False
        return self._create_event("torque_was_disabled")

    async def read_state(self, *, normalize: bool = True) -> dict:  # noqa: ARG002
        try:
            async with self._bus_lock, asyncio.timeout(HARDWARE_TIMEOUT_COMMAND):
                state = await asyncio.to_thread(self._robot.read_state_dict)
            return self._create_event(
                "state_was_updated",
                state=state,
                is_controlled=self.is_controlled,
            )
        except Exception as e:
            logger.error(f"Robot read error: {e}")
            raise

    async def read_forces(self) -> dict | None:
        async with self._bus_lock, asyncio.timeout(HARDWARE_TIMEOUT_COMMAND):
            forces = await asyncio.to_thread(self._robot.read_force_dict)

        if forces is None:
            return None

        return self._create_event(
            "force_was_updated",
            state=forces,
            is_controlled=self.is_controlled,
        )

    async def set_forces(self, forces: dict) -> dict:
        async with self._bus_lock, asyncio.timeout(HARDWARE_TIMEOUT_COMMAND):
            return await asyncio.to_thread(self._robot.set_force_dict, forces)

    def features(self) -> list[str]:
        return self._robot.feature_names()
