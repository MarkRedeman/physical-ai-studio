"""SO-101 protocol adapter.

Wraps physicalai's ``SO101`` driver behind the backend's ``RobotClient`` ABC.
All unit conversion is handled by physicalai.
"""

import asyncio
from typing import Literal

import numpy as np
from loguru import logger
from physicalai.robot.so101 import SO101

from robots.robot_client import RobotClient
from schemas.robot import RobotType

HARDWARE_TIMEOUT_CONNECT = 10.0
HARDWARE_TIMEOUT_COMMAND = 5.0

RobotMode = Literal["follower", "teleoperator"]


def _clamp(value: float, limit: float) -> float:
    return max(min(value, limit), -limit)


def _clamp_joints(current: dict[str, float], target: dict[str, float], max_distance: float) -> dict[str, float]:
    return {key: value + _clamp(target[key] - value, max_distance) for key, value in current.items()}


class SO101Adapter(RobotClient):
    """Adapt physicalai's :class:`SO101` to the backend's :class:`RobotClient` interface."""

    name = "So101"

    def __init__(
        self,
        robot: SO101,
        mode: RobotMode,
    ) -> None:
        self._robot = robot
        self._mode = mode
        self._bus_lock = asyncio.Lock()

        self.previous_target: dict[str, float] | None = None
        self.is_controlled: bool = False

    def _observation_to_state(self, values: np.ndarray) -> dict[str, float]:
        return {f"{name}.pos": float(values[i]) for i, name in enumerate(self._robot.joint_names)}

    def _state_to_action(self, joints: dict[str, float]) -> np.ndarray:
        result = np.empty(len(self._robot.joint_names), dtype=np.float32)
        for i, name in enumerate(self._robot.joint_names):
            result[i] = joints[f"{name}.pos"]
        return result

    @property
    def robot_type(self) -> RobotType:
        if self._mode == "follower":
            return RobotType.SO101_FOLLOWER
        return RobotType.SO101_LEADER

    @property
    def is_connected(self) -> bool:
        return self._robot.is_connected()

    async def connect(self) -> None:
        logger.info(f"Connecting to SO101 {self._mode} on {self._robot.port}")
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
        logger.info(f"Disconnecting SO101 {self._mode} on {self._robot.port}")
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
        await self._move_to_target(joints, goal_time)
        return self._create_event("joints_state_was_set", joints=joints)

    async def enable_torque(self) -> dict:
        logger.info("Enabling torque")
        async with self._bus_lock, asyncio.timeout(HARDWARE_TIMEOUT_COMMAND):
            await asyncio.to_thread(self._robot._set_torque, enabled=True)
        self.is_controlled = True
        return self._create_event("torque_was_enabled")

    async def disable_torque(self) -> dict:
        logger.info("Disabling torque")
        async with self._bus_lock, asyncio.timeout(HARDWARE_TIMEOUT_COMMAND):
            await asyncio.to_thread(self._robot._set_torque, enabled=False)
        self.is_controlled = False
        return self._create_event("torque_was_disabled")

    async def read_state(self, *, normalize: bool = True) -> dict:  # noqa: ARG002
        try:
            state = await self._get_state()
            return self._create_event(
                "state_was_updated",
                state=state,
                is_controlled=self.is_controlled,
            )
        except Exception as e:
            logger.error(f"Robot read error: {e}")
            raise

    async def read_forces(self) -> dict | None:
        return self._create_event(
            "force_was_updated",
            state=None,
            is_controlled=self.is_controlled,
        )

    async def set_forces(self, forces: dict) -> dict:
        raise NotImplementedError("Force control is not implemented for SO101")

    def features(self) -> list[str]:
        return [f"{name}.pos" for name in self._robot.joint_names]

    async def _get_state(self) -> dict[str, float]:
        async with self._bus_lock, asyncio.timeout(HARDWARE_TIMEOUT_COMMAND):
            obs = await asyncio.to_thread(self._robot.get_observation)
        return self._observation_to_state(obs.joint_positions)

    async def _move_to_target(self, joints: dict, goal_time: float) -> None:
        max_delta = self._robot.max_speed * goal_time

        async with self._bus_lock, asyncio.timeout(HARDWARE_TIMEOUT_COMMAND):
            obs = await asyncio.to_thread(self._robot.get_observation)
        current = obs.joint_positions.astype(np.float32)

        target = self._state_to_action(joints)

        if self.previous_target is not None:
            prev = self._state_to_action(self.previous_target)
            for i in range(len(current)):
                current[i] = current[i] + _clamp(prev[i] - current[i], max_delta * 2)

        clamped = np.array(
            [current[i] + _clamp(target[i] - current[i], max_delta) for i in range(len(current))],
            dtype=np.float32,
        )

        self.previous_target = self._observation_to_state(clamped)

        async with self._bus_lock, asyncio.timeout(HARDWARE_TIMEOUT_COMMAND):
            await asyncio.to_thread(self._robot.send_action, clamped)
