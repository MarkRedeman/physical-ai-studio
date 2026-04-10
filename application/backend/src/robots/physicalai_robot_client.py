import asyncio
from typing import Any

import numpy as np
from loguru import logger
from physicalai.robot import Robot

from robots.robot_client import RobotClient
from schemas.robot import RobotType

# Timeout for hardware operations (seconds)
HARDWARE_TIMEOUT_CONNECT = 10.0
HARDWARE_TIMEOUT_COMMAND = 5.0


class PhysicalAIRobotClient(RobotClient):
    """RobotClient adapter for any physicalai Robot protocol implementation.

    Bridges the synchronous physicalai Robot protocol into the async
    RobotClient interface used by the backend. All blocking hardware
    calls are dispatched via asyncio.to_thread with a bus lock to
    ensure thread-safety.

    Joint positions are converted between radians (physicalai) and
    degrees (backend convention).
    """

    def __init__(
        self,
        robot: Robot,
        robot_type_value: RobotType,
        joint_names: list[str],
        name: str,
    ) -> None:
        self._robot = robot
        self._robot_type = robot_type_value
        self._joint_names = joint_names
        self.name = name
        self._connected = False
        self._bus_lock = asyncio.Lock()

    @property
    def robot_type(self) -> RobotType:
        return self._robot_type

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def connect(self) -> None:
        """Connect to the robot via the physicalai driver."""
        logger.info(f"Connecting PhysicalAIRobotClient '{self.name}' ({self._robot_type})")
        try:
            async with self._bus_lock, asyncio.timeout(HARDWARE_TIMEOUT_CONNECT):
                await asyncio.to_thread(self._robot.connect)
            self._connected = True
            logger.info(f"PhysicalAIRobotClient '{self.name}' connected")
        except TimeoutError:
            logger.error(f"Timeout connecting to '{self.name}'")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to '{self.name}': {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from the robot via the physicalai driver."""
        logger.info(f"Disconnecting PhysicalAIRobotClient '{self.name}'")
        try:
            async with self._bus_lock, asyncio.timeout(HARDWARE_TIMEOUT_COMMAND):
                await asyncio.to_thread(self._robot.disconnect)
            logger.info(f"PhysicalAIRobotClient '{self.name}' disconnected")
        except TimeoutError:
            logger.warning(f"Timeout during disconnect of '{self.name}' - forcing cleanup")
        except Exception as e:
            logger.error(f"Error during disconnect of '{self.name}': {e}")
        finally:
            self._connected = False

    async def ping(self) -> dict:
        """Send ping command. Returns event dict with timestamp."""
        return self._create_event("pong")

    async def set_joints_state(self, joints: dict, goal_time: float) -> dict:  # noqa: ARG002
        """Set joint positions.

        Converts the {joint_name}.pos dict (degrees) to a numpy array
        (radians) and sends it via the physicalai Robot.send_action().
        """
        action = self._joints_dict_to_array(joints)
        async with self._bus_lock, asyncio.timeout(HARDWARE_TIMEOUT_COMMAND):
            await asyncio.to_thread(self._robot.send_action, action)
        return self._create_event("joints_state_was_set", joints=joints)

    async def enable_torque(self) -> dict:
        """Enable torque if the underlying robot supports it."""
        if hasattr(self._robot, "_set_torque"):
            logger.info(f"Enabling torque on '{self.name}'")
            async with self._bus_lock, asyncio.timeout(HARDWARE_TIMEOUT_COMMAND):
                await asyncio.to_thread(self._robot._set_torque, enabled=True)
        else:
            logger.debug(f"Torque control not available on '{self.name}', skipping")
        return self._create_event("torque_was_enabled")

    async def disable_torque(self) -> dict:
        """Disable torque if the underlying robot supports it."""
        if hasattr(self._robot, "_set_torque"):
            logger.info(f"Disabling torque on '{self.name}'")
            async with self._bus_lock, asyncio.timeout(HARDWARE_TIMEOUT_COMMAND):
                await asyncio.to_thread(self._robot._set_torque, enabled=False)
        else:
            logger.debug(f"Torque control not available on '{self.name}', skipping")
        return self._create_event("torque_was_disabled")

    async def read_state(self, *, normalize: bool = True) -> dict:  # noqa: ARG002
        """Read current robot state.

        Calls physicalai Robot.get_observation(), unpacks the state
        numpy array into a {joint_name}.pos dict with values in degrees.
        """
        try:
            async with self._bus_lock, asyncio.timeout(HARDWARE_TIMEOUT_COMMAND):
                observation: dict[str, Any] = await asyncio.to_thread(self._robot.get_observation)
            state = self._observation_to_state_dict(observation)
            return self._create_event(
                "state_was_updated",
                state=state,
                is_controlled=self._connected,
            )
        except Exception as e:
            logger.error(f"Robot read error on '{self.name}': {e}")
            raise

    async def read_forces(self) -> dict | None:
        """Read current robot forces. Not supported by physicalai Robot protocol."""
        return self._create_event(
            "force_was_updated",
            state=None,
            is_controlled=self._connected,
        )

    async def set_forces(self, forces: dict) -> dict:  # noqa: ARG002
        """Set robot forces. Not supported by physicalai Robot protocol."""
        raise NotImplementedError(f"set_forces is not supported by PhysicalAIRobotClient '{self.name}'")

    def features(self) -> list[str]:
        """Get robot features. Returns list of joint position keys."""
        return [f"{joint}.pos" for joint in self._joint_names]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _observation_to_state_dict(self, observation: dict[str, Any]) -> dict[str, float]:
        """Convert a physicalai observation to a {joint_name}.pos dict in degrees."""
        state_array: np.ndarray = observation["state"]
        return {f"{name}.pos": float(np.rad2deg(state_array[i])) for i, name in enumerate(self._joint_names)}

    def _joints_dict_to_array(self, joints: dict) -> np.ndarray:
        """Convert a {joint_name}.pos dict (degrees) to a numpy array (radians)."""
        action = np.zeros(len(self._joint_names), dtype=np.float32)
        for i, name in enumerate(self._joint_names):
            key = f"{name}.pos"
            if key in joints:
                action[i] = np.deg2rad(joints[key])
        return action
