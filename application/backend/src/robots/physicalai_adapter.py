import asyncio
from typing import Literal, Protocol

from loguru import logger

from robots.robot_client import RobotClient
from schemas.robot import RobotType

HARDWARE_TIMEOUT_CONNECT = 10.0
HARDWARE_TIMEOUT_COMMAND = 5.0

RobotMode = Literal["follower", "leader", "teleoperator"]


class PhysicalAIRobotDriver(Protocol):
    @property
    def joint_names(self) -> list[str]: ...

    def connect(self) -> None: ...

    def disconnect(self) -> None: ...

    def is_connected(self) -> bool: ...

    def read_state_dict(self) -> dict[str, float]: ...

    def send_state_dict(self, joints: dict[str, float], goal_time: float) -> None: ...

    def read_force_dict(self) -> dict[str, float] | None: ...

    def set_force_dict(self, forces: dict[str, float]) -> dict: ...

    def feature_names(self) -> list[str]: ...


class PhysicalAIRobotAdapter(RobotClient):
    name = "PhysicalAIRobot"

    def __init__(
        self,
        *,
        robot: PhysicalAIRobotDriver,
        mode: RobotMode,
        follower_type: RobotType,
        leader_type: RobotType,
        emit_force_event_when_none: bool = False,
        delegate_torque: bool = False,
    ) -> None:
        self._robot = robot
        self._mode = mode
        self._follower_type = follower_type
        self._leader_type = leader_type
        self._emit_force_event_when_none = emit_force_event_when_none
        self._delegate_torque = delegate_torque
        self._bus_lock = asyncio.Lock()
        self.is_controlled = False

    @property
    def robot_type(self) -> RobotType:
        if self._mode == "follower":
            return self._follower_type
        return self._leader_type

    @property
    def is_connected(self) -> bool:
        return self._robot.is_connected()

    async def connect(self) -> None:
        logger.info(f"Connecting physicalai robot in {self._mode} mode")
        try:
            async with self._bus_lock, asyncio.timeout(HARDWARE_TIMEOUT_CONNECT):
                await asyncio.to_thread(self._robot.connect)
            self.is_controlled = self._mode == "follower"
        except TimeoutError:
            logger.error("Timeout connecting to robot")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to robot: {e}")
            raise

    async def disconnect(self) -> None:
        logger.info(f"Disconnecting physicalai robot in {self._mode} mode")
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
        async with self._bus_lock, asyncio.timeout(HARDWARE_TIMEOUT_COMMAND):
            await asyncio.to_thread(self._robot.send_state_dict, joints, goal_time)

        return self._create_event("joints_state_was_set", joints=joints)

    async def enable_torque(self) -> dict:
        if self._delegate_torque:
            set_torque = getattr(self._robot, "set_torque", None)
            if set_torque is None:
                msg = "Robot driver does not provide set_torque"
                raise RuntimeError(msg)
            async with self._bus_lock, asyncio.timeout(HARDWARE_TIMEOUT_COMMAND):
                await asyncio.to_thread(set_torque, enabled=True)
        self.is_controlled = True
        return self._create_event("torque_was_enabled")

    async def disable_torque(self) -> dict:
        if self._delegate_torque:
            set_torque = getattr(self._robot, "set_torque", None)
            if set_torque is None:
                msg = "Robot driver does not provide set_torque"
                raise RuntimeError(msg)
            async with self._bus_lock, asyncio.timeout(HARDWARE_TIMEOUT_COMMAND):
                await asyncio.to_thread(set_torque, enabled=False)
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

        if forces is None and not self._emit_force_event_when_none:
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
