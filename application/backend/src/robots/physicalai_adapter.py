import asyncio
from typing import Literal

import numpy as np
from loguru import logger
from physicalai.robot.interface import Robot, RobotObservation

from robots.robot_client import RobotClient
from schemas.robot import RobotType

HARDWARE_TIMEOUT_CONNECT = 10.0
HARDWARE_TIMEOUT_COMMAND = 5.0

RobotMode = Literal["follower", "leader", "teleoperator"]


class PhysicalAIRobotAdapter(RobotClient):
    name = "PhysicalAIRobot"

    def __init__(
        self,
        *,
        robot: Robot,
        mode: RobotMode,
        follower_type: RobotType,
        leader_type: RobotType,
        include_velocities: bool = False,
        convert_non_gripper_rad_to_deg: bool = False,
        pass_goal_time: bool = False,
        goal_time_scale: float = 1.0,
        emit_force_event_when_none: bool = False,
        external_effort_gain: float = 0.1,
    ) -> None:
        self._robot = robot
        self._mode = mode
        self._follower_type = follower_type
        self._leader_type = leader_type
        self._include_velocities = include_velocities
        self._convert_non_gripper_rad_to_deg = convert_non_gripper_rad_to_deg
        self._pass_goal_time = pass_goal_time
        self._goal_time_scale = goal_time_scale
        self._emit_force_event_when_none = emit_force_event_when_none
        self._external_effort_gain = external_effort_gain
        self._bus_lock = asyncio.Lock()
        self.is_controlled = False

    def _position_from_robot(self, name: str, value: float) -> float:
        if self._convert_non_gripper_rad_to_deg and name != "gripper":
            return float(np.rad2deg(value))
        return value

    def _position_to_robot(self, name: str, value: float) -> float:
        if self._convert_non_gripper_rad_to_deg and name != "gripper":
            return float(np.deg2rad(value))
        return value

    def _observation_to_state(self, observation: RobotObservation) -> dict[str, float]:
        state: dict[str, float] = {}
        for i, name in enumerate(self._robot.joint_names):
            raw_position = float(observation.joint_positions[i])
            state[f"{name}.pos"] = self._position_from_robot(name, raw_position)

        if self._include_velocities:
            sensor_data = observation.sensor_data
            if sensor_data is None or "velocities" not in sensor_data:
                msg = "Robot observation is missing velocity data"
                raise RuntimeError(msg)
            velocities = sensor_data["velocities"]
            for i, name in enumerate(self._robot.joint_names):
                state[f"{name}.vel"] = float(velocities[i])

        return state

    def _state_to_action(self, joints: dict[str, float]) -> np.ndarray:
        action = np.empty(len(self._robot.joint_names), dtype=np.float32)
        for i, name in enumerate(self._robot.joint_names):
            action[i] = self._position_to_robot(name, float(joints[f"{name}.pos"]))
        return action

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
        action = self._state_to_action(joints)
        async with self._bus_lock, asyncio.timeout(HARDWARE_TIMEOUT_COMMAND):
            if self._pass_goal_time:
                await asyncio.to_thread(self._robot.send_action, action, goal_time=self._goal_time_scale * goal_time)
            else:
                await asyncio.to_thread(self._robot.send_action, action)

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
                observation = await asyncio.to_thread(self._robot.get_observation)
            state = self._observation_to_state(observation)

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
            observation = await asyncio.to_thread(self._robot.get_observation)

        sensor_data = observation.sensor_data
        if sensor_data is None or "efforts" not in sensor_data:
            if self._emit_force_event_when_none:
                return self._create_event(
                    "force_was_updated",
                    state=None,
                    is_controlled=self.is_controlled,
                )
            return None

        efforts = sensor_data["efforts"]
        forces = {f"{name}.eff": float(efforts[i]) for i, name in enumerate(self._robot.joint_names)}

        return self._create_event(
            "force_was_updated",
            state=forces,
            is_controlled=self.is_controlled,
        )

    async def set_forces(self, forces: dict) -> dict:
        if self._mode == "follower":
            logger.warning("Cannot send forces to a follower arm")
            return forces

        set_external_efforts = getattr(self._robot, "set_external_efforts", None)
        if not callable(set_external_efforts):
            return forces

        efforts = np.zeros(len(self._robot.joint_names), dtype=np.float32)
        for i, name in enumerate(self._robot.joint_names):
            efforts[i] = float(forces.get(f"{name}.eff", 0.0))

        async with self._bus_lock, asyncio.timeout(HARDWARE_TIMEOUT_COMMAND):
            await asyncio.to_thread(set_external_efforts, efforts, gain=self._external_effort_gain)
        return forces

    def features(self) -> list[str]:
        position_features = [f"{name}.pos" for name in self._robot.joint_names]
        if not self._include_velocities:
            return position_features
        velocity_features = [f"{name}.vel" for name in self._robot.joint_names]
        return position_features + velocity_features
