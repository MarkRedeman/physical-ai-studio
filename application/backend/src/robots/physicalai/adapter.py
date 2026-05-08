import asyncio
from typing import Literal

import numpy as np
from loguru import logger
from physicalai.robot.so101 import SO101
from physicalai.robot.so101.constants import MAX_SPEED_RAD_S, RADIANS_PER_TICK, SO101_JOINT_ORDER
from physicalai.robot.trossen import BimanualWidowXAI, WidowXAI

from robots.robot_client import RobotClient
from schemas.calibration import Calibration
from schemas.robot import RobotType

HARDWARE_TIMEOUT_CONNECT = 10.0
HARDWARE_TIMEOUT_COMMAND = 5.0

RobotMode = Literal["follower", "leader", "teleoperator"]
PhysicalAIRobot = SO101 | WidowXAI | BimanualWidowXAI


def _clamp(value: float, limit: float) -> float:
    return max(min(value, limit), -limit)


def _clamp_joints(current: dict[str, float], target: dict[str, float], max_distance: float) -> dict[str, float]:
    return {key: value + _clamp(target[key] - value, max_distance) for key, value in current.items()}


class PhysicalAIRobotAdapter(RobotClient):
    name = "PhysicalAI"

    def __init__(
        self,
        robot: PhysicalAIRobot,
        mode: RobotMode,
        robot_type_follower: RobotType,
        robot_type_leader: RobotType,
        calibration: Calibration | None = None,
    ) -> None:
        self._robot = robot
        self._mode = mode
        self._bus_lock = asyncio.Lock()

        self._robot_type_follower = robot_type_follower
        self._robot_type_leader = robot_type_leader

        self.is_controlled: bool = mode == "follower"

        self._is_so101 = robot_type_follower in {RobotType.SO101_FOLLOWER}
        self._is_bimanual = robot_type_follower in {RobotType.TROSSEN_BIMANUAL_WIDOWXAI_FOLLOWER}
        self._is_widowxai = robot_type_follower in {RobotType.TROSSEN_WIDOWXAI_FOLLOWER}

        self.previous_target: dict[str, float] | None = None
        self._joint_params: dict[str, dict] = {}

        if self._is_so101:
            if calibration is None:
                raise ValueError("SO101 adapter requires calibration")
            self.name = "So101"
            self._init_so101_joint_params(calibration)
        elif self._is_bimanual:
            self.name = "BimanualWidowXAI"
        elif self._is_widowxai:
            self.name = "WidowXAI"
        else:
            raise ValueError(f"Unsupported robot type for PhysicalAIRobotAdapter: {robot_type_follower}")

    @property
    def robot_type(self) -> RobotType:
        if self._mode == "follower":
            return self._robot_type_follower
        return self._robot_type_leader

    @property
    def is_connected(self) -> bool:
        return self._robot.is_connected()

    async def connect(self) -> None:
        endpoint = self._connection_endpoint()
        logger.info(f"Connecting to {self.name} {self._mode} at {endpoint}")
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
        endpoint = self._connection_endpoint()
        logger.info(f"Disconnecting {self.name} {self._mode} at {endpoint}")
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
            raise RuntimeError("Cannot send actions to a leader robot")

        if self._is_so101:
            await self._move_so101_to_target(joints, goal_time)
        else:
            positions = self._state_to_action(joints)
            target_goal_time = 3 * goal_time if self._is_widowxai else goal_time
            async with self._bus_lock, asyncio.timeout(HARDWARE_TIMEOUT_COMMAND):
                await asyncio.to_thread(self._robot.send_action, positions, goal_time=target_goal_time)

        return self._create_event("joints_state_was_set", joints=joints)

    async def enable_torque(self) -> dict:
        if self._is_so101:
            logger.info("Enabling torque")
            async with self._bus_lock, asyncio.timeout(HARDWARE_TIMEOUT_COMMAND):
                await asyncio.to_thread(self._robot._set_torque, enabled=True)
        self.is_controlled = True
        return self._create_event("torque_was_enabled")

    async def disable_torque(self) -> dict:
        if self._is_so101:
            logger.info("Disabling torque")
            async with self._bus_lock, asyncio.timeout(HARDWARE_TIMEOUT_COMMAND):
                await asyncio.to_thread(self._robot._set_torque, enabled=False)
        self.is_controlled = False
        return self._create_event("torque_was_disabled")

    async def read_state(self, *, normalize: bool = True) -> dict:  # noqa: ARG002
        try:
            async with self._bus_lock, asyncio.timeout(HARDWARE_TIMEOUT_COMMAND):
                obs = await asyncio.to_thread(self._robot.get_observation)
            state = self._observation_to_state(obs)
            return self._create_event(
                "state_was_updated",
                state=state,
                is_controlled=self.is_controlled,
            )
        except Exception as e:
            logger.error(f"Robot read error: {e}")
            raise

    async def read_forces(self) -> dict | None:
        if self._is_so101:
            return self._create_event(
                "force_was_updated",
                state=None,
                is_controlled=self.is_controlled,
            )

        if self._is_widowxai and self._mode == "leader":
            return None
        if self._is_bimanual and self._mode != "follower":
            return None

        async with self._bus_lock, asyncio.timeout(HARDWARE_TIMEOUT_COMMAND):
            obs = await asyncio.to_thread(self._robot.get_observation)

        sensor_data = obs.sensor_data
        if sensor_data is None:
            raise RuntimeError("Robot observation is missing sensor data")

        forces = {}
        for i, name in enumerate(self._robot.joint_names):
            forces[f"{name}.eff"] = float(sensor_data["efforts"][i])

        return self._create_event(
            "force_was_updated",
            state=forces,
            is_controlled=self.is_controlled,
        )

    async def set_forces(self, forces: dict) -> dict:
        if self._is_so101:
            raise NotImplementedError("Force control is not implemented for SO101")

        if self._is_widowxai and self._mode == "follower":
            logger.warning("Cannot send forces to a follower arm")
            return forces
        if self._is_bimanual and self._mode == "follower":
            logger.warning("Cannot send forces to a follower robot")
            return forces

        efforts = np.zeros(len(self._robot.joint_names), dtype=np.float64)
        for i, name in enumerate(self._robot.joint_names):
            efforts[i] = forces.get(f"{name}.eff", 0.0)

        async with self._bus_lock, asyncio.timeout(HARDWARE_TIMEOUT_COMMAND):
            await asyncio.to_thread(self._robot.set_external_efforts, efforts, 0.1)

        return forces

    def features(self) -> list[str]:
        if self._is_so101:
            return [f"{name}.pos" for name in SO101_JOINT_ORDER]

        positions: list[str] = [f"{name}.pos" for name in self._robot.joint_names]
        velocities: list[str] = [f"{name}.vel" for name in self._robot.joint_names]
        return positions + velocities

    def _observation_to_state(self, obs) -> dict[str, float]:
        if self._is_so101:
            return self._radians_to_normalized(obs.joint_positions)

        result: dict[str, float] = {}
        sensor_data = obs.sensor_data
        if sensor_data is None:
            raise RuntimeError("Robot observation is missing sensor data")

        for i, name in enumerate(self._robot.joint_names):
            if self._is_gripper_joint(name):
                pos = float(obs.joint_positions[i])
            else:
                pos = float(np.rad2deg(obs.joint_positions[i]))

            vel = float(sensor_data["velocities"][i])
            result[f"{name}.pos"] = pos
            result[f"{name}.vel"] = vel

        return result

    def _state_to_action(self, joints: dict) -> np.ndarray:
        if self._is_so101:
            return self._normalized_to_radians(joints)

        positions = np.zeros(len(self._robot.joint_names), dtype=np.float64)

        for i, name in enumerate(self._robot.joint_names):
            if self._is_gripper_joint(name):
                positions[i] = joints[f"{name}.pos"]
            else:
                positions[i] = np.deg2rad(joints[f"{name}.pos"])

        return positions

    def _is_gripper_joint(self, joint_name: str) -> bool:
        if self._is_bimanual:
            return joint_name.split("_", 1)[1] == "gripper"
        return joint_name == "gripper"

    def _connection_endpoint(self) -> str:
        if self._is_so101:
            return self._robot.port
        if self._is_widowxai:
            return self._robot.ip
        return "bimanual"

    def _init_so101_joint_params(self, calibration: Calibration) -> None:
        for name in SO101_JOINT_ORDER:
            cal_val = calibration.values[name]
            direction = -1 if cal_val.drive_mode == 1 else 1
            rng = cal_val.range_max - cal_val.range_min
            is_gripper = name == "gripper"

            if rng == 0:
                scale = 0.0
                bias = 0.0
            elif is_gripper:
                scale = 100.0 / (direction * RADIANS_PER_TICK * rng)
                bias = ((cal_val.homing_offset - cal_val.range_min) / rng) * 100.0
            else:
                scale = 200.0 / (direction * RADIANS_PER_TICK * rng)
                bias = ((cal_val.homing_offset - cal_val.range_min) / rng) * 200.0 - 100.0

            if cal_val.drive_mode == 1 and not is_gripper:
                scale = -scale
                bias = -bias
            elif cal_val.drive_mode == 1 and is_gripper:
                scale = -scale
                bias = 100.0 - bias

            self._joint_params[name] = {
                "scale": scale,
                "bias": bias,
                "is_gripper": is_gripper,
            }

    def _radians_to_normalized(self, radians: np.ndarray) -> dict[str, float]:
        result: dict[str, float] = {}
        for i, name in enumerate(SO101_JOINT_ORDER):
            p = self._joint_params[name]
            norm = float(radians[i]) * p["scale"] + p["bias"]
            if p["is_gripper"]:
                result[f"{name}.pos"] = max(0.0, min(100.0, norm))
            else:
                result[f"{name}.pos"] = max(-100.0, min(100.0, norm))
        return result

    def _normalized_to_radians(self, joints: dict[str, float]) -> np.ndarray:
        result = np.empty(len(SO101_JOINT_ORDER), dtype=np.float32)
        for i, name in enumerate(SO101_JOINT_ORDER):
            p = self._joint_params[name]
            result[i] = (joints[f"{name}.pos"] - p["bias"]) / p["scale"] if p["scale"] != 0 else 0.0
        return result

    async def _move_so101_to_target(self, joints: dict, goal_time: float) -> None:
        max_rad = MAX_SPEED_RAD_S * goal_time

        async with self._bus_lock, asyncio.timeout(HARDWARE_TIMEOUT_COMMAND):
            obs = await asyncio.to_thread(self._robot.get_observation)
        current_rad = obs.joint_positions

        target_rad = self._normalized_to_radians(joints)

        if self.previous_target is not None:
            prev_rad = self._normalized_to_radians(self.previous_target)
            for i in range(len(current_rad)):
                current_rad[i] = current_rad[i] + _clamp(prev_rad[i] - current_rad[i], max_rad * 2)

        clamped_rad = np.array(
            [current_rad[i] + _clamp(target_rad[i] - current_rad[i], max_rad) for i in range(len(current_rad))],
            dtype=np.float32,
        )

        self.previous_target = self._radians_to_normalized(clamped_rad)

        async with self._bus_lock, asyncio.timeout(HARDWARE_TIMEOUT_COMMAND):
            await asyncio.to_thread(self._robot.send_action, clamped_rad)
