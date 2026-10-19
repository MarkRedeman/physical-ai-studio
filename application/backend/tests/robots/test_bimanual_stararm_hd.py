from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from robots.catalog.bimanual_stararm_hd import BimanualStarArmHD
from schemas.robot import RobotAdapter

if TYPE_CHECKING:
    from physicalai.capture.frame import Frame


@dataclass
class _Observation:
    joint_positions: np.ndarray
    timestamp: float
    sensor_data: dict[str, np.ndarray] | None = None
    images: dict[str, Frame] | None = None

    @property
    def state(self) -> np.ndarray:
        return self.joint_positions


class _StarArmHD:
    joint_names = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_yaw",
        "wrist_roll",
        "gripper",
    ]

    def __init__(self, values: list[float], timestamp: float) -> None:
        self._observation = _Observation(np.array(values, dtype=np.float32), timestamp)

    @property
    def device_ids(self) -> tuple[str, ...]:
        return ()

    def connect(self) -> None:
        pass

    def disconnect(self) -> None:
        pass

    def is_connected(self) -> bool:
        return True

    def get_observation(self) -> _Observation:
        return self._observation

    def send_action(self, action: np.ndarray, *, goal_time: float = 0.1) -> None:
        del action, goal_time


def test_bimanual_stararm_hd_maps_each_arm_into_widowx_action_space() -> None:
    robot = BimanualStarArmHD(
        left=_StarArmHD([1, 2, 3, 4, 5, 6, 55], 10),
        right=_StarArmHD([-1, -2, -3, -4, -5, -6, 0], 11),
    )

    observation = robot.get_observation()

    assert robot.joint_names == [
        "left_shoulder_pan",
        "left_shoulder_lift",
        "left_elbow_flex",
        "left_wrist_flex",
        "left_wrist_yaw",
        "left_wrist_roll",
        "left_gripper",
        "right_shoulder_pan",
        "right_shoulder_lift",
        "right_elbow_flex",
        "right_wrist_flex",
        "right_wrist_yaw",
        "right_wrist_roll",
        "right_gripper",
    ]
    np.testing.assert_allclose(
        observation.joint_positions,
        [-1, 2, -3, -4, 5, -6, 0.05, 1, -2, 3, 4, -5, 6, 0],
    )
    assert observation.timestamp == 10


def test_bimanual_stararm_hd_leader_is_registered_with_two_connections() -> None:
    robot = RobotAdapter.validate_python(
        {
            "id": "00000000-0000-0000-0000-000000000001",
            "name": "Dual HD",
            "type": "Bimanual_StarArm_102_HD_Leader",
            "payload": {
                "connection_string_left": "/dev/ttyUSB0",
                "connection_string_right": "/dev/ttyUSB1",
            },
        }
    )

    assert robot.type == "Bimanual_StarArm_102_HD_Leader"
