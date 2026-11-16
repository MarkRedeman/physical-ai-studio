"""A config-exportable robot driver for the robot worker tests.

It is rebuilt from its config in a spawned child, so it reports through files:
every ``send_action`` appends a JSON line to ``log``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from physicalai.config import export_config


@dataclass
class FakeObservation:
    joint_positions: np.ndarray
    timestamp: float
    sensor_data: dict[str, np.ndarray] | None = None


@export_config
class FakeRobot:
    def __init__(
        self,
        log: str,
        joints: list[str],
        start: list[float],
        fail_connect: bool = False,
        efforts: list[float] | None = None,
    ) -> None:
        self.log = log
        self._joints = joints
        self._position = np.asarray(start, dtype=np.float64)
        self._fail_connect = fail_connect
        self._connected = False
        self.efforts = efforts

    @property
    def joint_names(self) -> list[str]:
        return self._joints

    @property
    def device_ids(self) -> tuple[str, ...]:
        return (f"fake:{self.log}",)

    def connect(self) -> None:
        if self._fail_connect:
            raise ConnectionError("Failed to open serial port /dev/fake")
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def get_observation(self) -> FakeObservation:
        sensors = {"efforts": np.asarray(self.efforts)} if self.efforts is not None else None
        return FakeObservation(self._position.copy(), time.monotonic(), sensors)

    def send_action(self, action: np.ndarray, *, goal_time: float = 0.1) -> None:
        self._position = np.asarray(action, dtype=np.float64)
        with Path(self.log).open("a") as log:
            log.write(json.dumps({"action": self._position.tolist(), "goal_time": goal_time}) + "\n")


def sent_actions(log: Path) -> list[dict]:
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


class FeedbackRobot(FakeRobot):
    """A leader implementing the optional effort-feedback contract."""

    def set_external_efforts(self, efforts: np.ndarray, gain: float = 1.0) -> None:
        with Path(self.log + ".efforts").open("a") as log:
            log.write(json.dumps({"efforts": efforts.tolist(), "gain": gain}) + "\n")
