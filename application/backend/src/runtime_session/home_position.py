"""Home-pose compatibility and motion supervision, without owning a robot connection."""

from __future__ import annotations

import hashlib
import json
import math
from collections import deque
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np

from schemas.home_position import HomeCapability, HomeJointLimits, HomeMotion, HomePose

if TYPE_CHECKING:
    from collections.abc import Callable
    from uuid import UUID

    from runtime_session.store import Sample
    from workers.robot_worker import RobotRecipe

HOME_EVENTS = "evt/home_motion"
MAX_SAMPLE_AGE_S = 0.25
MAX_MOVE_S = 60.0
SETTLE_S = 0.3
SETTLE_TIMEOUT_S = 3.0
_ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")


def home_key(robot_id: UUID, operation: str) -> str:
    """Address a home operation on the already-owned follower worker."""
    return f"rpc/home/{robot_id}/{operation}"


def capability_for_so101(recipe: RobotRecipe, serial_number: str) -> HomeCapability | None:  # noqa: PLR0911 - fail closed at each capability gate
    """Enable only the calibrated single-arm position adapter with a verified serial identity.

    Limits are derived from calibration; speeds are deliberately conservative software limits,
    not a collision-free path guarantee. A pose does not cross unit/calibration/identity changes.
    """
    if not serial_number or recipe.driver.get("class_path") != "physicalai_so101_plugin.SO101":
        return None
    args = recipe.driver.get("init_args", {})
    if args.get("role", "follower") != "follower" or args.get("allow_uncalibrated", False):
        return None
    if args.get("disable_torque_on_disconnect", False):
        return None
    unit = args.get("unit", "normalized")
    calibration = args.get("calibration")
    if unit not in {"normalized", "degrees"} or not isinstance(calibration, dict):
        return None
    if recipe.joint_names != tuple(f"{name}.pos" for name in (*_ARM_JOINTS, "gripper")):
        return None
    from physicalai_so101_plugin import SO101Calibration

    try:
        checked = SO101Calibration.from_dict(calibration)
    except (TypeError, ValueError, KeyError):
        return None
    limits = {}
    for name in _ARM_JOINTS:
        joint = checked.joints[name]
        degrees_span = (joint.range_max - joint.range_min) * 360 / 4095
        scale = 1.0 if unit == "degrees" else 200 / degrees_span
        bound = degrees_span / 2 if unit == "degrees" else 100.0
        limits[f"{name}.pos"] = HomeJointLimits(
            minimum=-bound,
            maximum=bound,
            velocity=min(10 * scale, 5.0) if unit == "normalized" else 10.0,
            acceleration=min(20 * scale, 10.0) if unit == "normalized" else 20.0,
            tolerance=0.5 * scale,
            tracking_error=2 * scale,
        )
    identity = {
        "robot_id": str(recipe.robot_id),
        "serial": serial_number,
        "unit": unit,
        "calibration": checked.to_dict(),
    }
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return HomeCapability(robot_id=recipe.robot_id, fingerprint=fingerprint, unit=unit, joints=limits)


class HomeReturn:
    """Runs on the follower worker's loop; every command is checked against measured feedback."""

    def __init__(
        self,
        capability: HomeCapability,
        names: tuple[str, ...],
        publish: Callable[[HomeMotion], None],
        *,
        hz: float = 100,
    ) -> None:
        self.capability = capability
        self.names = names
        self.publish = publish
        self.motion: HomeMotion | None = None
        self.latest: Sample | None = None
        self.history: deque[Sample] = deque(maxlen=100)
        self._start: np.ndarray | None = None
        self._target: np.ndarray | None = None
        self._last_command: np.ndarray | None = None
        self._started = 0.0
        self._duration = 0.0
        self._lease = 0.0
        self._settled_at: float | None = None
        self._last_publish = 0.0
        self._period = 1 / hz
        self._last_tick = 0.0
        self._path_elapsed = 0.0
        self._sequence = 0

    @property
    def moving(self) -> bool:
        return self.motion is not None and self.motion.state == "moving"

    def observe(self, sample: Sample, now: float) -> None:
        if (
            sample.names != self.names
            or sample.value.shape != (len(self.names),)
            or not np.all(np.isfinite(sample.value))
        ):
            self._finish("failed", "Robot returned invalid joint observations.")
            raise ValueError("Robot returned invalid joint observations.")
        if now - sample.timestamp > MAX_SAMPLE_AGE_S or sample.timestamp > now + 0.05:
            self._finish("failed", "Robot observations are stale.")
            raise ValueError("Robot observations are stale.")
        self.latest = sample
        self.history.append(sample)

    def _sample(self, now: float) -> Sample:
        if self.latest is None or now - self.latest.timestamp > MAX_SAMPLE_AGE_S:
            raise ValueError("Wait for fresh follower observations.")
        return self.latest

    def validate(self, pose: HomePose) -> None:
        capability = self.capability
        if (pose.robot_id, pose.fingerprint, pose.unit) != (
            capability.robot_id,
            capability.fingerprint,
            capability.unit,
        ):
            raise ValueError("Home belongs to a different robot, calibration, or joint unit. Set a new home.")
        if set(pose.positions) != set(capability.joints):
            raise ValueError("Home must contain every arm joint and no gripper commands.")
        for name, value in pose.positions.items():
            limit = capability.joints[name]
            if not math.isfinite(value) or not limit.minimum <= value <= limit.maximum:
                raise ValueError(f"Home joint {name} is outside its calibrated limits.")

    def capture(self, now: float) -> HomePose:
        if self.moving:
            raise ValueError("Stop the return before replacing home.")
        sample = self._sample(now)
        recent = [entry for entry in self.history if sample.timestamp - entry.timestamp <= SETTLE_S + 0.05]
        if not recent or sample.timestamp - recent[0].timestamp < SETTLE_S:
            raise ValueError("Hold the follower still briefly before setting home.")
        for name, limit in self.capability.joints.items():
            index = self.names.index(name)
            if np.ptp([entry.value[index] for entry in recent]) > limit.tolerance:
                raise ValueError("Hold the follower still briefly before setting home.")
        pose = HomePose(
            robot_id=self.capability.robot_id,
            fingerprint=self.capability.fingerprint,
            unit=self.capability.unit,
            positions={name: float(sample.value[self.names.index(name)]) for name in self.capability.joints},
            captured_at=datetime.now(UTC),
        )
        self.validate(pose)
        return pose

    def start(self, operation_id: UUID, pose: HomePose, now: float) -> HomeMotion:
        if self.motion is not None and self.motion.operation_id == operation_id:
            return self.motion  # Repeated transport requests never restart a trajectory.
        if self.moving:
            raise ValueError("A return to home is already running.")
        self.validate(pose)
        sample = self._sample(now)
        start = sample.value.copy()
        if not 0 <= start[self.names.index("gripper.pos")] <= 100:
            raise ValueError("Gripper observations are outside the supported range.")
        target = start.copy()  # The measured gripper stays unchanged for the entire move.
        duration = 2.0
        for name, limit in self.capability.joints.items():
            index = self.names.index(name)
            if not limit.minimum <= start[index] <= limit.maximum:
                raise ValueError(f"Current joint {name} is outside its calibrated limits.")
            target[index] = pose.positions[name]
            distance = abs(target[index] - start[index])
            # Quintic smoothstep peak derivatives: 1.875 and 10/sqrt(3).
            duration = max(
                duration,
                1.875 * distance / limit.velocity,
                math.sqrt((10 / math.sqrt(3)) * distance / limit.acceleration),
            )
        if duration + SETTLE_TIMEOUT_S > MAX_MOVE_S:
            raise ValueError("This return exceeds the maximum motion duration. Move closer and set a new home.")
        self._start, self._target, self._last_command = start, target, start.copy()
        self._started, self._duration, self._lease = now, duration, now
        self._last_tick, self._path_elapsed = now, 0.0
        self._sequence = 0
        self._settled_at = None
        self.motion = HomeMotion(operation_id=operation_id, robot_id=self.capability.robot_id, state="moving")
        if all(
            abs(target[self.names.index(name)] - start[self.names.index(name)]) <= limit.tolerance
            for name, limit in self.capability.joints.items()
        ):
            self.motion = self.motion.model_copy(
                update={"state": "reached", "progress": 1.0, "message": "The follower is already at home."}
            )
        self.publish_motion()
        return self.motion

    def renew(self, operation_id: UUID, now: float) -> None:
        if not self.moving or self.motion.operation_id != operation_id:
            raise ValueError("That home return is no longer active.")
        # A late renewal cannot resurrect an expired operation.
        if now - self._lease > self.capability.lease_timeout_s:
            self.cancel(operation_id, "Browser heartbeat expired.", failed=True)
            raise ValueError("The home return expired.")
        self._lease = now

    def cancel(self, operation_id: UUID, message: str = "Return stopped.", *, failed: bool = False) -> HomeMotion:
        if self.motion is None or self.motion.operation_id != operation_id:
            raise ValueError("That home return is no longer active.")
        if self.moving:
            self._finish("failed" if failed else "cancelled", message)
        return self.motion

    def _finish(self, state: str, message: str | None = None) -> None:
        if not self.moving:
            return
        self.motion = self.motion.model_copy(update={"state": state, "message": message})
        self.publish_motion()

    def publish_motion(self) -> None:
        """Publish ordered state so delayed progress cannot overwrite completion or cancellation."""
        self._sequence += 1
        self.motion = self.motion.model_copy(update={"sequence": self._sequence})
        self.publish(self.motion)

    def command(self, now: float) -> np.ndarray | None:  # noqa: PLR0911 - each safety failure stops independently
        if not self.moving:
            return None
        sample = self._sample(now)
        if now - self._last_tick > 0.1 or now < self._last_tick:
            self._finish("failed", "Robot control timing was interrupted; return stopped.")
            return sample.value.copy()
        if now - self._lease > self.capability.lease_timeout_s:
            self._finish("failed", "Browser heartbeat expired; return stopped.")
            return sample.value.copy()
        for name, limit in self.capability.joints.items():
            index = self.names.index(name)
            if abs(sample.value[index] - self._last_command[index]) > limit.tracking_error:
                self._finish("failed", f"{name} is not following the commanded motion; return stopped.")
                return sample.value.copy()
        elapsed = now - self._started
        if elapsed > self._duration + SETTLE_TIMEOUT_S:
            self._finish("failed", "Home was not reached within the motion timeout.")
            return sample.value.copy()
        # Never catch up with a large position jump after a delayed worker tick.
        self._path_elapsed += min(now - self._last_tick, self._period)
        self._last_tick = now
        fraction = min(self._path_elapsed / self._duration, 1.0)
        blend = fraction**3 * (10 - 15 * fraction + 6 * fraction**2)
        command = self._start + (self._target - self._start) * blend
        if fraction == 1.0:
            reached = all(
                abs(sample.value[self.names.index(name)] - self._target[self.names.index(name)]) <= limit.tolerance
                for name, limit in self.capability.joints.items()
            )
            self._settled_at = (self._settled_at if self._settled_at is not None else now) if reached else None
            if self._settled_at is not None and now - self._settled_at >= SETTLE_S:
                self.motion = self.motion.model_copy(update={"progress": 1.0})
                self._finish("reached", "Home reached. The follower is held; teleoperation remains off.")
                return sample.value.copy()
        self._last_command = command.copy()
        self.motion = self.motion.model_copy(update={"progress": fraction})
        if now - self._last_publish >= 0.1:
            self._last_publish = now
            self.publish_motion()
        return command
