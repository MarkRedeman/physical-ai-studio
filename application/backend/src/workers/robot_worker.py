"""One robot in its own process: measure into ``state``, follow ``action``.

The worker does not know what writes ``action``; a leader arm, a policy or a
move to a home position all look the same from here. On connect it seeds
``action`` with the measured position, so with nothing driving it the arm holds
still. An action stamped in the future is approached linearly: each tick moves
the last command ``1/hz / time left`` of the way to the action, so it arrives at
the stamp. Anything stamped within a tick is sent as is. The worker interpolates
because drivers may ignore ``goal_time`` (SO-101 does).

Any exception ends the worker. Nothing retries silently.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager, contextmanager, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from loguru import logger

from exceptions import RobotDeviceAlreadyOwnedError
from robots.features import feature_names
from runtime_session.bus import Bus, publish_status
from runtime_session.home_position import HOME_EVENTS, HomeReturn, home_key
from runtime_session.store import ACTION, STATE, FeatureSpec, Sample, Store
from schemas.home_position import (
    CancelHomeReturnCommand,
    CaptureHomePositionCommand,
    HomeCapability,
    HomeMotion,
    HomePose,
    RenewHomeReturnCommand,
    ReturnHomeCommand,
)
from workers.base import ProcessWorker
from workers.rate import Rate

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator
    from uuid import UUID

    from physicalai.robot import Robot

DEFAULT_ROBOT_HZ = 100.0


@dataclass(frozen=True, slots=True)
class RobotRecipe:
    """What a child needs to build the robot. Plain data, because it is pickled."""

    robot_id: UUID
    driver: dict[str, Any]
    """The driver as a ``physicalai.config`` dict; rebuilt in the child."""
    joint_names: tuple[str, ...]
    """Store names (``<joint>.pos``), in the driver's order."""
    device_ids: tuple[str, ...]

    @classmethod
    def from_driver(cls, robot_id: UUID, driver: Robot) -> RobotRecipe:
        """Describe a driver that was built but not connected."""
        from physicalai.config import Config

        return cls(
            robot_id=robot_id,
            driver=Config.from_instance(driver).to_dict(),
            joint_names=tuple(feature_names(list(driver.joint_names))),
            device_ids=tuple(driver.device_ids),
        )

    @property
    def spec(self) -> FeatureSpec:
        return FeatureSpec(self.joint_names)


@contextmanager
def claim_robot(recipe: RobotRecipe) -> Iterator[None]:
    """Hold a robot's devices exclusively. See :func:`claim_devices`."""
    # Not the SharedRobot name: a SharedRobot seeing its name locked would try to join us.
    with claim_devices(f"studio-worker-{recipe.robot_id}", recipe.device_ids):
        yield


@contextmanager
def claim_devices(name: str, device_ids: tuple[str, ...]) -> Iterator[None]:
    """Hold devices exclusively, against workers, wizards and ``SharedRobot`` owners.

    Taken by the parent, whose open lock file dies with it, so a child that is
    killed cannot leave the device locked. ``name`` must be unique to the holder.

    Raises:
        RobotDeviceAlreadyOwnedError: Something else holds ``name`` or one of the devices.
    """
    # shortcut: physicalai's lock is private API; replace once physicalai exports a device claim.
    from physicalai.robot.transport._lock import DEVICE_KIND, LockContention, acquire_locks

    try:
        locks = acquire_locks(name, device_ids)
    except LockContention as error:
        raise RobotDeviceAlreadyOwnedError(
            device_ids=(error.identity,) if error.kind == DEVICE_KIND else device_ids or None
        ) from error
    try:
        yield
    finally:
        locks.release_all()


class RobotWorker(ProcessWorker["Robot"]):
    """Drive one robot from the store, or only observe it.

    ``observe_only`` never sends an action, for the robot form's live view.
    Connecting still applies the driver's own setup (an SO-101 follower
    enables torque), as attaching a ``SharedRobot`` did.
    """

    def __init__(  # noqa: PLR0913 - existing worker configuration plus optional home capability
        self,
        recipe: RobotRecipe,
        store_name: str,
        spec: FeatureSpec,
        *,
        hz: float,
        observe_only: bool = False,
        session_id: str | None = None,
        component: str | None = None,
        home_capability: HomeCapability | None = None,
    ) -> None:
        """Pass ``session_id`` and ``component`` to report a failure on that session's bus."""
        super().__init__(name=f"RobotWorker-{recipe.robot_id}")
        self.recipe = recipe
        self.store_name = store_name
        self.spec = spec
        self.hz = hz
        self.observe_only = observe_only
        self.session_id = session_id
        self.component = component
        self._bus: Bus | None = None
        self.home_capability = home_capability
        self._home: HomeReturn | None = None

    @asynccontextmanager
    async def lifecycle(self) -> AsyncIterator[Robot]:
        from physicalai.config import instantiate

        self._store = Store.attach(self.spec, self.store_name)
        try:
            with Bus.join(self.session_id) if self.session_id else nullcontext() as self._bus:
                driver = cast("Robot", instantiate(self.recipe.driver))
                driver.connect()
                self._driver = driver
                try:
                    self._check_joint_order(driver)
                    if self.home_capability is not None and not self.observe_only and self._bus is not None:
                        self._home = HomeReturn(
                            self.home_capability,
                            self.spec.joint_names,
                            lambda motion: self._bus.publish(HOME_EVENTS, motion),
                            hz=self.hz,
                        )
                        for operation, model, handler in (
                            ("capture", CaptureHomePositionCommand, self._capture_home),
                            ("return", ReturnHomeCommand, self._return_home),
                            ("renew", RenewHomeReturnCommand, self._renew_home),
                            ("cancel", CancelHomeReturnCommand, self._cancel_home),
                        ):
                            self._bus.serve(home_key(self.recipe.robot_id, operation), model, handler)
                    if not self.observe_only:
                        observation = driver.get_observation()
                        self._store.write(ACTION, observation.joint_positions, timestamp=observation.timestamp)
                    yield driver
                finally:
                    if self._home is not None and self._home.moving:
                        try:
                            self._cancel_home(
                                CancelHomeReturnCommand(request_id="", operation_id=self._home.motion.operation_id)
                            )
                        except Exception:
                            logger.exception("Could not hold follower during home-return shutdown")
                    driver.disconnect()
        finally:
            self._store.close()

    def _check_joint_order(self, driver: Robot) -> None:
        """The store's vectors follow the spec's order; a different robot would get the wrong joints."""
        connected = tuple(feature_names(list(driver.joint_names)))
        if connected != self.spec.joint_names:
            raise RuntimeError(f"Robot reports joints {connected}, the session expects {self.spec.joint_names}")

    async def run_loop(self, context: Robot) -> None:
        try:
            await self._drive(context)
        except Exception as error:
            if self._bus is not None and self.component is not None:
                publish_status(self._bus, self.component, "failed", str(error))
            raise

    async def _drive(self, context: Robot) -> None:
        rate = Rate(self.hz, name=self.name)
        tick = 1.0 / self.hz
        command = None  # the last one sent, not measured: measuring would feed tracking lag back in
        while not self.should_stop():
            observation = context.get_observation()
            self._store.write(STATE, observation.joint_positions, timestamp=observation.timestamp)
            if self._home is not None:
                now = time.monotonic()
                self._home.observe(
                    Sample(observation.joint_positions.copy(), self.spec.joint_names, observation.timestamp), now
                )
                home_command = self._home.command(now)
                if home_command is not None:
                    command = home_command
                    self._store.write(ACTION, command, timestamp=now)
                    context.send_action(command, goal_time=tick)
                    await rate.sleep()
                    continue
            action = None if self.observe_only else self._store.read(ACTION)
            if action is not None:
                if command is None:
                    command = observation.joint_positions
                remaining = action.timestamp - time.monotonic()
                command = command + (action.value - command) * (tick / remaining) if remaining > tick else action.value
                context.send_action(command, goal_time=tick)
            await rate.sleep()
        logger.info("Robot worker {} stopped", self.name)

    def _home_required(self) -> HomeReturn:
        if self._home is None:
            raise ValueError("Return to home is not supported by this follower configuration.")
        return self._home

    def _capture_home(self, request: CaptureHomePositionCommand) -> HomePose:
        if request.robot_id != self.recipe.robot_id:
            raise ValueError("That is not this session's follower.")
        return self._home_required().capture(time.monotonic())

    def _return_home(self, request: ReturnHomeCommand) -> HomeMotion:
        return self._home_required().start(request.operation_id, request.home, time.monotonic())

    def _hold_home(self) -> None:
        home = self._home_required()
        if home.latest is None:
            raise ValueError("No follower measurement is available to hold.")
        self._store.write(ACTION, home.latest.value, timestamp=time.monotonic())
        self._driver.send_action(home.latest.value, goal_time=1 / self.hz)

    def _renew_home(self, request: RenewHomeReturnCommand) -> HomeMotion:
        home = self._home_required()
        try:
            home.renew(request.operation_id, time.monotonic())
        finally:
            if not home.moving:
                self._hold_home()
        return home.motion

    def _cancel_home(self, request: CancelHomeReturnCommand) -> HomeMotion:
        home = self._home_required()
        motion = home.cancel(request.operation_id)
        try:
            self._hold_home()
        except Exception:
            self.request_stop()  # Do not continue issuing actions after a failed stop command.
            home.motion = motion.model_copy(
                update={"state": "failed", "message": "Could not command hold. Use the physical emergency stop."}
            )
            home.publish_motion()
            raise
        return motion
