"""Trossen WidowX AI Setup Worker — websocket-driven diagnostics and verification.

Unlike the SO101 setup worker, Trossen robots have no per-motor setup or calibration
flow. The Trossen SDK handles calibration internally (is_calibrated is always True).

This worker performs:
  1. IP reachability check (ping)
  2. Driver configuration (trossen_arm.TrossenArmDriver.configure) — this also homes
     the robot to its zero position as a side effect
  3. Joint position streaming for 3D preview verification

Architecture mirrors ``RobotWorker``: two concurrent asyncio tasks —
a *broadcast loop* that reads the driver and pushes events to the client,
and a *command loop* that receives commands from the client.

Streaming behaviour is controlled entirely by ``self.phase``:
  - ``VERIFICATION`` → stream positions + velocities as ``state_was_updated``
    events (degrees for body joints, meters for gripper).
  - All other phases → no streaming (the loop idles).
"""

import asyncio
import time
from enum import StrEnum
from typing import Any

import numpy as np
import trossen_arm
from loguru import logger

from robots.discovery.ip import IPDiscovery
from schemas.robot import RobotType
from workers.transport.worker_transport import WorkerTransport
from workers.transport_worker import TransportWorker, WorkerState

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_yaw",
    "wrist_roll",
    "gripper",
]

CONFIGURE_TIMEOUT = 10  # seconds — generous for network latency
FPS = 20  # Streaming rate for position reads (~20Hz, matching original Trossen rate)


def _end_effector_for_type(robot_type: str) -> trossen_arm.StandardEndEffector:
    """Map robot type string to the correct Trossen end-effector variant."""
    if robot_type == RobotType.TROSSEN_WIDOWXAI_LEADER:
        return trossen_arm.StandardEndEffector.wxai_v0_leader
    return trossen_arm.StandardEndEffector.wxai_v0_follower


# ---------------------------------------------------------------------------
# Setup phases
# ---------------------------------------------------------------------------


class TrossenSetupPhase(StrEnum):
    """Phases of the Trossen setup wizard state machine.

    The broadcast loop uses this to decide what to stream:
      - VERIFICATION → positions + velocities (degrees for body, meters for
        gripper) via ``state_was_updated`` events.
      - Everything else → no streaming (idle sleep).
    """

    CONNECTING = "connecting"
    DIAGNOSTICS = "diagnostics"
    VERIFICATION = "verification"


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


class TrossenSetupWorker(TransportWorker):
    """Websocket worker for the Trossen WidowX AI setup wizard.

    Protocol overview (client sends commands, server sends events):

    After connection the worker immediately runs IP ping + driver configuration
    and sends a diagnostics_result event. Then it starts two concurrent tasks:
    a broadcast loop (phase-driven streaming) and a command loop.

    Commands:
        {"command": "ping"}
        {"command": "re_probe"}           — re-run ping + configure check
        {"command": "enter_verification"} — transition to VERIFICATION phase

    Events sent to client:
        {"event": "status", "state": ..., "phase": ..., "message": ...}
        {"event": "diagnostics_result", ...}
        {"event": "state_was_updated", "state": {...}}  — joint positions for 3D preview
        {"event": "error", "message": ...}
    """

    def __init__(
        self,
        transport: WorkerTransport,
        robot_type: str,
        connection_string: str,
    ) -> None:
        super().__init__(transport)
        self.robot_type = robot_type
        self.connection_string = connection_string
        self.phase = TrossenSetupPhase.CONNECTING

        self.driver: trossen_arm.TrossenArmDriver | None = None
        self.end_effector = _end_effector_for_type(robot_type)
        self.is_leader = robot_type == RobotType.TROSSEN_WIDOWXAI_LEADER

        # Diagnostics results
        self.diagnostics_result: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _require_driver(self) -> trossen_arm.TrossenArmDriver:
        """Return the driver, raising if not connected."""
        if self.driver is None:
            raise RuntimeError("Trossen driver is not configured")
        return self.driver

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Main worker lifecycle.

        Connects to the driver, runs initial diagnostics (IP ping +
        configure), then starts two concurrent tasks:
          - ``_broadcast_loop`` — reads the driver and pushes events
          - ``_command_loop`` — receives and dispatches frontend commands
        """
        try:
            await self.transport.connect()
            self.state = WorkerState.RUNNING

            # Run diagnostics (ping + configure)
            await self._run_diagnostics()

            # Two concurrent tasks — mirrors RobotWorker architecture
            await self.run_concurrent(
                asyncio.create_task(self._broadcast_loop()),
                asyncio.create_task(self._command_loop()),
            )

        except Exception as e:
            self.state = WorkerState.ERROR
            self.error_message = str(e)
            logger.exception(f"Trossen setup worker error: {e}")
            await self._send_event("error", message=str(e))
        finally:
            await self._cleanup()
            await self.shutdown()

    # ------------------------------------------------------------------
    # Phase: Diagnostics (ping + configure)
    # ------------------------------------------------------------------

    async def _run_diagnostics(self) -> None:
        """Ping the IP and attempt to configure the Trossen driver."""
        self.phase = TrossenSetupPhase.CONNECTING
        await self._send_phase_status(f"Pinging {self.connection_string}...")

        # Step 1: IP reachability
        ip_reachable = await IPDiscovery.ping(self.connection_string, ping_timeout=2.0)

        if not ip_reachable:
            self.phase = TrossenSetupPhase.DIAGNOSTICS
            self.diagnostics_result = {
                "event": "diagnostics_result",
                "ip_reachable": False,
                "configure_ok": False,
                "motor_count": 0,
                "motor_names": [],
                "robot_type": self.robot_type,
                "connection_string": self.connection_string,
                "error_message": f"IP address {self.connection_string} is not reachable",
            }
            await self.transport.send_json(self.diagnostics_result)
            return

        await self._send_phase_status("IP reachable, configuring robot...")

        # Step 2: Configure the driver
        configure_ok = False
        error_message: str | None = None

        try:
            # Clean up any existing driver before re-configuring
            if self.driver is not None:
                try:
                    self.driver.cleanup()
                except Exception:
                    pass

            driver = trossen_arm.TrossenArmDriver()

            # Run configure in a thread with timeout
            await asyncio.wait_for(
                asyncio.to_thread(
                    driver.configure,
                    trossen_arm.Model.wxai_v0,
                    self.end_effector,
                    self.connection_string,
                    True,  # clear_on_cleanup
                    timeout=5,
                ),
                timeout=CONFIGURE_TIMEOUT,
            )

            # If we get here, configure succeeded — the robot is alive
            self.driver = driver
            configure_ok = True

            # Verify we can read positions (confirms all motors are responsive)
            positions = await asyncio.to_thread(driver.get_all_positions)
            logger.info(
                f"Trossen setup: configured successfully, "
                f"{len(positions)} motors responding at {self.connection_string}"
            )

        except TimeoutError:
            error_message = f"Configuration timed out after {CONFIGURE_TIMEOUT}s"
            logger.warning(f"Trossen setup: {error_message}")
        except Exception as e:
            error_message = f"Configuration failed: {e}"
            logger.warning(f"Trossen setup: {error_message}")

        self.phase = TrossenSetupPhase.DIAGNOSTICS
        self.diagnostics_result = {
            "event": "diagnostics_result",
            "ip_reachable": True,
            "configure_ok": configure_ok,
            "motor_count": len(MOTOR_NAMES) if configure_ok else 0,
            "motor_names": MOTOR_NAMES if configure_ok else [],
            "robot_type": self.robot_type,
            "connection_string": self.connection_string,
            "error_message": error_message,
        }
        await self.transport.send_json(self.diagnostics_result)

        if configure_ok:
            await self._send_phase_status("Diagnostics complete — robot is ready")

    # ------------------------------------------------------------------
    # Phase: Verification
    # ------------------------------------------------------------------

    async def _enter_verification(self) -> None:
        """Enter VERIFICATION phase — the broadcast loop auto-starts streaming."""
        self.phase = TrossenSetupPhase.VERIFICATION

    # ------------------------------------------------------------------
    # Broadcast loop — phase-driven streaming
    # ------------------------------------------------------------------

    async def _broadcast_loop(self) -> None:
        """Read the Trossen driver and push events to the client.

        Behaviour depends on the current phase:

          - ``VERIFICATION`` — stream positions + velocities as
            ``state_was_updated`` events (degrees for body joints,
            meters for gripper).
          - All other phases — idle (no driver reads).
        """
        read_interval = 1.0 / FPS

        try:
            while not self._stop_requested:
                start_time = time.perf_counter()

                try:
                    await self._broadcast_tick()
                except Exception as e:
                    logger.warning(f"Broadcast loop error: {e}")

                elapsed = time.perf_counter() - start_time
                await asyncio.sleep(max(0.001, read_interval - elapsed))
        except asyncio.CancelledError:
            pass

    async def _broadcast_tick(self) -> None:
        """Single iteration of the broadcast loop, dispatched by phase."""
        match self.phase:
            case TrossenSetupPhase.VERIFICATION:
                await self._broadcast_normalized_positions()
            case _:
                # No streaming needed — the sleep in _broadcast_loop
                # keeps us from busy-spinning.
                pass

    async def _broadcast_normalized_positions(self) -> None:
        """Read positions + velocities and send a ``state_was_updated`` event.

        Positions are in degrees (gripper in meters) — same format as the
        Trossen robot client's ``read_state()``.  Uses the same event format
        as ``RobotWorker._broadcast_loop`` so the frontend can reuse its
        joint-sync logic.
        """
        driver = self._require_driver()

        positions = await asyncio.to_thread(driver.get_all_positions)
        velocities = await asyncio.to_thread(driver.get_all_velocities)

        state: dict[str, float] = {}
        for index, name in enumerate(MOTOR_NAMES):
            if index < len(positions):
                # Convert radians to degrees for body joints; gripper stays in meters
                state[f"{name}.pos"] = (
                    float(np.rad2deg(positions[index])) if "gripper" not in name else float(positions[index])
                )
            if index < len(velocities):
                state[f"{name}.vel"] = float(velocities[index])

        await self.transport.send_json(
            {
                "event": "state_was_updated",
                "state": state,
            }
        )

    # ------------------------------------------------------------------
    # Command loop
    # ------------------------------------------------------------------

    async def _command_loop(self) -> None:
        """Wait for and handle commands from the frontend."""
        try:
            while not self._stop_requested:
                data = await self.transport.receive_command()
                if data is None:
                    continue

                command = data.get("command", "")
                logger.debug(f"Trossen setup worker received command: {command}")

                try:
                    await self._dispatch_command(command, data)
                except Exception as e:
                    logger.exception(f"Error handling command '{command}': {e}")
                    await self._send_event("error", message=str(e))
        except asyncio.CancelledError:
            pass

    async def _dispatch_command(self, command: str, data: dict[str, Any]) -> None:  # noqa: ARG002
        """Dispatch a single command received from the frontend."""
        match command:
            case "ping":
                await self.transport.send_json({"event": "pong"})

            case "re_probe":
                # Idle the broadcast loop while re-running diagnostics
                self.phase = TrossenSetupPhase.CONNECTING
                await self._run_diagnostics()

            case "enter_verification":
                await self._enter_verification()

            case _:
                await self._send_event("error", message=f"Unknown command: {command}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _send_phase_status(self, message: str) -> None:
        """Send a status event with current phase info."""
        await self.transport.send_json(
            {
                "event": "status",
                "state": self.state.value,
                "phase": self.phase.value,
                "message": message,
            }
        )

    async def _send_event(self, event: str, **kwargs: Any) -> None:
        """Send a named event with arbitrary payload."""
        await self.transport.send_json({"event": event, **kwargs})

    async def _cleanup(self) -> None:
        """Disconnect the Trossen driver."""
        if self.driver is not None:
            try:
                # Home to zeros before disconnecting
                self.driver.set_all_modes(trossen_arm.Mode.position)
                self.driver.set_all_positions(
                    np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                    2.0,
                    True,
                )
            except Exception:
                logger.debug("Failed to home Trossen robot during cleanup", exc_info=True)
            try:
                self.driver.cleanup()
            except Exception:
                logger.debug("Failed to cleanup Trossen driver", exc_info=True)
            self.driver = None
