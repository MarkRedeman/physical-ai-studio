"""CLI tool to reset an SO101 robot arm to a near-factory state.

This script connects to an SO101 Leader or Follower robot via USB serial
and resets:
  1. Calibration registers: Homing_Offset → 0, Min_Position_Limit → 0,
     Max_Position_Limit → 4095
  2. PID / configuration registers back to factory defaults
  3. Motor IDs: all motors back to ID 1 (one at a time, with pauses)

After a full reset the robot will appear as if it just came out of the box:
  - All 6 motors respond on ID 1 (so the setup wizard's probe will find 0/6)
  - Calibration is at factory defaults (homing_offset=0, range 0–4095)

Modes:
  --mode calibration   Reset calibration + config only (keep motor IDs)
  --mode full          Reset calibration + config AND motor IDs → 1

Usage:
    uv run src/cli_robot_reset.py --robot-type SO101_Follower --serial-number ABC123 --mode full
    uv run src/cli_robot_reset.py --robot-type SO101_Leader --serial-number ABC123 --mode calibration


    uv run src/cli_robot_reset.py --robot-type SO101_Leader --serial-number 5AAF270363 --mode full


WARNING: The "full" mode sets ALL motor IDs to 1. After this, you cannot
individually address motors until you run the motor setup flow again (which
assigns IDs one motor at a time while only one is physically connected).
"""

import sys
from enum import StrEnum

import click
from lerobot.motors.feetech.feetech import FeetechMotorsBus
from lerobot.motors.motors_bus import Motor, MotorNormMode
from serial.tools import list_ports

# ---------------------------------------------------------------------------
# Constants — must stay in sync with cli_robot_setup.py
# ---------------------------------------------------------------------------

FEETECH_PIDS = {29987, 21971}
STS3215_MODEL_NUMBER = 777

MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


class SO101Type(StrEnum):
    SO101_FOLLOWER = "SO101_Follower"
    SO101_LEADER = "SO101_Leader"


class ResetMode(StrEnum):
    CALIBRATION = "calibration"
    FULL = "full"


def build_motors_dict() -> dict[str, Motor]:
    """Build the motor dict expected by FeetechMotorsBus for an SO101 arm."""
    norm_body = MotorNormMode.RANGE_M100_100
    norm_gripper = MotorNormMode.RANGE_0_100
    return {
        "shoulder_pan": Motor(1, "sts3215", norm_body),
        "shoulder_lift": Motor(2, "sts3215", norm_body),
        "elbow_flex": Motor(3, "sts3215", norm_body),
        "wrist_flex": Motor(4, "sts3215", norm_body),
        "wrist_roll": Motor(5, "sts3215", norm_body),
        "gripper": Motor(6, "sts3215", norm_gripper),
    }


# Factory defaults for STS3215 registers.
# These are the values we reset TO.
FACTORY_DEFAULTS: dict[str, int] = {
    # Calibration
    "Homing_Offset": 0,
    "Min_Position_Limit": 0,
    "Max_Position_Limit": 4095,
    # PID
    "P_Coefficient": 32,
    "I_Coefficient": 0,
    "D_Coefficient": 32,
    # Acceleration
    "Acceleration": 0,
    "Maximum_Acceleration": 0,
    # Communication
    "Return_Delay_Time": 250,
    # Operating mode
    "Operating_Mode": 0,
    # Torque / protection (gripper-specific values are overridden below)
    "Max_Torque_Limit": 1000,
    "Protection_Current": 500,
    "Overload_Torque": 50,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def find_port_by_serial(serial_number: str) -> str | None:
    for port in list_ports.comports():
        if getattr(port, "serial_number", None) == serial_number:
            return port.device
    return None


def list_feetech_devices() -> list[tuple[str, str]]:
    devices = []
    for port in list_ports.comports():
        if port.pid in FEETECH_PIDS:
            serial = getattr(port, "serial_number", None) or "unknown"
            devices.append((port.device, serial))
    return devices


def _unlock_motor(bus: FeetechMotorsBus, motor_name: str) -> None:
    """Disable torque and unlock EEPROM for a single motor."""
    bus.write("Torque_Enable", motor_name, 0)
    bus.write("Lock", motor_name, 0)


# ---------------------------------------------------------------------------
# Reset: calibration + configuration registers
# ---------------------------------------------------------------------------


def reset_calibration_and_config(bus: FeetechMotorsBus) -> list[str]:
    """Reset calibration and configuration registers to factory defaults.

    Returns a list of log messages describing what was done.
    """
    log: list[str] = []

    for motor_name in bus.motors:
        # Verify the motor is reachable
        model = bus.ping(bus.motors[motor_name].id)
        if model is None:
            log.append(f"  {motor_name} (ID {bus.motors[motor_name].id}): NOT FOUND — skipping")
            continue

        _unlock_motor(bus, motor_name)

        for reg_name, default_value in FACTORY_DEFAULTS.items():
            try:
                bus.write(reg_name, motor_name, default_value)
            except Exception as e:
                log.append(f"  {motor_name}: failed to write {reg_name}={default_value}: {e}")
                continue

        log.append(f"  {motor_name} (ID {bus.motors[motor_name].id}): registers reset to factory defaults")

    return log


# ---------------------------------------------------------------------------
# Reset: motor IDs → 1
# ---------------------------------------------------------------------------


def reset_motor_ids(bus: FeetechMotorsBus) -> list[str]:
    """Reset all motor IDs back to 1 (factory default).

    This writes ID=1 to each motor that responds at its current (assigned) ID.
    After this, ALL motors will share ID 1 and cannot be individually addressed.

    Returns a list of log messages.
    """
    log: list[str] = []

    for motor_name, motor in bus.motors.items():
        current_id = motor.id

        if current_id == 1:
            log.append(f"  {motor_name} (ID {current_id}): already at ID 1 — skipping")
            continue

        # Check if the motor is reachable at its current ID
        model = bus.ping(current_id)
        if model is None:
            log.append(f"  {motor_name} (ID {current_id}): NOT FOUND — skipping")
            continue

        # Unlock and write new ID
        _unlock_motor(bus, motor_name)

        try:
            bus.write("ID", motor_name, 1)
            log.append(f"  {motor_name}: ID {current_id} → 1")
        except Exception as e:
            log.append(f"  {motor_name}: failed to change ID {current_id} → 1: {e}")

    return log


# ---------------------------------------------------------------------------
# Main reset flow
# ---------------------------------------------------------------------------


def run_reset(
    robot_type: SO101Type,
    serial_number: str,
    mode: ResetMode,
) -> int:
    """Run the reset sequence. Returns exit code (0=success, 1=failure)."""

    # Find USB port
    port = find_port_by_serial(serial_number)
    if port is None:
        click.secho(f"No USB device found with serial number '{serial_number}'", fg="red")
        devices = list_feetech_devices()
        if devices:
            click.echo("\nAvailable Feetech USB devices:")
            for dev, sn in devices:
                click.echo(f"  {dev}  (serial: {sn})")
        return 1

    click.echo(f"Found {robot_type} at {port} (serial: {serial_number})")

    # Connect
    motors = build_motors_dict()
    bus = FeetechMotorsBus(port=port, motors=motors)

    try:
        bus.connect(handshake=False)
    except (ConnectionError, OSError) as e:
        click.secho(f"Connection failed: {e}", fg="red")
        return 1

    click.secho("Connected.", fg="green")

    try:
        # Step 1: Reset calibration + config registers
        click.echo()
        click.echo("Resetting calibration and configuration registers...")
        log = reset_calibration_and_config(bus)
        for line in log:
            click.echo(line)

        # Step 2: Reset motor IDs (only in full mode)
        if mode == ResetMode.FULL:
            click.echo()
            click.echo("Resetting motor IDs to 1...")
            log = reset_motor_ids(bus)
            for line in log:
                click.echo(line)

        click.echo()
        click.secho("Reset complete.", fg="green", bold=True)

        if mode == ResetMode.FULL:
            click.echo()
            click.secho(
                "All motors now have ID 1. To use the robot again, run the\n"
                "setup wizard and go through the Motor Setup step (connect\n"
                "motors one at a time to assign unique IDs).",
                fg="yellow",
            )
        elif mode == ResetMode.CALIBRATION:
            click.echo()
            click.echo(
                "Calibration and config registers have been reset. Motor IDs\n"
                "are unchanged. Run the setup wizard's Calibration step to\n"
                "recalibrate."
            )

    finally:
        try:
            bus.disconnect()
        except Exception:
            try:
                bus.port_handler.closePort()
            except Exception:
                pass

    return 0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--robot-type",
    type=click.Choice([t.value for t in SO101Type], case_sensitive=False),
    required=True,
    help="Type of SO101 arm to reset.",
)
@click.option(
    "--serial-number",
    required=True,
    help="USB serial number of the robot's controller board.",
)
@click.option(
    "--mode",
    type=click.Choice([m.value for m in ResetMode], case_sensitive=False),
    required=True,
    help=(
        "'calibration' resets calibration + config registers only (keeps motor IDs). "
        "'full' also resets all motor IDs back to 1 (factory default)."
    ),
)
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    default=False,
    help="Skip confirmation prompt.",
)
def reset(robot_type: str, serial_number: str, mode: str, yes: bool) -> None:
    """Reset an SO101 robot arm to near-factory state.

    Use --mode calibration to reset only calibration and configuration
    registers (homing offsets, PID, range limits, etc.) while keeping
    the motor IDs intact.

    Use --mode full to additionally reset ALL motor IDs back to 1.
    This simulates a brand-new robot and requires the motor setup
    flow to reassign IDs one at a time.
    """
    rt = SO101Type(robot_type)
    rm = ResetMode(mode)

    if not yes:
        click.echo()
        click.echo(f"  Robot type:    {rt}")
        click.echo(f"  Serial number: {serial_number}")
        click.echo(f"  Reset mode:    {rm}")
        click.echo()

        if rm == ResetMode.FULL:
            click.secho(
                "WARNING: Full reset will set ALL motor IDs to 1.\n"
                "You will need to reconnect motors one at a time to\n"
                "reassign IDs via the setup wizard.",
                fg="yellow",
                bold=True,
            )
            click.echo()

        if not click.confirm("Proceed with reset?"):
            click.echo("Aborted.")
            sys.exit(0)

    exit_code = run_reset(rt, serial_number, rm)
    sys.exit(exit_code)


if __name__ == "__main__":
    reset()
