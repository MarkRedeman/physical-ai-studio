"""CLI tool to verify the setup state of an SO101 robot arm.

This script connects to an SO101 Leader or Follower robot via USB serial
and checks:
  1. Whether the USB device can be found by serial number
  2. Whether all 6 expected motors respond with the correct model number
  3. The Angular_Resolution register value per motor (gear ratio diagnostic)
  4. Whether calibration values have been written to the motor EEPROM

Usage:
    uv run src/cli_robot_setup.py --robot-type SO101_Follower --serial-number ABC123
    uv run src/cli_robot_setup.py --robot-type SO101_Leader --serial-number ABC123

Exit codes:
    0 - Robot is fully ready (all motors present and calibrated)
    1 - Cannot connect (serial number not found or port error)
    2 - Motors not set up (some motor IDs missing or wrong model)
    3 - Motors present but not calibrated (or partially calibrated)
"""

import sys
from dataclasses import dataclass
from enum import StrEnum

import click
from lerobot.motors.feetech.feetech import FeetechMotorsBus
from lerobot.motors.motors_bus import Motor, MotorCalibration, MotorNormMode
from serial.tools import list_ports

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# USB Product IDs for the Feetech UART board (CH340 chipset)
FEETECH_PIDS = {29987, 21971}

# Expected model number for STS3215 motors (returned by ping)
STS3215_MODEL_NUMBER = 777

# Default calibration values indicating an uncalibrated motor.
# Factory defaults for STS3215: homing_offset=0, range_min=0, range_max=4095
DEFAULT_HOMING_OFFSET = 0
DEFAULT_RANGE_MIN = 0
DEFAULT_RANGE_MAX = 4095


class SO101Type(StrEnum):
    SO101_FOLLOWER = "SO101_Follower"
    SO101_LEADER = "SO101_Leader"


# Motor layout shared by both Leader and Follower.
# Both use 6x STS3215 motors with IDs 1-6.
MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def build_motors_dict() -> dict[str, Motor]:
    """Build the motor dict expected by FeetechMotorsBus for an SO101 arm."""
    norm_mode_body = MotorNormMode.RANGE_M100_100
    norm_mode_gripper = MotorNormMode.RANGE_0_100
    return {
        "shoulder_pan": Motor(1, "sts3215", norm_mode_body),
        "shoulder_lift": Motor(2, "sts3215", norm_mode_body),
        "elbow_flex": Motor(3, "sts3215", norm_mode_body),
        "wrist_flex": Motor(4, "sts3215", norm_mode_body),
        "wrist_roll": Motor(5, "sts3215", norm_mode_body),
        "gripper": Motor(6, "sts3215", norm_mode_gripper),
    }


# Expected gear ratios per joint, per robot type.
# These are physical properties - used for display/reminder only.
EXPECTED_GEAR_RATIOS: dict[SO101Type, dict[str, str]] = {
    SO101Type.SO101_FOLLOWER: {
        "shoulder_pan": "1/191",
        "shoulder_lift": "1/191",
        "elbow_flex": "1/147",
        "wrist_flex": "1/147",
        "wrist_roll": "1/147",
        "gripper": "1/345",
    },
    SO101Type.SO101_LEADER: {
        "shoulder_pan": "1/345",
        "shoulder_lift": "1/345",
        "elbow_flex": "1/345",
        "wrist_flex": "1/345",
        "wrist_roll": "1/345",
        "gripper": "1/345",
    },
}


# ---------------------------------------------------------------------------
# Result data structures
# ---------------------------------------------------------------------------


@dataclass
class MotorProbeResult:
    name: str
    motor_id: int
    found: bool
    model_number: int | None = None
    model_correct: bool = False
    angular_resolution: int | None = None
    expected_gear_ratio: str = ""


@dataclass
class MotorCalibrationResult:
    name: str
    motor_id: int
    homing_offset: int = 0
    range_min: int = 0
    range_max: int = 4095
    is_calibrated: bool = False


@dataclass
class SetupCheckResult:
    robot_type: SO101Type
    serial_number: str
    port: str | None = None
    connected: bool = False
    connection_error: str | None = None
    motor_probes: list[MotorProbeResult] | None = None
    calibration_results: list[MotorCalibrationResult] | None = None

    @property
    def all_motors_found(self) -> bool:
        if not self.motor_probes:
            return False
        return all(p.found and p.model_correct for p in self.motor_probes)

    @property
    def motors_found_count(self) -> int:
        if not self.motor_probes:
            return 0
        return sum(1 for p in self.motor_probes if p.found and p.model_correct)

    @property
    def all_calibrated(self) -> bool:
        if not self.calibration_results:
            return False
        return all(c.is_calibrated for c in self.calibration_results)

    @property
    def calibrated_count(self) -> int:
        if not self.calibration_results:
            return 0
        return sum(1 for c in self.calibration_results if c.is_calibrated)

    @property
    def exit_code(self) -> int:
        if not self.connected:
            return 1
        if not self.all_motors_found:
            return 2
        if not self.all_calibrated:
            return 3
        return 0

    @property
    def verdict(self) -> str:
        if not self.connected:
            return "CONNECTION_FAILED"
        if not self.all_motors_found:
            return "NEEDS_MOTOR_SETUP"
        if not self.all_calibrated:
            return "NEEDS_CALIBRATION"
        return "READY"


# ---------------------------------------------------------------------------
# Step 1: Resolve serial number to USB port
# ---------------------------------------------------------------------------


def find_port_by_serial(serial_number: str) -> str | None:
    """Find the USB device path for a given serial number.

    Returns the device path (e.g. /dev/ttyUSB0) or None if not found.
    """
    for port in list_ports.comports():
        if getattr(port, "serial_number", None) == serial_number:
            return port.device
    return None


def list_feetech_devices() -> list[tuple[str, str]]:
    """List all detected Feetech UART devices.

    Returns a list of (device_path, serial_number) tuples.
    """
    devices = []
    for port in list_ports.comports():
        if port.pid in FEETECH_PIDS:
            serial = getattr(port, "serial_number", None) or "unknown"
            devices.append((port.device, serial))
    return devices


# ---------------------------------------------------------------------------
# Step 2 & 3: Connect and probe motors
# ---------------------------------------------------------------------------


def probe_motors(
    bus: FeetechMotorsBus,
    robot_type: SO101Type,
) -> list[MotorProbeResult]:
    """Ping each expected motor ID and check its model number.

    The bus must already be connected (with handshake=False).
    """
    gear_ratios = EXPECTED_GEAR_RATIOS[robot_type]
    results = []

    for name, motor in bus.motors.items():
        model_nb = bus.ping(motor.id)
        found = model_nb is not None
        model_correct = model_nb == STS3215_MODEL_NUMBER if found else False

        result = MotorProbeResult(
            name=name,
            motor_id=motor.id,
            found=found,
            model_number=model_nb,
            model_correct=model_correct,
            expected_gear_ratio=gear_ratios.get(name, "unknown"),
        )

        # Read Angular_Resolution register if the motor is reachable
        if found:
            try:
                angular_res = bus.read("Angular_Resolution", name, normalize=False)
                result.angular_resolution = angular_res
            except Exception:
                # Non-critical: register may not be supported on all firmware
                result.angular_resolution = None

        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Step 4: Check calibration
# ---------------------------------------------------------------------------


def is_default_calibration(cal: MotorCalibration) -> bool:
    """Check if calibration values are factory defaults (uncalibrated).

    Factory defaults for STS3215: homing_offset=0, range_min=0, range_max=4095
    """
    return (
        cal.homing_offset == DEFAULT_HOMING_OFFSET
        and cal.range_min == DEFAULT_RANGE_MIN
        and cal.range_max == DEFAULT_RANGE_MAX
    )


def check_calibration(bus: FeetechMotorsBus) -> list[MotorCalibrationResult]:
    """Read calibration from motor EEPROM and check against defaults."""
    calibration = bus.read_calibration()
    results = []
    for name, cal in calibration.items():
        results.append(
            MotorCalibrationResult(
                name=name,
                motor_id=cal.id,
                homing_offset=cal.homing_offset,
                range_min=cal.range_min,
                range_max=cal.range_max,
                is_calibrated=not is_default_calibration(cal),
            )
        )
    return results


# ---------------------------------------------------------------------------
# Main check flow
# ---------------------------------------------------------------------------


def run_setup_check(robot_type: SO101Type, serial_number: str) -> SetupCheckResult:
    """Run the full setup check sequence. Returns a structured result."""
    result = SetupCheckResult(robot_type=robot_type, serial_number=serial_number)

    # Step 1: Find the USB port
    port = find_port_by_serial(serial_number)
    if port is None:
        result.connection_error = f"No USB device found with serial number '{serial_number}'"
        return result
    result.port = port

    # Step 2: Connect to the bus (no handshake - we do our own probing)
    motors = build_motors_dict()
    bus = FeetechMotorsBus(port=port, motors=motors)

    try:
        bus.connect(handshake=False)
    except (ConnectionError, OSError) as e:
        result.connection_error = str(e)
        return result

    result.connected = True

    try:
        # Step 3: Probe each motor
        result.motor_probes = probe_motors(bus, robot_type)

        # Step 4 & 5: Check calibration (only if all motors are present)
        if result.all_motors_found:
            result.calibration_results = check_calibration(bus)
    finally:
        try:
            bus.disconnect()
        except Exception:
            # Best effort cleanup
            try:
                bus.port_handler.closePort()
            except Exception:
                pass

    return result


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def print_result(result: SetupCheckResult) -> None:
    """Print a human-readable summary of the setup check."""
    click.echo()
    click.echo(f"=== {result.robot_type} Setup Check ===")
    click.echo(f"Serial Number: {result.serial_number}")
    click.echo(f"Port:          {result.port or 'not found'}")
    click.echo()

    # Connection status
    if not result.connected:
        click.secho(f"Connection ... FAILED", fg="red")
        click.echo(f"  Error: {result.connection_error}")
        _print_available_devices()
        return

    click.secho("Connection ... OK", fg="green")
    click.echo()

    # Motor probe results
    click.echo("Motors:")
    if result.motor_probes:
        for probe in result.motor_probes:
            name_padded = f"{probe.name:<15}"
            id_str = f"(ID {probe.motor_id})"

            if not probe.found:
                status = click.style("NOT FOUND", fg="red")
                click.echo(f"  {name_padded} {id_str} . {status}")
            elif not probe.model_correct:
                status = click.style(
                    f"WRONG MODEL (expected {STS3215_MODEL_NUMBER}, got {probe.model_number})",
                    fg="red",
                )
                click.echo(f"  {name_padded} {id_str} . {status}")
            else:
                status = click.style("FOUND", fg="green")
                angular_str = (
                    f"Angular_Resolution={probe.angular_resolution}"
                    if probe.angular_resolution is not None
                    else "Angular_Resolution=N/A"
                )
                gear_str = f"expected gearing: {probe.expected_gear_ratio}"
                click.echo(f"  {name_padded} {id_str} . {status}  [{angular_str}]  ({gear_str})")

    motors_ok = result.all_motors_found
    count = result.motors_found_count
    if motors_ok:
        click.secho(f"Motor Setup .. OK ({count}/6 motors found)", fg="green")
    else:
        click.secho(f"Motor Setup .. INCOMPLETE ({count}/6 motors found)", fg="red")

    click.echo()

    # Calibration results
    if not motors_ok:
        click.secho(
            "Calibration .. SKIPPED (cannot check calibration with missing motors)",
            fg="yellow",
        )
    elif result.calibration_results:
        click.echo("Calibration:")
        for cal in result.calibration_results:
            name_padded = f"{cal.name:<15}"
            id_str = f"(ID {cal.motor_id})"
            values_str = f"offset={cal.homing_offset}, min={cal.range_min}, max={cal.range_max}"

            if cal.is_calibrated:
                status = click.style("CALIBRATED", fg="green")
            else:
                status = click.style("NOT CALIBRATED", fg="red")

            click.echo(f"  {name_padded} {id_str} . {status}  ({values_str})")

        if result.all_calibrated:
            cal_count = result.calibrated_count
            click.secho(f"Calibration .. OK ({cal_count}/6 motors calibrated)", fg="green")
        else:
            cal_count = result.calibrated_count
            click.secho(
                f"Calibration .. INCOMPLETE ({cal_count}/6 motors calibrated)",
                fg="red",
            )

    click.echo()

    # Overall verdict
    verdict = result.verdict
    color = "green" if verdict == "READY" else "red"
    click.secho(f"Overall: {verdict}", fg=color, bold=True)
    click.echo()


def _print_available_devices() -> None:
    """Print all detected Feetech UART devices to help the user."""
    devices = list_feetech_devices()
    if devices:
        click.echo()
        click.echo("Available Feetech USB devices:")
        for device, serial in devices:
            click.echo(f"  {device}  (serial: {serial})")
    else:
        click.echo()
        click.secho("No Feetech USB devices detected.", fg="yellow")
        click.echo("Make sure the robot's controller board is connected via USB.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--robot-type",
    type=click.Choice([t.value for t in SO101Type], case_sensitive=False),
    required=True,
    help="Type of SO101 arm to check.",
)
@click.option(
    "--serial-number",
    required=True,
    help="USB serial number of the robot's controller board.",
)
def check(robot_type: str, serial_number: str) -> None:
    """Check the setup state of an SO101 robot arm.

    Connects to the robot via USB serial and verifies:
    - All 6 motors are present and have the correct model number
    - The Angular_Resolution register value per motor (gear ratio diagnostic)
    - Calibration values have been written to motor EEPROM
    """
    rt = SO101Type(robot_type)
    result = run_setup_check(rt, serial_number)
    print_result(result)
    sys.exit(result.exit_code)


if __name__ == "__main__":
    check()
