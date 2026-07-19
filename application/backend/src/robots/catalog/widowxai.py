from physicalai.robot.trossen import BimanualWidowXAI, WidowXAI

from schemas.robot import RobotType, TrossenBimanualPayload, TrossenSingleArmPayload

from .types import CatalogRobot, CatalogRobotFactory, RobotAdapterOptions, RobotCatalogDefinition

_TROSSEN_TO_URDF = {
    "shoulder_pan.pos": ["joint_0"],
    "shoulder_lift.pos": ["joint_1"],
    "elbow_flex.pos": ["joint_2"],
    "wrist_flex.pos": ["joint_3"],
    "wrist_yaw.pos": ["joint_4"],
    "wrist_roll.pos": ["joint_5"],
    "gripper.pos": ["left_carriage_joint", "right_carriage_joint"],
}

_BIMANUAL_TROSSEN_TO_URDF = {
    "left_shoulder_pan.pos": ["follower_left_joint_0"],
    "left_shoulder_lift.pos": ["follower_left_joint_1"],
    "left_elbow_flex.pos": ["follower_left_joint_2"],
    "left_wrist_flex.pos": ["follower_left_joint_3"],
    "left_wrist_yaw.pos": ["follower_left_joint_4"],
    "left_wrist_roll.pos": ["follower_left_joint_5"],
    "left_gripper.pos": ["follower_left_left_carriage_joint", "follower_left_right_carriage_joint"],
    "right_shoulder_pan.pos": ["follower_right_joint_0"],
    "right_shoulder_lift.pos": ["follower_right_joint_1"],
    "right_elbow_flex.pos": ["follower_right_joint_2"],
    "right_wrist_flex.pos": ["follower_right_joint_3"],
    "right_wrist_yaw.pos": ["follower_right_joint_4"],
    "right_wrist_roll.pos": ["follower_right_joint_5"],
    "right_gripper.pos": ["follower_right_left_carriage_joint", "follower_right_right_carriage_joint"],
}


async def _build_trossen_single_arm_driver(
    robot: CatalogRobot[TrossenSingleArmPayload], _factory: CatalogRobotFactory
) -> WidowXAI:
    role = "follower" if robot.type == RobotType.TROSSEN_WIDOWXAI_FOLLOWER else "leader"
    return WidowXAI(ip=robot.payload.connection_string, role=role)


async def _build_trossen_bimanual_driver(
    robot: CatalogRobot[TrossenBimanualPayload], _factory: CatalogRobotFactory
) -> BimanualWidowXAI:
    mode = "follower" if robot.type == RobotType.TROSSEN_BIMANUAL_WIDOWXAI_FOLLOWER else "leader"
    left_driver = WidowXAI(ip=robot.payload.connection_string_left, role=mode)
    right_driver = WidowXAI(ip=robot.payload.connection_string_right, role=mode)
    return BimanualWidowXAI(left=left_driver, right=right_driver)


async def _ping(ip: str, ping_timeout: float = 1.0) -> bool:
    import asyncio
    import sys

    param = "-n" if sys.platform.lower().startswith("win") else "-c"
    command = ["ping", param, "1", "-W", str(int(ping_timeout * 1000)), ip]
    proc = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        return (await asyncio.wait_for(proc.wait(), timeout=ping_timeout + 0.5)) == 0
    except TimeoutError:
        proc.kill()
        await proc.wait()
        return False


async def identify_trossen_robot_visually(robot: TrossenSingleArmRobot) -> None:
    """Identify the robot by moving the gripper from current to open to closed to initial."""
    import trossen_arm
    from loguru import logger

    driver = trossen_arm.TrossenArmDriver()

    logger.info("Configuring the drivers...")
    driver.configure(
        trossen_arm.Model.wxai_v0,
        trossen_arm.StandardEndEffector.wxai_v0_leader,
        robot.payload.connection_string,
        True,
        timeout=5,
    )

    driver.set_gripper_mode(trossen_arm.Mode.position)
    driver.set_gripper_position(0.02, 0.5, True)
    driver.set_gripper_mode(trossen_arm.Mode.position)
    driver.set_gripper_position(0.0, 0.5, True)


class TrossenSingleArmProbe:
    """Probe for Trossen single-arm robots — IP-based gripper identification."""

    async def discover(self, manager: PortScanner) -> list[SerialPortInfo]:  # noqa: ARG002
        return []

    async def identify(
        self,
        payload: dict[str, Any],
        manager: PortScanner | None = None,  # noqa: ARG002
        joint: str | None = None,  # noqa: ARG002
    ) -> None:
        robot_payload = TrossenSingleArmPayload(**payload)
        now = datetime.now()
        robot = TrossenSingleArmRobot(
            id=UUID(int=0),
            name="",
            type="Trossen_WidowXAI_Follower",
            payload=robot_payload,
            active_calibration_id=None,
            created_at=now,
            updated_at=now,
        )
        await identify_trossen_robot_visually(robot)

    async def is_online(self, payload: dict[str, Any], manager: PortScanner | None = None) -> bool:  # noqa: ARG002
        robot_payload = TrossenSingleArmPayload(**payload)
        if not robot_payload.connection_string:
            return False
        return await _ping(robot_payload.connection_string)


class TrossenBimanualProbe:
    """Probe for Trossen bimanual robots — per-arm IP-based identification."""

    async def discover(self, manager: PortScanner) -> list[SerialPortInfo]:  # noqa: ARG002
        return []

    async def identify(
        self,
        payload: dict[str, Any],
        manager: PortScanner | None = None,  # noqa: ARG002
        joint: str | None = None,  # noqa: ARG002
    ) -> None:
        await TrossenSingleArmProbe().identify(payload)

    async def is_online(self, payload: dict[str, Any], manager: PortScanner | None = None) -> bool:  # noqa: ARG002
        import asyncio

        robot_payload = TrossenBimanualPayload(**payload)
        left = robot_payload.connection_string_left
        right = robot_payload.connection_string_right
        if not left or not right:
            return False
        left_ok, right_ok = await asyncio.gather(_ping(left), _ping(right))
        return left_ok and right_ok


_SINGLE_ARM_PROBE = TrossenSingleArmProbe()
_BIMANUAL_PROBE = TrossenBimanualProbe()


def get_definitions() -> list[RobotCatalogDefinition]:
    """Return built-in WidowX AI robot catalog definitions."""
    return [
        RobotCatalogDefinition(
            type=RobotType.TROSSEN_WIDOWXAI_FOLLOWER,
            display_name="Trossen WidowX AI Follower",
            role="follower",
            urdf_path=f"/api/robots/catalog/{RobotType.TROSSEN_WIDOWXAI_FOLLOWER}/urdf",
            package_map={
                "trossen_arm_description": f"/api/robots/catalog/{RobotType.TROSSEN_WIDOWXAI_FOLLOWER}",
            },
            joint_map=_TROSSEN_TO_URDF,
            urdf_relative_path="widowx/urdf/generated/wxai/wxai_follower.urdf",
            robot_builder=_build_trossen_single_arm_driver,
            adapter_options=RobotAdapterOptions(include_velocities=True, goal_time_scale=1.0, external_effort_gain=0.1),
            probe=_SINGLE_ARM_PROBE,
        ),
        RobotCatalogDefinition(
            type=RobotType.TROSSEN_WIDOWXAI_LEADER,
            display_name="Trossen WidowX AI Leader",
            role="leader",
            urdf_path=f"/api/robots/catalog/{RobotType.TROSSEN_WIDOWXAI_LEADER}/urdf",
            package_map={
                "trossen_arm_description": f"/api/robots/catalog/{RobotType.TROSSEN_WIDOWXAI_LEADER}",
            },
            joint_map=_TROSSEN_TO_URDF,
            urdf_relative_path="widowx/urdf/generated/wxai/wxai_follower.urdf",
            robot_builder=_build_trossen_single_arm_driver,
            adapter_options=RobotAdapterOptions(include_velocities=True, goal_time_scale=1.0, external_effort_gain=0.1),
            probe=_SINGLE_ARM_PROBE,
        ),
        RobotCatalogDefinition(
            type=RobotType.TROSSEN_BIMANUAL_WIDOWXAI_FOLLOWER,
            display_name="Trossen Bimanual WidowX AI Follower",
            role="follower",
            urdf_path=f"/api/robots/catalog/{RobotType.TROSSEN_BIMANUAL_WIDOWXAI_FOLLOWER}/urdf",
            package_map={
                "trossen_arm_description": f"/api/robots/catalog/{RobotType.TROSSEN_BIMANUAL_WIDOWXAI_FOLLOWER}",
            },
            joint_map=_BIMANUAL_TROSSEN_TO_URDF,
            urdf_relative_path="widowx/urdf/generated/stationary_ai.urdf",
            robot_builder=_build_trossen_bimanual_driver,
            adapter_options=RobotAdapterOptions(include_velocities=True, goal_time_scale=1.0, external_effort_gain=0.1),
            probe=_BIMANUAL_PROBE,
        ),
        RobotCatalogDefinition(
            type=RobotType.TROSSEN_BIMANUAL_WIDOWXAI_LEADER,
            display_name="Trossen Bimanual WidowX AI Leader",
            role="leader",
            urdf_path=f"/api/robots/catalog/{RobotType.TROSSEN_BIMANUAL_WIDOWXAI_LEADER}/urdf",
            package_map={
                "trossen_arm_description": f"/api/robots/catalog/{RobotType.TROSSEN_BIMANUAL_WIDOWXAI_LEADER}",
            },
            joint_map=_BIMANUAL_TROSSEN_TO_URDF,
            urdf_relative_path="widowx/urdf/generated/stationary_ai.urdf",
            robot_builder=_build_trossen_bimanual_driver,
            adapter_options=RobotAdapterOptions(include_velocities=True, goal_time_scale=1.0, external_effort_gain=0.1),
            probe=_BIMANUAL_PROBE,
        ),
    ]
