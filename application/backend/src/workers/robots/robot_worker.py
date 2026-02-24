import asyncio
import time
from pathlib import Path

from lerobot.robots.so101_follower import SO101FollowerConfig
from loguru import logger

from robots.widowxai.trossen_widowx_ai_follower import TrossenWidowXAIFollower
from robots.widowxai.trossen_widowx_ai_leader import TrossenWidowXAILeader
from exceptions import ResourceNotFoundError, ResourceType
from schemas.robot import NetworkIpRobotConfig, Robot, RobotType
from services.robot_calibration_service import RobotCalibrationService, find_robot_port
from utils.serial_robot_tools import RobotConnectionManager
from workers.robots.commands import handle_command, parse_command
from robots.feetech_robot_client import FeetechRobotClient
from robots.websocket_robot_client import WebsocketRobotClient
from robots.robot_client import RobotClient
from workers.transport.worker_transport import WorkerTransport
from workers.transport_worker import TransportWorker, WorkerState, WorkerStatus


class RobotWorker(TransportWorker):
    """Orchestrates robot communication over configurable transport."""

    def __init__(
        self,
        robot: Robot,
        transport: WorkerTransport,
        robot_manager: RobotConnectionManager,
        calibration_service: RobotCalibrationService,
        fps: int = 30,
        normalize: bool = False,
    ) -> None:
        super().__init__(transport)
        self.robot_manager = robot_manager
        self.calibration_service = calibration_service

        self.robot = robot
        self.client: RobotClient | None = None

        self.fps = fps
        self.normalize = normalize

    async def run(self) -> None:
        """Main worker loop."""
        try:
            await self.transport.connect()

            self.client = await self.create_robot_client()

            try:
                await self.client.connect()
                self.state = WorkerState.RUNNING

                logger.info(f"Created new robot client connection: {self.robot.id}")
                await self.transport.send_json(WorkerStatus(state=self.state, message="Robot connected").to_json())
            except Exception as e:
                logger.error(f"Failed to connect robot client: {e}")
                raise

            await self.run_concurrent(
                asyncio.create_task(self._broadcast_loop()),
                asyncio.create_task(self._command_loop()),
            )

        except Exception as e:
            self.state = WorkerState.ERROR
            self.error_message = str(e)
            logger.exception(f"Worker error: {e}")
            await self.transport.send_json(WorkerStatus(state=self.state, message=str(e)).to_json())
        finally:
            await self.shutdown()

    async def _broadcast_loop(self) -> None:
        """Listen to robot state updates and forward to client."""
        read_interval = 1 / self.fps
        try:
            previous_values = None

            while not self._stop_requested:
                if self.client is None:
                    await asyncio.sleep(0.1)
                    continue

                start_time = time.perf_counter()
                try:
                    state = await self.client.read_state(normalize=self.normalize)

                    # Only send if joint values changed (ignore timestamp)
                    current_values = state.get("state")
                    if current_values != previous_values:
                        previous_values = current_values
                        self.last_state = state
                        await self.transport.send_json(state)

                except Exception as e:
                    logger.error(f"Error reading robot state: {e}")
                    await asyncio.sleep(1)

                elapsed = time.perf_counter() - start_time
                sleep_time = max(0.001, read_interval - elapsed)
                await asyncio.sleep(sleep_time)

        except asyncio.CancelledError:
            pass

    async def _command_loop(self) -> None:
        """Handle incoming commands from client."""
        try:
            while not self._stop_requested:
                command = await self.transport.receive_command()

                if not self.client or command is None:
                    continue

                try:
                    robot_command = parse_command(command)
                    response = await handle_command(self.client, robot_command)

                    if response:
                        await self.transport.send_json(response)

                except Exception as e:
                    logger.warning("Received unknown command: {} from command {}", e, command)
                    await self.transport.send_json(RobotClient._create_event("error", message=str(e)))
        except asyncio.CancelledError:
            pass

    async def shutdown(self) -> None:
        """Graceful shutdown."""
        logger.info(f"Shutting down robot worker: {self.robot.id}")
        try:
            if self.client is not None:
                await self.client.disconnect()
        except Exception as e:
            logger.error(f"Failed to disconnect robot client: {e}")
        await super().shutdown()

    async def create_robot_client(self) -> RobotClient:
        factory = RobotClientFactory(
            self.robot_manager,
            self.calibration_service,
        )

        return await factory.build(self.robot)


class RobotClientFactory:
    calibration_service: RobotCalibrationService
    robot_manager: RobotConnectionManager

    def __init__(
        self,
        robot_manager: RobotConnectionManager,
        calibration_service: RobotCalibrationService,
    ) -> None:
        self.robot_manager = robot_manager
        self.calibration_service = calibration_service

    async def build(self, robot: Robot, normalize: bool = True) -> RobotClient:
        if robot.type == RobotType.TROSSEN_WIDOWXAI_LEADER:
            config = NetworkIpRobotConfig(
                type="leader", robot_type=RobotType.TROSSEN_WIDOWXAI_LEADER, connection_string=robot.connection_string
            )
            return TrossenWidowXAILeader(config=config)
        if robot.type == RobotType.TROSSEN_WIDOWXAI_FOLLOWER:
            config = NetworkIpRobotConfig(
                type="follower", robot_type=RobotType.TROSSEN_WIDOWXAI_FOLLOWER, connection_string=robot.connection_string
            )
            return TrossenWidowXAIFollower(config=config)


        if (str(robot.id) == "ab1e9694-a8af-4d51-abfb-b0743eebd6f6"):
            logger.info("Building a websocket client for lekiwi")
            # ws://localhost:8000/api/robot/ws?serial_id=5A7C121475&robot_type=lekiwi&calibration_id=kiwi
            #return WebsocketRobotClient("ws://localhost:8080/lekiwi/control?leader=so101", normalize)
            return WebsocketRobotClient("ws://localhost:8008/api/robot/ws?serial_id=5A7C121475&robot_type=lekiwi&calibration_id=kiwi&fps=60", normalize)


        if (str(robot.id) == "dd862523-3e04-428e-930d-9cc4a514b187"):
            return WebsocketRobotClient("ws://localhost:8080/lekiwi/control?leader=lekiwi", normalize)
            # return WebsocketRobotClient("ws://localhost:8080/lekiwi/control?leader=so101", normalize)

        if (str(robot.id) == "dd862523-3e04-428e-930d-9cc4a514b187"):
            return WebsocketRobotClient("ws://localhost:8080/so101/control", normalize)
        else:
            config = await self.get_robot_config(robot, )

            logger.info("Using config: {}", config)

            return FeetechRobotClient(config, normalize)

    async def get_robot_config(self, robot: Robot) -> SO101FollowerConfig:
        """
        Load robot configuration with calibration data.

        Args:
            robot: The robot to configure
            robot_manager: Service for discovering robot ports
            calibration_service: Service for loading calibration data

        Returns:
            SO101FollowerConfig configured with port and calibration

        Raises:
            ResourceNotFoundError: If robot port cannot be found
        """
        port = await find_robot_port(self.robot_manager, robot)
        if port is None:
            raise ResourceNotFoundError(ResourceType.ROBOT, robot.serial_id)

        if robot.active_calibration_id is None:
            return SO101FollowerConfig(port=port)

        calibration = await self.calibration_service.get_calibration(robot.active_calibration_id)

        return SO101FollowerConfig(port=port, id=str(calibration.id), calibration_dir=Path(calibration.file_path).parent)
