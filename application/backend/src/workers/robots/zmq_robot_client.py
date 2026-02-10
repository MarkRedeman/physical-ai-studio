import json

import zmq
import zmq.asyncio
from loguru import logger

from .robot_client import RobotClient


class ZMQRobotClient(RobotClient):
    """Robot client for robots connected via ZMQ."""

    def __init__(
        self,
        zmq_endpoint: str,
    ) -> None:
        self.zmq_endpoint = zmq_endpoint
        self.context = zmq.asyncio.Context()
        self.socket = None
        self.is_controlled = False

    async def connect(self) -> None:
        """Connect to remote robot via ZMQ."""
        logger.info(f"Connecting to ZMQ robot at {self.zmq_endpoint}")
        try:
            self.socket = self.context.socket(zmq.REQ)
            self.socket.connect(self.zmq_endpoint)
        except Exception as e:
            logger.error(f"Failed to connect to ZMQ robot: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from remote robot."""
        logger.info("Disconnecting from ZMQ robot")
        if self.socket:
            self.socket.close()
            self.context.term()

    async def ping(self) -> dict:
        """Send ping to remote robot."""
        await self._send_command("ping", {})
        return self._create_event("pong")

    async def set_joints_state(self, joints: dict) -> dict:
        """Set joint state on remote robot."""
        await self._send_command("set_joints_state", joints)
        return self._create_event("joints_state_was_set", joints=joints)

    async def enable_torque(self) -> dict:
        """Enable torque on remote robot."""
        await self._send_command("enable_torque", {})
        self.is_controlled = True
        return self._create_event("torque_was_enabled")

    async def disable_torque(self) -> dict:
        """Disable torque on remote robot."""
        await self._send_command("disable_torque", {})
        self.is_controlled = False
        return self._create_event("torque_was_disabled")

    async def read_state(self) -> dict:
        """Read current state from remote robot."""
        try:
            response = await self._send_command("read_state", {})
            return self._create_event(
                "state_was_updated",
                state=response.get("state"),
                is_controlled=response.get("is_controlled", False),
            )
        except Exception as e:
            logger.error(f"Failed to read remote state: {e}")
            raise

    async def _send_command(self, command: str, payload: dict) -> dict:
        """Send command to remote robot and receive response."""
        request = json.dumps({"command": command, "payload": payload})
        await self.socket.send_string(request)
        response_str = await self.socket.recv_string()
        return json.loads(response_str)
