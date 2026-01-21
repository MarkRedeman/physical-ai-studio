import asyncio
import json

import websockets
from loguru import logger

from .robot_client import RobotClient


class WebSocketRobotClient(RobotClient):
    """Robot client that proxies commands to another instance via WebSocket."""

    def __init__(
        self,
        websocket_url: str,
        connect_timeout: float = 10.0,
    ) -> None:
        """
        Initialize WebSocket robot client.
        
        Args:
            websocket_url: URL of the remote robot endpoint (e.g., ws://remote-server/api/robots/ws?robot=...)
            connect_timeout: Timeout for establishing connection in seconds
        """
        self.websocket_url = websocket_url
        self.connect_timeout = connect_timeout
        self.websocket: websockets.WebSocketClientProtocol | None = None
        self.is_controlled = False
        self._read_task: asyncio.Task | None = None
        self.running = False
        self.last_state: dict | None = None
        self._state_lock = asyncio.Lock()
        self._pending_responses: dict[str, asyncio.Event] = {}
        self._response_data: dict[str, dict] = {}

    async def connect(self) -> None:
        """Connect to the remote robot via WebSocket."""
        logger.info(f"Connecting to remote robot at {self.websocket_url}")
        try:
            self.websocket = await asyncio.wait_for(
                websockets.connect(self.websocket_url),
                timeout=self.connect_timeout,
            )
            self.running = True
            
            # Start background task to listen for messages
            self._read_task = asyncio.create_task(self._listen_loop())
            
            logger.info(f"Connected to remote robot at {self.websocket_url}")
        
        except TimeoutError:
            raise RuntimeError(
                f"Failed to connect to remote robot within {self.connect_timeout}s"
            )
        except Exception as e:
            logger.error(f"Failed to connect to remote robot: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from the remote robot."""
        logger.info("Disconnecting from remote robot")
        self.running = False
        
        if self._read_task:
            self._read_task.cancel()
            try:
                await self._read_task
            except asyncio.CancelledError:
                pass
            self._read_task = None
        
        if self.websocket:
            try:
                await self.websocket.close()
            except Exception as e:
                logger.warning(f"Error closing websocket: {e}")
            finally:
                self.websocket = None

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
        async with self._state_lock:
            if self.last_state is None:
                return {}
            
            return self._create_event(
                "joints_state_was_updated",
                state=self.last_state.get("state"),
                is_controlled=self.is_controlled,
            )

    async def _send_command(self, event: str, payload: dict) -> dict:
        """
        Send a command to the remote robot and wait for acknowledgment.
        
        This sends the command and waits for a response, but the continuous
        state updates are handled by _listen_loop().
        """
        if not self.websocket:
            raise RuntimeError("Not connected to remote robot")

        command = {
            "event": event,
            "payload": payload,
        }

        # Create event for this command
        response_key = f"{event}_{id(command)}"
        response_event = asyncio.Event()
        self._pending_responses[response_key] = response_event

        try:
            # Send command
            await self.websocket.send(json.dumps(command))
            logger.debug(f"Sent command to remote robot: {event}")

            # Wait for response with timeout
            await asyncio.wait_for(response_event.wait(), timeout=5.0)

            # Get the response data
            response = self._response_data.pop(response_key)
            logger.debug(f"Received response from remote robot: {response}")

            return response

        except TimeoutError:
            raise RuntimeError(f"Timeout waiting for response to '{event}' command")
        except Exception as e:
            logger.error(f"Error sending command to remote robot: {e}")
            raise
        finally:
            self._pending_responses.pop(response_key, None)

    async def _listen_loop(self) -> None:
        """
        Continuously listen to messages from the remote robot.
        
        Updates state for state_was_updated events.
        Sends acknowledgments for command responses.
        """
        try:
            while self.running:
                try:
                    message_str = await self.websocket.recv()
                    message = json.loads(message_str)
                    logger.debug(f"Received message from remote: {message.get('event')}")

                    event_type = message.get("event")

                    # Handle state updates
                    if event_type == "joints_state_was_updated":
                        async with self._state_lock:
                            self.last_state = message
                            if message.get("is_controlled"):
                                self.is_controlled = True

                    # Handle command responses
                    elif event_type in ["pong", "joints_state_was_set", "torque_was_enabled", "torque_was_disabled"]:
                        # Find matching pending response and unblock it
                        for response_key in list(self._pending_responses.keys()):
                            event = response_key.split("_")[0]
                            if (event_type == "pong" and event == "ping") or event_type.startswith(event):
                                self._response_data[response_key] = message
                                self._pending_responses[response_key].set()
                                break

                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON from remote robot: {e}")
                except Exception as e:
                    logger.error(f"Error in listen loop: {e}")
                    if self.running:
                        await asyncio.sleep(1)

        except asyncio.CancelledError:
            logger.debug("Listen loop cancelled")
        finally:
            logger.debug("Listen loop stopped")
