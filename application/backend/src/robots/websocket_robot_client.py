"""WebSocket-based robot client implementation.

This module provides a RobotClient implementation that communicates with a remote
robot server over WebSocket. The server continuously pushes state updates, and
the client sends commands in a fire-and-forget manner.
"""

import asyncio
import json
from typing import Any, Optional

import websockets
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from schemas.robot import RobotType

from .robot_client import RobotClient

# Constants
CONNECTION_TIMEOUT = 10.0  # seconds
STATE_READ_TIMEOUT = 5.0  # seconds for waiting on state in read_state()
MAX_RECONNECT_ATTEMPTS = 3
DEFAULT_RECONNECT_DELAY = 1.0  # seconds
MAX_RECONNECT_DELAY = 10.0  # seconds


class WebSocketConnectionError(Exception):
    """Raised when WebSocket connection fails."""

    pass


class WebsocketRobotClient(RobotClient):
    """
    RobotClient implementation that communicates over WebSocket.

    This client connects to a remote WebSocket server that controls a robot.
    The server continuously pushes state updates, which are stored and returned
    by `read_state()`. Commands are sent in a fire-and-forget manner.

    The client uses a background listener task to continuously receive messages
    from the server, allowing state updates to arrive asynchronously while
    commands can be sent at any time.

    Args:
        websocket_url: WebSocket URL of the remote robot server
            (e.g., 'ws://localhost:8000/robot?normalize=true')
            The normalize preference can be passed as a query parameter.
        normalize: Default normalization setting for joint values.
            Currently used for documentation; actual normalization is
            controlled via the websocket_url query parameter.

    Example:
        ```python
        client = WebsocketRobotClient("ws://192.168.1.100:8765/robot?normalize=true")
        await client.connect()

        # State is continuously pushed by server
        state = await client.read_state()

        # Commands are fire-and-forget
        await client.set_joints_state({"shoulder": 0.5, "elbow": -0.3})

        await client.disconnect()
        ```
    """

    def __init__(self, websocket_url: str, normalize: bool = True) -> None:
        """
        Initialize the WebSocket robot client.

        Args:
            websocket_url: WebSocket URL of the remote robot server.
                Normalization preference should be passed as query parameter.
            normalize: Default normalization setting (for documentation).
        """
        self.websocket_url = websocket_url
        self.normalize = normalize

        # WebSocket connection (type is websockets.WebSocketClientProtocol at runtime)
        self._websocket: Any = None
        self._is_connected = False
        self._is_controlled = False

        # State management - single latest state, not a buffer
        self._latest_state: Optional[dict] = None
        self._state_event = asyncio.Event()

        # Background listener task
        self._listener_task: Optional[asyncio.Task] = None

        # Lock for sending commands
        self._send_lock = asyncio.Lock()

    @property
    def is_connected(self) -> bool:
        """Check if currently connected to the remote server."""
        return self._is_connected and self._websocket is not None

    @retry(
        stop=stop_after_attempt(MAX_RECONNECT_ATTEMPTS),
        wait=wait_exponential(multiplier=1, min=DEFAULT_RECONNECT_DELAY, max=MAX_RECONNECT_DELAY),
        retry=retry_if_exception_type((websockets.exceptions.WebSocketException, ConnectionError, OSError)),
        reraise=True,
    )
    async def connect(self) -> None:
        """
        Connect to the remote robot server and start the background listener.

        Uses tenacity for automatic retry with exponential backoff.

        Raises:
            WebSocketConnectionError: If connection fails after max retries.
        """
        logger.info("Connecting to WebSocket robot server: {}", self.websocket_url)

        try:
            self._websocket = await asyncio.wait_for(
                websockets.connect(self.websocket_url),
                timeout=CONNECTION_TIMEOUT,
            )
            self._is_connected = True

            # Start background listener task
            self._listener_task = asyncio.create_task(self._listener_loop())

            logger.info("Connected to WebSocket robot server: {}", self.websocket_url)

        except asyncio.TimeoutError as e:
            logger.error("Connection timeout to {}", self.websocket_url)
            raise WebSocketConnectionError(f"Connection timeout to {self.websocket_url}") from e
        except Exception as e:
            logger.error("Connection error: {}", e)
            raise

    async def disconnect(self) -> None:
        """
        Disconnect from the remote robot server.

        Cancels the background listener task and closes the WebSocket connection.
        """
        logger.info("Disconnecting from WebSocket robot server")

        self._is_connected = False

        # Cancel listener task
        if self._listener_task is not None:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
            self._listener_task = None

        # Close WebSocket
        if self._websocket is not None:
            try:
                await self._websocket.close()
            except Exception as e:
                logger.warning("Error closing WebSocket connection: {}", e)
            finally:
                self._websocket = None

        # Clear state
        self._latest_state = None
        self._state_event.clear()

        logger.debug("Disconnected from WebSocket robot server")

    async def _listener_loop(self) -> None:
        """
        Background task that continuously receives messages from the server.

        Handles incoming messages based on event type:
        - state_was_updated: Updates _latest_state and signals _state_event
        - Other events: Logged for debugging

        This task runs until cancelled or the connection is closed.
        """
        if self._websocket is None:
            return

        logger.debug("Starting WebSocket listener loop")

        try:
            async for message in self._websocket:
                try:
                    data = json.loads(message)

                    if not isinstance(data, dict):
                        logger.warning("Received non-dict message: {}", type(data).__name__)
                        continue

                    await self._handle_message(data)

                except json.JSONDecodeError as e:
                    logger.warning("Invalid JSON from server: {}", e)

        except websockets.exceptions.ConnectionClosed as e:
            logger.warning("WebSocket connection closed: {}", e)
            self._is_connected = False
        except asyncio.CancelledError:
            logger.debug("Listener loop cancelled")
            raise
        except Exception as e:
            logger.error("Error in listener loop: {}", e)
            self._is_connected = False

    async def _handle_message(self, data: dict) -> None:
        """
        Handle an incoming message from the server.

        Args:
            data: Parsed JSON message dictionary.
        """
        event_type = data.get("event")

        if event_type == "state_was_updated":
            # Update latest state and signal waiters
            self._latest_state = data

            # Update is_controlled from server if present
            if "is_controlled" in data:
                self._is_controlled = data["is_controlled"]

            self._state_event.set()

        elif event_type == "pong":
            logger.debug("Received pong from server")

        elif event_type == "torque_was_enabled":
            self._is_controlled = True
            logger.debug("Torque enabled confirmed by server")

        elif event_type == "torque_was_disabled":
            self._is_controlled = False
            logger.debug("Torque disabled confirmed by server")

        elif event_type == "joints_state_was_set":
            return

        elif event_type == "error":
            logger.warning("Error from server: {}", data.get("message", "unknown"))

        else:
            logger.debug("Received event: {}", event_type)

    async def ping(self) -> dict:
        """
        Send ping command to the server.

        This is fire-and-forget; returns a synthetic pong response immediately.
        The actual pong from the server is handled by the background listener.

        Returns:
            A synthetic 'pong' event dictionary with timestamp.
        """
        await self._send_command({"command": "ping"})
        return self._create_event("pong")

    async def set_joints_state(self, joints: dict) -> dict:
        """
        Set target positions for multiple joints.

        This is fire-and-forget; the command is sent and returns immediately.

        Args:
            joints: Dictionary mapping joint names to target positions.

        Returns:
            A synthetic confirmation event dictionary with timestamp.
        """
        await self._send_command({"command": "set_joints_state", "joints": joints})
        self._is_controlled = True
        return self._create_event("joints_state_was_set", joints=joints)

    async def enable_torque(self) -> dict:
        """
        Enable torque on all motors.

        This is fire-and-forget; the command is sent and returns immediately.

        Returns:
            A synthetic confirmation event dictionary with timestamp.
        """
        await self._send_command({"command": "enable_torque"})
        self._is_controlled = True
        return self._create_event("torque_was_enabled")

    async def disable_torque(self) -> dict:
        """
        Disable torque on all motors.

        This is fire-and-forget; the command is sent and returns immediately.

        Returns:
            A synthetic confirmation event dictionary with timestamp.
        """
        await self._send_command({"command": "disable_torque"})
        self._is_controlled = False
        return self._create_event("torque_was_disabled")

    async def read_state(self, *, normalize: bool = True) -> dict:
        """
        Read the current position of all joints.

        Returns the latest state pushed by the server. If no state has been
        received yet, waits up to STATE_READ_TIMEOUT seconds for one to arrive.

        Note: The normalize parameter is ignored; normalization is controlled
        via the websocket_url query parameter set during construction.

        Args:
            normalize: Ignored. Present for interface compatibility.

        Returns:
            Event dictionary containing the current joint states,
            or an empty dict if timeout occurs waiting for state.
        """
        # If we already have state, return it immediately
        if self._latest_state is not None:
            return self._latest_state

        # Wait for state to arrive
        try:
            await asyncio.wait_for(
                self._state_event.wait(),
                timeout=STATE_READ_TIMEOUT,
            )
            # State event was set, return the state
            if self._latest_state is not None:
                return self._latest_state
        except asyncio.TimeoutError:
            logger.warning("Timeout waiting for state after {}s", STATE_READ_TIMEOUT)

        # Return empty dict on timeout or if state is still None
        return {}

    async def _send_command(self, command: dict) -> None:
        """
        Send a command to the remote server (fire-and-forget).

        Args:
            command: Command dictionary to send.

        Raises:
            WebSocketConnectionError: If not connected.
        """
        if not self.is_connected or self._websocket is None:
            raise WebSocketConnectionError("Not connected to robot server")

        async with self._send_lock:
            try:
                command_json = json.dumps(command)
                await self._websocket.send(command_json)

            except websockets.exceptions.ConnectionClosed as e:
                self._is_connected = False
                logger.error("Connection closed while sending command: {}", e)
                raise WebSocketConnectionError(f"Connection closed: {e}") from e

    def features(self) -> list[str]:
        """Get Robot features. Returns list with joints."""
        return []

    async def read_forces(self) -> dict | None:
        """Read current robot forces. Returns state dict with timestamp."""
        return None

    async def set_forces(self, forces: dict) -> dict:
        """Set current robot forces. Returns event dict with timestamp."""
        raise Exception("Not implemented for Feetech Robot Client")

    @property
    def robot_type(self) -> RobotType:
        """Specify the RobotType"""
        return RobotType("SO101_Follower")
