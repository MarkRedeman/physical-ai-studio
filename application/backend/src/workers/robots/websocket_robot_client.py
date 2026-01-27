"""WebSocket-based robot client implementation.

This module provides a RobotClient implementation that communicates with a remote
robot server over WebSocket, forwarding commands and receiving responses.
"""

import asyncio
import json
from collections import deque
from typing import Any, Optional

import websockets
from loguru import logger
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .robot_client import RobotClient

# Constants
RECEIVE_TIMEOUT = 5.0  # seconds
CONNECTION_TIMEOUT = 10.0  # seconds
MAX_RECONNECT_ATTEMPTS = 3
DEFAULT_RECONNECT_DELAY = 1.0  # seconds
MAX_RECONNECT_DELAY = 10.0  # seconds


class WebSocketConnectionError(Exception):
    """Raised when WebSocket connection fails."""

    pass


class WebSocketResponseError(Exception):
    """Raised when response is invalid or times out."""

    pass


class WebsocketRobotClient(RobotClient):
    """
    RobotClient implementation that communicates over WebSocket.

    Connects to a remote WebSocket server that controls a robot,
    forwarding commands and receiving responses/state updates.

    This client acts as a proxy, sending commands to a remote robot server
    and receiving responses. It does not implement the Calibratable protocol
    as calibration is typically handled by the remote server directly.

    Args:
        websocket_url: WebSocket URL of the remote robot server
            (e.g., 'ws://localhost:8000/robot' or 'wss://...')
        normalize: Default normalization setting for joint values.
            This is sent with read_state commands.

    Example:
        ```python
        client = WebsocketRobotClient("ws://192.168.1.100:8765/robot")
        await client.connect()

        state = await client.read_state()
        await client.set_joints_state({"shoulder": 0.5, "elbow": -0.3})

        await client.disconnect()
        ```
    """

    def __init__(self, websocket_url: str, normalize: bool = True) -> None:
        """
        Initialize the WebSocket robot client.

        Args:
            websocket_url: WebSocket URL of the remote robot server.
            normalize: Default normalization setting for joint values.
        """
        self.websocket_url = websocket_url
        self.normalize = normalize

        self._websocket: Optional[websockets.WebSocketClientProtocol] = None
        self._is_connected = False
        self._is_controlled = False
        self._lock = asyncio.Lock()

        # Buffer for unsolicited state updates
        self._state_buffer: deque[dict] = deque(maxlen=10)

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
        Connect to the remote robot server.

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
            logger.info("Connected to WebSocket robot server: {}", self.websocket_url)

        except asyncio.TimeoutError as e:
            logger.error("Connection timeout to {}", self.websocket_url)
            raise WebSocketConnectionError(f"Connection timeout to {self.websocket_url}") from e
        except RetryError as e:
            logger.error(
                "Failed to connect after {} attempts: {}",
                MAX_RECONNECT_ATTEMPTS,
                e.last_attempt.exception(),
            )
            raise WebSocketConnectionError(
                f"Failed to connect to {self.websocket_url} after {MAX_RECONNECT_ATTEMPTS} attempts"
            ) from e
        except Exception as e:
            logger.error("Connection error: {}", e)
            raise

    async def disconnect(self) -> None:
        """
        Disconnect from the remote robot server.

        Safely closes the WebSocket connection if open.
        """
        logger.info("Disconnecting from WebSocket robot server")

        self._is_connected = False

        if self._websocket is not None:
            try:
                await self._websocket.close()
            except Exception as e:
                logger.warning("Error closing WebSocket connection: {}", e)
            finally:
                self._websocket = None

        self._state_buffer.clear()
        logger.debug("Disconnected from WebSocket robot server")

    async def ping(self) -> dict:
        """
        Send ping command to verify communication.

        Returns:
            A 'pong' event dictionary from the server.

        Raises:
            WebSocketResponseError: If no response or invalid response.
        """
        return await self._send_command({"command": "ping"})

    async def set_joints_state(self, joints: dict) -> dict:
        """
        Set target positions for multiple joints.

        Args:
            joints: Dictionary mapping joint names to target positions.

        Returns:
            Confirmation event dictionary from the server.

        Raises:
            WebSocketResponseError: If no response or invalid response.
        """
        response = await self._send_command({"command": "set_joints_state", "joints": joints})

        # Track controlled state based on successful joint commands
        self._is_controlled = True

        return response

    async def enable_torque(self) -> dict:
        """
        Enable torque on all motors.

        Returns:
            Confirmation event dictionary from the server.

        Raises:
            WebSocketResponseError: If no response or invalid response.
        """
        response = await self._send_command({"command": "enable_torque"})
        self._is_controlled = True
        return response

    async def disable_torque(self) -> dict:
        """
        Disable torque on all motors.

        Returns:
            Confirmation event dictionary from the server.

        Raises:
            WebSocketResponseError: If no response or invalid response.
        """
        response = await self._send_command({"command": "disable_torque"})
        self._is_controlled = False
        return response

    async def read_state(self, *, normalize: bool = True) -> dict:
        """
        Read the current position of all joints.

        First checks the buffer for any unsolicited state updates.
        If buffer is empty, sends a read_state command to the server.

        Args:
            normalize: Whether to return normalized values.

        Returns:
            Event dictionary containing the current joint states.

        Raises:
            WebSocketResponseError: If no response or invalid response.
        """
        # Check buffer first for any buffered state updates
        if self._state_buffer:
            state = self._state_buffer.popleft()
            logger.debug("Returning buffered state update")
            return state

        # No buffered state, request fresh state from server
        return await self._send_command({"command": "read_state", "normalize": normalize})

    async def _send_command(self, command: dict) -> dict:
        """
        Send a command to the remote server and wait for response.

        Args:
            command: Command dictionary to send.

        Returns:
            Response dictionary from the server.

        Raises:
            WebSocketConnectionError: If not connected.
            WebSocketResponseError: If response times out or is invalid.
        """
        if not self.is_connected or self._websocket is None:
            raise WebSocketConnectionError("Not connected to robot server")

        async with self._lock:
            try:
                # Send command as JSON
                command_json = json.dumps(command)
                await self._websocket.send(command_json)
                logger.debug("Sent command: {}", command.get("command"))

                # Wait for response with timeout
                response = await self._receive_response()
                return response

            except websockets.exceptions.ConnectionClosed as e:
                self._is_connected = False
                logger.error("Connection closed while sending command: {}", e)
                raise WebSocketConnectionError(f"Connection closed: {e}") from e

    async def _receive_response(self) -> dict:
        """
        Receive and parse a response from the server.

        Handles buffering of unsolicited state updates while waiting
        for the expected response.

        Returns:
            Response dictionary.

        Raises:
            WebSocketResponseError: If response times out or is invalid.
        """
        if self._websocket is None:
            raise WebSocketConnectionError("Not connected")

        try:
            response_text = await asyncio.wait_for(
                self._websocket.recv(),
                timeout=RECEIVE_TIMEOUT,
            )

            response = json.loads(response_text)

            if not isinstance(response, dict):
                raise WebSocketResponseError(f"Expected dict response, got {type(response).__name__}")

            # Check if this is a state update that should be buffered
            event_type = response.get("event")
            if event_type == "state_was_updated":
                # Update is_controlled from server response if present
                if "is_controlled" in response:
                    self._is_controlled = response["is_controlled"]

            logger.debug("Received response: {}", event_type)
            return response

        except asyncio.TimeoutError as e:
            raise WebSocketResponseError(f"Response timeout after {RECEIVE_TIMEOUT}s") from e
        except json.JSONDecodeError as e:
            raise WebSocketResponseError(f"Invalid JSON response: {e}") from e
        except websockets.exceptions.ConnectionClosed as e:
            self._is_connected = False
            raise WebSocketConnectionError(f"Connection closed: {e}") from e

    async def _buffer_state_update(self, state: dict) -> None:
        """
        Buffer an unsolicited state update for later retrieval.

        Args:
            state: State update dictionary to buffer.
        """
        self._state_buffer.append(state)
        logger.debug("Buffered state update, buffer size: {}", len(self._state_buffer))
