import asyncio
import threading
import time
from typing import Any, Optional

import cv2
import numpy as np
import websockets
from loguru import logger
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from frame_source import VideoCaptureBase

# Constants
MAX_RECONNECT_ATTEMPTS = 3
DEFAULT_RECONNECT_DELAY = 1.0  # seconds
MAX_RECONNECT_DELAY = 10.0  # seconds
RECEIVE_TIMEOUT = 5.0  # seconds
CONNECTION_TIMEOUT = 10.0  # seconds
DEFAULT_FPS = 30


class WebSocketFrameError(Exception):
    """Raised when an invalid or empty frame is received from WebSocket."""

    pass


class WebSocketCapture(VideoCaptureBase):
    """
    Capture video frames from a WebSocket connection.

    This class receives raw JPEG bytes from the WebSocket, decodes them to numpy arrays,
    and presents them through the standard VideoCaptureBase interface.

    Args:
        source: WebSocket URL (e.g., 'ws://localhost:8000/video' or 'wss://...')
        fps: Target frame rate for consumption (default: 30)
        **kwargs: Additional parameters passed to base class
    """

    def __init__(self, source: str, fps: int = DEFAULT_FPS, **kwargs):
        """
        Initialize WebSocket capture.

        Args:
            source: WebSocket URL (e.g., 'ws://localhost:8000/video' or 'wss://...')
            fps: Target frame rate for frame consumption limiting
            **kwargs: Additional parameters
        """
        self._lock = threading.Lock()
        super().__init__(source, **kwargs)
        self._fps = fps
        self._target_frame_time = 1.0 / fps
        self._last_read_time: float = 0.0
        self._frame_queue: Optional[asyncio.Queue] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event: Optional[threading.Event] = None
        self._connected_event: Optional[threading.Event] = None
        self._last_frame: Optional[np.ndarray] = None
        self._is_connected = False

    @property
    def fps(self) -> int:
        """Get the target frame rate."""
        return self._fps

    @fps.setter
    def fps(self, value: int) -> None:
        """Set the target frame rate."""
        if value <= 0:
            raise ValueError("FPS must be positive")
        self._fps = value
        self._target_frame_time = 1.0 / value

    @property
    def is_connected(self) -> bool:
        """Thread-safe getter for connection status."""
        with self._lock:
            return self._is_connected

    @is_connected.setter
    def is_connected(self, value: bool) -> None:
        """Thread-safe setter for connection status."""
        with self._lock:
            self._is_connected = value

    def connect(self) -> bool:
        """Connect to WebSocket server."""
        try:
            logger.info("Connecting to WebSocket: {}", self.source)
            self._stop_event = threading.Event()
            self._connected_event = threading.Event()
            self._is_connected = False

            self._thread = threading.Thread(target=self._run_event_loop, daemon=True)
            self._thread.start()

            # Wait for connection with timeout
            if self._connected_event.wait(timeout=CONNECTION_TIMEOUT):
                if self.is_connected:
                    logger.info("Connected to WebSocket: {}", self.source)
                    return True

            logger.error("Failed to establish WebSocket connection: {}", self.source)
            self._stop_event.set()
            return False

        except Exception as e:
            logger.error("Error connecting to WebSocket: {}", e)
            return False

    def disconnect(self) -> bool:
        """Disconnect from WebSocket server."""
        if not self._stop_event:
            return True

        try:
            self._stop_event.set()

            if self._thread:
                self._thread.join(timeout=2)

            self.is_connected = False
            with self._lock:
                self._last_frame = None
            logger.info("Disconnected from WebSocket")
            return True

        except Exception as e:
            logger.error("Error disconnecting from WebSocket: {}", e)
            self.is_connected = False
            return False

    def _run_event_loop(self) -> None:
        """Run the asyncio event loop for WebSocket connection."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._frame_queue = asyncio.Queue(maxsize=1)

        try:
            self._loop.run_until_complete(self._connection_loop())
        except RetryError as e:
            logger.error(
                "WebSocket connection failed after {} attempts: {}",
                MAX_RECONNECT_ATTEMPTS,
                e.last_attempt.exception(),
            )
        except Exception as e:
            logger.error("WebSocket loop error: {}", e)
        finally:
            self._loop.close()
            self.is_connected = False

    async def _connection_loop(self) -> None:
        """
        Main connection loop with automatic reconnection using tenacity.

        Attempts to maintain a connection to the WebSocket server,
        reconnecting with exponential backoff on failures up to MAX_RECONNECT_ATTEMPTS.
        """
        await self._receive_frames_with_retry()

    @retry(
        stop=stop_after_attempt(MAX_RECONNECT_ATTEMPTS),
        wait=wait_exponential(multiplier=1, min=DEFAULT_RECONNECT_DELAY, max=MAX_RECONNECT_DELAY),
        retry=retry_if_exception_type((websockets.ConnectionClosed, ConnectionError, OSError)),
        reraise=True,
    )
    async def _receive_frames_with_retry(self) -> None:
        """Connect to WebSocket and receive frames with automatic retry on connection failures."""
        try:
            await self._receive_frames()
        except websockets.ConnectionClosed as e:
            logger.warning("WebSocket connection closed: {}", e)
            self.is_connected = False
            raise
        except (ConnectionError, OSError) as e:
            logger.warning("WebSocket connection error: {}", e)
            self.is_connected = False
            raise

    async def _receive_frames(self) -> None:
        """Connect to WebSocket and receive frames as raw JPEG bytes."""
        async with websockets.connect(self.source) as websocket:
            self.is_connected = True
            self._connected_event.set()  # type: ignore[union-attr]
            logger.debug("WebSocket connection established")

            while not self._stop_event.is_set():  # type: ignore[union-attr]
                try:
                    # Use timeout to allow checking stop_event periodically
                    frame_bytes = await asyncio.wait_for(websocket.recv(), timeout=RECEIVE_TIMEOUT)

                    # Validate we received bytes
                    if not isinstance(frame_bytes, bytes):
                        raise WebSocketFrameError(f"Expected bytes, received {type(frame_bytes).__name__}")

                    if len(frame_bytes) == 0:
                        raise WebSocketFrameError("Received empty frame")

                    # Decode JPEG bytes to numpy array
                    self._decode_and_store_frame(frame_bytes)

                except asyncio.TimeoutError:
                    # Timeout is expected - allows us to check stop_event
                    continue
                except WebSocketFrameError as e:
                    logger.warning("Invalid frame received: {}", e)
                    continue
                except websockets.ConnectionClosed:
                    raise

    def _decode_and_store_frame(self, frame_bytes: bytes) -> None:
        """
        Decode JPEG bytes to numpy array and store in queue.

        Args:
            frame_bytes: Raw JPEG bytes from WebSocket
        """
        try:
            # Decode JPEG to numpy array
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                raise WebSocketFrameError("Failed to decode JPEG frame")

            # Cache the last frame
            with self._lock:
                self._last_frame = frame

            # Update queue
            try:
                self._frame_queue.put_nowait((True, frame))  # type: ignore[union-attr]
            except asyncio.QueueFull:
                # Drop old frame and add new one
                try:
                    self._frame_queue.get_nowait()  # type: ignore[union-attr]
                except asyncio.QueueEmpty:
                    pass
                try:
                    self._frame_queue.put_nowait((True, frame))  # type: ignore[union-attr]
                except asyncio.QueueFull:
                    pass

        except WebSocketFrameError as e:
            logger.warning("Frame decode error: {}", e)
        except Exception as e:
            logger.error("Unexpected error decoding frame: {}", e)

    def _read_implementation(self) -> tuple[bool, Optional[np.ndarray]]:  # type: ignore[override]
        """
        Read decoded frame from the WebSocket connection.

        Applies frame rate limiting based on the configured FPS.

        Returns:
            Tuple of (success, frame) where frame is a numpy array (BGR format from cv2.imdecode)
        """
        if not self.is_connected or self._frame_queue is None:
            return False, None

        # Frame rate limiting
        current_time = time.perf_counter()
        elapsed = current_time - self._last_read_time
        if elapsed < self._target_frame_time:
            sleep_time = self._target_frame_time - elapsed
            time.sleep(sleep_time)

        self._last_read_time = time.perf_counter()

        try:
            return self._frame_queue.get_nowait()
        except asyncio.QueueEmpty:
            # Return cached last frame if available
            with self._lock:
                if self._last_frame is not None:
                    return True, self._last_frame.copy()
            return False, None

    def set_exposure(self, value: float) -> bool:
        """Set exposure (not supported for WebSocket streams)."""
        logger.warning("Exposure control not supported for WebSocket streams")
        self._exposure = value
        return False

    def get_exposure(self) -> Optional[float]:
        """Get exposure."""
        return self._exposure

    def set_gain(self, value: float) -> bool:
        """Set gain (not supported for WebSocket streams)."""
        logger.warning("Gain control not supported for WebSocket streams")
        self._gain = value
        return False

    def get_gain(self) -> Optional[float]:
        """Get gain."""
        return self._gain

    def enable_auto_exposure(self, enable: bool = True) -> bool:
        """Enable or disable auto exposure (not supported for WebSocket streams)."""
        logger.warning("Auto exposure control not supported for WebSocket streams")
        return False

    def set_frame_size(self, width: int, height: int) -> bool:
        """Set frame size (not supported for WebSocket streams)."""
        logger.warning("Frame size control not supported for WebSocket streams")
        return False

    def get_frame_size(self) -> Optional[tuple[int, int]]:
        """
        Get frame size from last decoded frame.

        Returns:
            Optional[tuple[int, int]]: (width, height) or None if no frame available
        """
        with self._lock:
            if self._last_frame is not None:
                height, width = self._last_frame.shape[:2]
                return width, height
        return None

    @classmethod
    def discover(cls) -> list[str]:
        """Discover method for WebSocket capture."""
        logger.info("WebSocketCapture requires manual configuration with WebSocket URLs.")
        return []

    @classmethod
    def get_config_schema(cls) -> dict[str, Any]:
        """Get configuration schema for WebSocket capture."""
        return {
            "title": "WebSocket Camera Configuration",
            "description": "Configure WebSocket connection for video streaming",
            "fields": [
                {
                    "name": "source",
                    "label": "WebSocket URL",
                    "type": "text",
                    "placeholder": "ws://localhost:8000/video",
                    "description": "WebSocket URL (ws:// or wss://)",
                    "required": True,
                },
                {
                    "name": "fps",
                    "label": "Frame Rate",
                    "type": "number",
                    "placeholder": "30",
                    "description": "Target frame rate for consumption",
                    "required": False,
                    "default": DEFAULT_FPS,
                },
            ],
        }

    def get_supported_formats(self) -> Optional[list[dict]]:
        """Get supported formats (JPEG for WebSocket)."""
        return [{"format": "JPEG", "description": "JPEG frames received and decoded"}]
