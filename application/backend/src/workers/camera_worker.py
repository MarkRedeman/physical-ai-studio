from workers.base import BaseWorker
import asyncio

import cv2
import numpy as np
from fastapi.websockets import WebSocketDisconnect
from frame_source import FrameSourceFactory
from frame_source.video_capture_base import VideoCaptureBase
from loguru import logger
from multiprocessing import Queue, Event

from schemas.project_camera import Camera
from utils.async_camera_capture import AsyncCameraCapture
from workers.transport.worker_transport import WorkerTransport
from workers.transport_worker import TransportWorker, WorkerState, WorkerStatus


def create_frames_source_from_camera(camera: Camera) -> VideoCaptureBase:
    """Very FrameSource factory call from camera schema object."""
    return FrameSourceFactory.create(
        "webcam" if camera.driver == "usb_camera" else camera.driver,
        camera.fingerprint,
        **camera.payload.model_dump(),
    )


class EmptyFrameError(Exception):
    pass


class CameraWorker(BaseWorker):
    """Orchestrates camera streaming over configurable transport."""
    ROLE="Camera"

    def __init__(
        self,
        config: Camera,
    ):
        super().__init__()
        self.config = config
        self.frame_queue = Queue()

    def setup(self) -> None:
        logger.info("Setting up camera...")
        self.camera = create_frames_source_from_camera(self.config)
        self.camera.connect()

    def run_loop(self) -> None:
        """Main worker loop."""
        try:
            while not self.should_stop():
                success, frame = self.camera.read()
                success, jpeg = cv2.imencode(".jpg", frame) #TODO just send bytes instead?
                if not success or jpeg is None:
                    raise RuntimeError("Failed to encode frame")
                self.frame_queue.put_nowait(jpeg)
        except Exception as e:
            logger.error(f"Frame capture error: {e}")
            self._stop_requested = True
            raise

    def teardown(self) -> None:
        self.camera.disconnect()
