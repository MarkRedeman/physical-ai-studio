from workers.base import BaseProcessWorker
import asyncio
import time

import cv2
import numpy as np
from frame_source import FrameSourceFactory
from frame_source.video_capture_base import VideoCaptureBase
from loguru import logger
from multiprocessing import Queue, Event

from multiprocessing.synchronize import Event as EventClass
from schemas.project_camera import Camera


def create_frames_source_from_camera(camera: Camera) -> VideoCaptureBase:
    """Very FrameSource factory call from camera schema object."""
    return FrameSourceFactory.create(
        "webcam" if camera.driver == "usb_camera" else camera.driver,
        camera.fingerprint,
        **camera.payload.model_dump(),
    )


class EmptyFrameError(Exception):
    pass


class CameraWorker(BaseProcessWorker):
    """Orchestrates camera streaming over configurable transport."""
    ROLE="Camera"

    def __init__(
        self,
        config: Camera,
        mp_stop_event: EventClass
    ):
        super().__init__(stop_event=mp_stop_event)
        self.stop_event = Event()
        self.config = config
        self.frame_queue = Queue()

    async def setup(self) -> None:
        self.camera = create_frames_source_from_camera(self.config)
        self.camera.connect()

    async def run_loop(self) -> None:
        """Main worker loop."""
        try:
            target_dt = 1 / self.config.payload.fps
            while not self.should_stop():
                t0 = time.perf_counter()
                success, frame = self.camera.read()
                success, jpeg = cv2.imencode(".jpg", frame) #TODO just send bytes instead?
                if not success or jpeg is None:
                    raise RuntimeError("Failed to encode frame")
                self.frame_queue.put_nowait(jpeg.tobytes())

                elapsed = time.perf_counter() - t0
                sleep_time = target_dt - elapsed
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
        except Exception as e:
            logger.error(f"Frame capture error: {e}")
            raise

    async def teardown(self) -> None:
        self.frame_queue.close()
        self.camera.disconnect()
