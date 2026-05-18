import ctypes
from workers.base import BaseProcessWorker
import asyncio
import time

import cv2
import numpy as np
from frame_source import FrameSourceFactory
from frame_source.video_capture_base import VideoCaptureBase
from loguru import logger
from multiprocessing import Queue, Event, Array

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
        super().__init__(stop_event=mp_stop_event, queues_to_cancel=[])
        self.loaded_event = Event()
        self._width = config.payload.width or 640
        self._height = config.payload.height or 480
        self._frame_data = Array(ctypes.c_uint8, self._width * self._height * 3)
        self.config = config

    def get_frame(self) -> np.ndarray:
        with self._frame_data.get_lock():
            return self.frame_from_buffer(self._frame_data.get_obj(), self._width, self._height)

    def _set_frame(self, data: np.ndarray) -> None:
        if data.shape[:2] != (self._height, self._width):
            data = cv2.resize(data, (self._width, self._height))
        with self._frame_data.get_lock():
            np.frombuffer(self._frame_data.get_obj(), dtype=np.uint8)[:] = data.reshape(-1)

    @staticmethod
    def frame_from_buffer(buffer, width: int, height: int) -> np.ndarray:
        return np.frombuffer(buffer, dtype=np.uint8).reshape(height, width, 3).copy()

    async def setup(self) -> None:
        self.camera = create_frames_source_from_camera(self.config)
        self.camera.connect()
        self.loaded_event.set()

    async def run_loop(self) -> None:
        """Main worker loop."""
        try:
            target_dt = 1 / self.config.payload.fps
            while not self.should_stop():
                t0 = time.perf_counter()
                success, frame = self.camera.read()
                if success and frame is not None:
                    self._set_frame(frame)
                elapsed = time.perf_counter() - t0
                sleep_time = target_dt - elapsed
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
        except Exception as e:
            logger.error(f"Frame capture error: {e}")
            raise

    async def teardown(self) -> None:
        #self.frame_queue.close()
        self.camera.disconnect()
