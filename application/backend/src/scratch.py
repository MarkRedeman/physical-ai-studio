import time
from multiprocessing import Event, set_start_method
from workers.base import BaseProcessWorker

from loguru import logger


class CameraWorker:
    pass


class TeleoperateWorker:
    pass


class RobotControlWorker(BaseProcessWorker):
    async def setup(self) -> None:
        await super().setup()
        self.duration = 5

    async def run_loop(self) -> None:

        start = time.perf_counter()
        logger.info(f"start example worker for {self.duration}")
        while not self.should_stop() and time.perf_counter() - start < self.duration:
            logger.info(f"{time.perf_counter() - start}")
            time.sleep(0.5)
        logger.info("done")
    

if __name__ == "__main__":
    stop_event = Event()
    robot_control = RobotControlWorker(stop_event=stop_event)
    robot_control.start()
    set_start_method("spawn", force=True)
    try:
        time.sleep(10)
    finally:
        stop_event.set()
        robot_control.join()
