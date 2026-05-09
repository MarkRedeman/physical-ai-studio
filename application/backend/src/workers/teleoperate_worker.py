import asyncio
import multiprocessing as mp
import queue
import time
from multiprocessing.synchronize import Event as EventClass

from loguru import logger

from robots.robot_client import RobotClient

from .base import BaseProcessWorker


class TeleoperateWorker(BaseProcessWorker):
    ROLE: str = "TeleoperateWorker"

    leader: RobotClient
    follower: RobotClient

    command_queue: mp.Queue
    output_queue: mp.Queue

    robots_loaded_event: EventClass
    disconnect_robot_event: EventClass

    def __init__(self, stop_event: EventClass):
        self.command_queue = mp.Queue()
        self.output_queue = mp.Queue(maxsize=1)
        super().__init__(
            stop_event=stop_event,
            queues_to_cancel=[self.command_queue, self.output_queue],
        )
        self.robots_loaded_event = mp.Event()
        self.disconnect_robot_event = mp.Event()

    @property
    def is_loaded(self) -> bool:
        return self.robots_loaded_event.is_set()

    def load_teleoperator(self, leader: RobotClient | None, follower: RobotClient, goal_time: float) -> None:
        """Send a load command to the worker process."""
        self.command_queue.put(("load", leader, follower, goal_time))

    async def disconnect_robots(self) -> None:
        """Signal the worker to stop teleoperation, disconnect robots and return to idle."""
        self.disconnect_robot_event.set()
        #TODO: Wait for disconnect to be ready

    async def wait_for_loading_to_complete(self) -> None:
        await asyncio.to_thread(self.robots_loaded_event.wait)

    async def run_loop(self) -> None:
        """Idle → load → loop teleoperate → disconnect → idle cycle."""

        while not self.should_stop():


            # Wait for a load command
            try:
                cmd = self.command_queue.get(timeout=1)
            except queue.Empty:
                continue

            if cmd[0] != "load":
                continue

            _, self.leader, self.follower, goal_time = cmd
            try:
                if self.leader is not None:
                    await self.leader.connect()
                await self.follower.connect()
                logger.info(f"Loading : {self.leader} ({self.follower})")
                self.robots_loaded_event.set()

                # Teleoperate loop until unload is requested
                while not self.should_stop() and not self.disconnect_robot_event.is_set():
                    start_loop_t = time.perf_counter()

                    if self.leader is not None:
                        actions = (await self.leader.read_state())["state"]
                        await self.follower.set_joints_state(actions, goal_time)
                        self.output_queue.put(actions)
                    else:
                        pass

                    dt_s = time.perf_counter() - start_loop_t
                    wait_time = goal_time - dt_s

                    if wait_time > 0:
                        await asyncio.sleep(wait_time)
                    else:
                        await asyncio.sleep(0)

            finally:
                logger.info("Teleoperating stopped, disconnecting robots.")
                self.disconnect_robot_event.clear()
                self.robots_loaded_event.clear()

                if self.leader:
                    await self.leader.disconnect()
                if self.follower:
                    await self.follower.disconnect()

    async def teardown(self) -> None:
        self.command_queue.close()
        self.output_queue.close()
        await super().teardown()
