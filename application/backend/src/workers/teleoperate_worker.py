import ctypes
import asyncio
import multiprocessing as mp
import queue
import time
from multiprocessing.synchronize import Event as EventClass

from loguru import logger

from robots.robot_client import RobotClient

from .base import BaseProcessWorker

BUFFER_LENGTH = 50

class TeleoperateWorker(BaseProcessWorker):
    ROLE: str = "TeleoperateWorker"

    leader: RobotClient | None
    follower: RobotClient

    command_queue: mp.Queue

    robots_loaded_event: EventClass
    disconnect_robot_event: EventClass

    def __init__(self, stop_event: EventClass):
        self.command_queue = mp.Queue()
        self._output_actions = mp.Array(ctypes.c_double, BUFFER_LENGTH)
        self._output_state = mp.Array(ctypes.c_double, BUFFER_LENGTH)
        super().__init__(
            stop_event=stop_event,
            queues_to_cancel=[self.command_queue],
        )
        self.robots_loaded_event = mp.Event()
        self.disconnect_robot_event = mp.Event()

    def get_state(self) -> list[float]:
        with self._output_state.get_lock():
            return list(self._output_state.get_obj())

    def _set_state(self, data: list[float]) -> None:
        with self._output_state.get_lock():
            state =  data + [0] * (BUFFER_LENGTH - len(data))
            self._output_state.get_obj()[:] = state

    def get_actions(self) -> list[float]:
        with self._output_actions.get_lock():
            return list(self._output_actions.get_obj())

    def _set_actions(self, data: list[float]) -> None:
        with self._output_actions.get_lock():
            actions =  data + [0] * (BUFFER_LENGTH - len(data))
            self._output_actions.get_obj()[:] = actions

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
                features = self.follower.features()
                if self.leader is not None:
                    await self.leader.connect()
                await self.follower.connect()
                logger.info(f"Loading : {self.leader} ({self.follower})")
                self.robots_loaded_event.set()

                # Teleoperate loop until unload is requested
                while not self.should_stop():
                    start_loop_t = time.perf_counter()

                    state = (self.follower.read_state())["state"]
                    self._set_state([state[key] for key in features])
                    if self.leader is not None:
                        actions = (self.leader.read_state())["state"]
                        self.follower.set_joints_state(actions, goal_time * 5)
                        self._set_actions([actions[key] for key in features])

                    dt_s = time.perf_counter() - start_loop_t
                    wait_time = goal_time - dt_s

                    if wait_time > 0:
                        await asyncio.sleep(wait_time)
                    else:
                        logger.warning(f"Did not meet target framespeed by {0 - wait_time}, {dt_s * 1000}ms")
                        await asyncio.sleep(0)

            finally:
                logger.info("Teleoperating stopped, disconnecting robots.")
                self.disconnect_robot_event.clear()
                self.robots_loaded_event.clear()

                if self.leader:
                    await self.leader.disconnect()
                if self.follower:
                    await self.follower.disconnect()

    def should_stop(self) -> bool:
        return super().should_stop() or self.disconnect_robot_event.is_set()

    async def teardown(self) -> None:
        self.command_queue.close()
        await super().teardown()
