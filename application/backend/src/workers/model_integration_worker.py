import asyncio
import ctypes
import multiprocessing as mp
from multiprocessing.synchronize import Event as EventClass

from control.environment_data_manifest import EnvironmentDataManifest
from control.utils import format_observation_for_model, get_observation_from_manifest
from schemas.model import Model
from workers.base import BaseProcessWorker, run_at_frequency
from workers.model_worker import ModelWorker
from workers.remote_model_worker import RemoteModelWorker


class ModelIntegration(BaseProcessWorker):
    ROLE = "ModelIntegrationWorker"

    _child_workers: list[BaseProcessWorker] = []

    def __init__(
        self,
        model: Model,
        backend: str,
        data_manifest: EnvironmentDataManifest,
        mp_terminate_event: EventClass,
        event_queue: mp.Queue,
    ):
        super().__init__(stop_event=mp_terminate_event, queues_to_cancel=[])
        self.loaded_event = mp.Event()
        self.data_manifest = data_manifest
        self.backend = backend
        self.model = model
        self.event_queue = event_queue
        self.model_integration = None
        self.is_running = False
        self.fps = 30  # TODO FPS
        self._task_buf = mp.Array(ctypes.c_char, 256)
        self._start_task_event = mp.Event()
        self._stop_task_event = mp.Event()

    async def setup(self) -> None:
        from control.sync_mixed_model_integration import SyncMixedModelIntegration

        try:
            remote_worker = RemoteModelWorker()
            remote_worker.connect()
            remote_worker.load_model(self.model, self.backend)
            self.model_integration = SyncMixedModelIntegration(model_worker=remote_worker, fps=self.fps)
        except ConnectionRefusedError:
            model_worker = ModelWorker(self.model, self.backend, stop_event=self._interrupt_event)
            self.model_integration = SyncMixedModelIntegration(model_worker=model_worker, fps=self.fps)
            model_worker.start()
            self._child_workers.append(model_worker)
        assert self.model_integration is not None  # noqa: S101
        await self.model_integration.setup()
        self.loaded_event.set()

    def start_task(self, task: str) -> None:
        self._task_buf.get_obj().value = task.encode("utf-8")[:255]  # type: ignore[misc, assignment]
        self._start_task_event.set()

    def stop_task(self) -> None:
        self._stop_task_event.set()

    async def run_loop(self) -> None:
        while not self.should_stop():
            async with run_at_frequency(self.fps):
                await asyncio.gather(
                    self._handle_start_task(),
                    self._handle_stop_task(),
                )

                if self.model_integration and self.is_running:
                    obs = get_observation_from_manifest(self.data_manifest)
                    observation = format_observation_for_model(obs, self.data_manifest)
                    action = self.model_integration.select_action(observation)
                    if action is not None:
                        with self.data_manifest.robot.actions.get_lock():
                            self.data_manifest.robot.actions.get_obj()[:] = action

    def get_task(self) -> str:
        return bytes(self._task_buf.get_obj()).rstrip(b"\x00").decode()

    async def teardown(self) -> None:
        for worker in self._child_workers:
            worker.stop()

        if self.model_integration:
            self.model_integration.teardown()

    async def _handle_start_task(self) -> None:
        if self._start_task_event.is_set():
            self._start_task_event.clear()
            self.is_running = True
            self.event_queue.put_nowait(
                {
                    "event": "start_task",
                    "state": {"is_running": True},
                }
            )

    async def _handle_stop_task(self) -> None:
        if self._stop_task_event.is_set():
            self._stop_task_event.clear()
            self.is_running = False
            self.event_queue.put_nowait(
                {
                    "event": "stop_task",
                    "state": {"is_running": False},
                }
            )
