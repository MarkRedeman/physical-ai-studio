import asyncio
import multiprocessing as mp
from multiprocessing.synchronize import Event as EventClass
from typing import Literal

from pydantic import BaseModel

from control.environment_integration import EnvironmentIntegration
from control.utils import format_observation_for_reporting, get_observation_from_manifest
from robots.robot_client_factory import RobotClientFactory
from schemas.dataset import Dataset
from schemas.environment import EnvironmentWithRelations
from schemas.model import Model
from workers.model_integration_worker import ModelIntegration
from workers.base import BaseThreadWorker, run_at_frequency
from workers.recording_worker import RecordingWorker


class RobotControlState(BaseModel):
    task: str | None = None
    model_loaded: bool = False
    dataset_loaded: bool = False
    environment_loaded: bool = False
    is_recording: bool = False
    follower_source: Literal["model", "teleoperation"] | None = None
    episodes_recorded: int = 0

MESSAGE_QUEUE_FREQUENCY = 10

class RobotControlOrchestrator(BaseThreadWorker):
    ROLE="RobotControlOrchestrator"

    environment: EnvironmentIntegration | None = None
    recording: RecordingWorker | None = None
    model: ModelIntegration | None = None

    def __init__(
        self, message_queue: asyncio.Queue, robot_client_factory: RobotClientFactory, mp_terminate_event: EventClass
    ):
        super().__init__(stop_event=mp_terminate_event)
        self.state = RobotControlState()
        self.event_queue = mp.Queue()
        self._mp_terminate_event = mp_terminate_event
        self.robot_client_factory = robot_client_factory
        self.message_queue = message_queue

    async def run_loop(self) -> None:
        while not self.should_stop():
            async with run_at_frequency(MESSAGE_QUEUE_FREQUENCY):
                while not self.event_queue.empty():
                    self._handle_event(self.event_queue.get())

    def _handle_event(self, event: dict) -> None:
        if event["event"] == "start_recording":
            self.state.is_recording = event["state"]["is_recording"]
            self._report_state()
        if event["event"] == "save_episode":
            self.state.is_recording = event["state"]["is_recording"]
            self.state.episodes_recorded = event["state"]["episodes_recorded"]
            self._report_state()
        if event["event"] == "discard_episode":
            self.state.is_recording = event["state"]["is_recording"]
            self._report_state()
        if event["event"] == "start_task":
            self.state.follower_source = "model" if event["state"]["is_running"] else "teleoperation"
            self._report_state()
        if event["event"] == "stop_task":
            self.state.follower_source = "model" if event["state"]["is_running"] else "teleoperation"
            self._report_state()

    async def load_environment(self, environment: EnvironmentWithRelations) -> None:
        """Load environment with cameras and robots."""
        try:
            environment_integration = EnvironmentIntegration(
                environment=environment,
                robot_client_factory=self.robot_client_factory,
                mp_terminate_event=self._mp_terminate_event,
            )
            await environment_integration.setup_environment()
            self.environment = environment_integration
            self.state.environment_loaded = True
        except Exception as e:
            self._report_error("environment", e)
        finally:
            self._report_state()

    async def load_dataset(self, dataset: Dataset) -> None:
        """Load dataset and setup recording."""
        if self.environment and self.environment.manifest:
            try:
                worker = RecordingWorker(
                    dataset=dataset,
                    data_manifest=self.environment.manifest,
                    mp_terminate_event=self._mp_terminate_event,
                    event_queue=self.event_queue,
                )
                worker.start()
                await asyncio.to_thread(worker.loaded_event.wait)
                self.recording = worker
                self.state.dataset_loaded = True
            except Exception as e:
                self._report_error("dataset", e)
            finally:
                self._report_state()
        else:
            self._report_error("dataset", ValueError("Cannot load dataset without environment."))

    async def load_model(self, model: Model, backend: str) -> None:
        """Load model for inference."""
        if self.environment and self.environment.manifest:
            try:
                worker = ModelIntegration(
                    model=model,
                    backend=backend,
                    data_manifest=self.environment.manifest,
                    mp_terminate_event=self._mp_terminate_event,
                    event_queue=self.event_queue
                )
                worker.start()
                await asyncio.to_thread(worker.loaded_event.wait)
                self.model = worker
                self.state.model_loaded = True
            except Exception as e:
                self._report_error("model", e)
            finally:
                self._report_state()

    def start_recording(self, task: str) -> None:
        """Start recording of specified task."""
        if self.recording:
            self.recording.start_episode(task)

    def save_episode(self) -> None:
        """Save recording."""
        if self.recording:
            self.recording.save_episode()

    def discard_episode(self) -> None:
        """Discard episode."""
        if self.recording:
            self.recording.discard_episode()

    def start_task(self, task: str) -> None:
        """Start task on model."""
        if self.model:
            self.model.start_task(task)
            self.set_follower_source("model")

    def stop_task(self) -> None:
        """Stop executing actions from model."""
        if self.model:
            self.model.stop_task()
            self.set_follower_source(None)

    def set_follower_source(self, follower_source: Literal["model", "teleoperation"] | None) -> None:
        """Sets teleoperation loop to follow either model or teleoperator."""
        if self.environment and self.environment.manifest:
            action_source = 0
            if follower_source == "teleoperation":
                action_source = 1
            if follower_source == "model":
                action_source = 2
            self.environment.manifest.robot.action_source.value = action_source
            self.state.follower_source = follower_source
            self._report_state()


    async def teardown(self) -> None:
        if self.environment:
            self.environment.teardown()
        if self.recording:
            self.recording.stop()
        if self.model:
            self.model.stop()

        self.event_queue.close()

    def get_observation(self) -> dict | None:
        if self.environment and self.environment.manifest:
            obs = get_observation_from_manifest(self.environment.manifest)
            return format_observation_for_reporting(obs, self.environment.manifest)
        return None

    def _report_state(self):
        self.message_queue.put_nowait(
            {
                "event": "state",
                "data": self.state.model_dump(),
            }
        )

    def _report_error(self, component: str, error: BaseException):
        """Report error in application."""
        self.message_queue.put_nowait(
            {
                "event": "error",
                "component": component,
                "data": str(error),
            }
        )
