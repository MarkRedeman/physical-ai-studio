from workers.base import BaseProcessWorker, run_at_frequency
import base64
import cv2
import numpy as np
import asyncio
from dataclasses import dataclass, field
import multiprocessing as mp
from multiprocessing.synchronize import Event as EventClass
from typing import Any, Literal

from pathlib import Path
from pydantic import BaseModel

from robots.robot_client_factory import RobotClientFactory
from schemas.dataset import Dataset
from schemas.environment import EnvironmentWithRelations, TeleoperatorRobotWithRobot
from schemas.model import Model
from workers.camera_worker import CameraWorker
from workers.teleoperate_worker import TeleoperateWorker

@dataclass
class CameraManifestEntry:
    id: str
    name: str
    width: int
    height: int
    frame_data: Any  # mp.Array[c_uint8], shared with CameraWorker


@dataclass
class RobotManifestEntry:
    name: str
    type: str
    features: list[str]
    state: Any    # mp.Array[c_double], shared with TeleoperateWorker
    actions: Any  # mp.Array[c_double], shared with TeleoperateWorker


@dataclass
class EnvironmentDataManifest:
    """Describes all data streams produced by a loaded environment."""
    robot: RobotManifestEntry
    cameras: list[CameraManifestEntry] = field(default_factory=list)

def build_lerobot_dataset_features(manifest: EnvironmentDataManifest, use_videos: bool = True) -> dict:
    """Build lerobot dataset features."""
    from lerobot.datasets.feature_utils import combine_feature_dicts
    from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
    from lerobot.processor import make_default_processors

    teleop_action_processor, _robot_action_processor, robot_observation_processor = make_default_processors()
    action_features: dict[str, Any] = {}
    observation_features: dict[str, Any] = {}
    for feature in manifest.robot.features:
        action_features[feature] = float
        observation_features[feature] = float

    for camera in manifest.cameras:
        observation_features[camera.name.lower()] = (camera.height, camera.width, 3)

    return combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=action_features),
            use_videos=use_videos,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=observation_features),
            use_videos=use_videos,
        ),
    )

def get_observation_from_manifest(manifest: EnvironmentDataManifest, timestamp: float = 0) -> dict:
    """Lightweight read-only get data from environments SharedMemory."""
    actions = list(manifest.robot.actions.get_obj())
    state = list(manifest.robot.state.get_obj())

    camera_images = {}
    for camera in manifest.cameras:
        frame = CameraWorker.frame_from_buffer(camera.frame_data.get_obj(), camera.width, camera.height)
        camera_images[camera.id] = frame

    return {
        "state": state,
        "action": actions,
        "images": camera_images,
        "timestamp": timestamp,
    }

def format_observation_for_dataset(observation: dict, manifest: EnvironmentDataManifest) -> tuple[dict, dict]:
    """Format observation for input of a dataset."""
    result = {i: observation["state"][k] for k,i in enumerate(manifest.robot.features)}
    actions = {i: observation["action"][k] for k,i in enumerate(manifest.robot.features)}
    for camera in manifest.cameras:
        camera_name = camera.name.lower()
        # RGB2BGR
        result[camera_name] = np.ascontiguousarray(observation["images"][camera.id][..., ::-1])


    return result, actions

def format_observation_for_reporting(observation: dict, manifest: EnvironmentDataManifest) -> dict:
    """Format observation for UI."""
    actions = {i: observation["action"][k] for k,i in enumerate(manifest.robot.features)}
    state = {i: observation["state"][k] for k,i in enumerate(manifest.robot.features)}
    camera_images = {}
    for camera in manifest.cameras:
        frame = observation["images"][camera.id]
        _, imagebytes = cv2.imencode(".jpg", frame)
        camera_images[camera.id] = base64.b64encode(imagebytes).decode()

    return {
        "state": state,
        "actions": actions,
        "cameras": camera_images,
        "timestamp": observation["timestamp"],
    }


class RecordingWorker(BaseProcessWorker):
    def __init__(self, dataset: Dataset, data_manifest: EnvironmentDataManifest, mp_terminate_event: EventClass):
        super().__init__(stop_event=mp_terminate_event, queues_to_cancel=[])
        self.loaded_event = mp.Event()
        #self.start_episode_event = TwoWayEvent()
        #self.save_episode_event = TwoWayEvent()
        #self.discard_episode_event = TwoWayEvent()
        self.dataset_config = dataset
        self.data_manifest = data_manifest
        self.fps = 50
        self._start_event = mp.Event()
        self._start_ack = mp.Event()
        self._save_event = mp.Event()
        self._save_ack = mp.Event()
        self._discard_event = mp.Event()
        self._discard_ack = mp.Event()
        self._task_buf = mp.Array('c', 256)        # shared task string
        self._is_recording = mp.Value('b', False)
        self._episodes_recorded = mp.Value('i', 0)

    def get_state(self) -> dict:
        return {
            "is_recording": self._is_recording.value,
            "episodes_recorded": self._episodes_recorded.value,
            "task": self.get_task(),
        }

    async def start_episode(self, task: str) -> None:
        self._start_ack.clear()
        self._task_buf.get_obj().value = task.encode('utf-8')[:255]
        self._start_event.set()
        await asyncio.to_thread(self._start_ack.wait)

    async def save_episode(self) -> None:
        self._save_ack.clear()
        self._save_event.set()
        await asyncio.to_thread(self._save_ack.wait)

    async def discard_episode(self) -> None:
        self._discard_ack.clear()
        self._discard_event.set()
        await asyncio.to_thread(self._discard_ack.wait)

    async def setup(self) -> None:
        from internal_datasets.lerobot.lerobot_dataset import InternalLeRobotDataset

        self.dataset = InternalLeRobotDataset(Path(self.dataset_config.path))
        features = build_lerobot_dataset_features(self.data_manifest)
        self.recording_mutation = self.dataset.start_recording_mutation(
            fps=RECORDING_FPS,
            features=features,
            robot_type=self.data_manifest.robot.type,
        )
        self.loaded_event.set()

    async def run_loop(self) -> None:
        while not self.should_stop():
            async with run_at_frequency(self.fps):
                await asyncio.gather(
                    self._handle_start_recording(),
                    self._handle_save_episode(),
                    self._handle_discard_episode(),
                )

                if self._is_recording.value:
                    obs = get_observation_from_manifest(self.data_manifest)
                    dataset_observation, actions = format_observation_for_dataset(obs, self.data_manifest)
                    self.recording_mutation.add_frame(dataset_observation, actions, self.get_task())

    async def teardown(self) -> None:
        if self.recording_mutation:
            self.recording_mutation.teardown()

    def get_task(self) -> str:
        return bytes(self._task_buf.get_obj()).rstrip(b"\x00").decode()

    async def _handle_start_recording(self) -> None:
        if self._start_event.is_set():
            #say(f"Start episode {self.state.episodes_recorded + 1}")
            print("Start recording")
            self._is_recording.value = True
            self._start_event.clear()
            self._start_ack.set()

    async def _handle_save_episode(self) -> None:
        if self._save_event.is_set():
            #say(f"Saving episode {self.state.episodes_recorded + 1}")
            self._save_event.clear()
            self.recording_mutation.save_episode()
            self._is_recording.value = False
            self._episodes_recorded.value += 1
            self._save_ack.set()

    async def _handle_discard_episode(self) -> None:
        if self._discard_event.is_set():
            #say("Discard episode")
            self._discard_event.clear()
            self.recording_mutation.discard_buffer()
            self._is_recording.value = False
            self._discard_ack.set()



class EnvironmentIntegration:
    """Responsible for setting up the workers for the robots and cameras in an environment."""
    manifest: EnvironmentDataManifest | None = None

    def __init__(self, environment: EnvironmentWithRelations, robot_client_factory: RobotClientFactory, mp_terminate_event: EventClass):
        self.robot = environment.robots[0]
        self.cameras = environment.cameras
        self.robot_client_factory = robot_client_factory
        self._mp_terminate_event = mp_terminate_event
        self._workers = []

    async def setup_environment(self):
        try:
            follower = await self.robot_client_factory.build(self.robot.robot)
            features = follower.features()

            leader = None
            if isinstance(self.robot.tele_operator, TeleoperatorRobotWithRobot) and self.robot.tele_operator.robot is not None:
                leader = await self.robot_client_factory.build(self.robot.tele_operator.robot)

            teleoperate_worker = TeleoperateWorker(follower, leader, 100, self._mp_terminate_event)
            self._workers.append(teleoperate_worker)

            robot_entry = RobotManifestEntry(
                name=self.robot.robot.name,
                type=self.robot.robot.type,
                features=features,
                state=teleoperate_worker._output_state,
                actions=teleoperate_worker._output_actions,
            )

            camera_entries = []
            for camera in self.cameras:
                worker = CameraWorker(camera, self._mp_terminate_event)
                self._workers.append(worker)
                camera_entries.append(CameraManifestEntry(
                    id=str(camera.id),
                    name=camera.name,
                    width=worker._width,
                    height=worker._height,
                    frame_data=worker._frame_data,
                ))

            for worker in self._workers:
                worker.start()

            for worker in self._workers:
                if hasattr(worker, "loaded_event"):
                    await asyncio.to_thread(worker.loaded_event.wait)

            self.manifest = EnvironmentDataManifest(robot=robot_entry, cameras=camera_entries)
        except Exception:
            for worker in self._workers:
                worker.stop()
            raise

    def teardown(self):
        for worker in self._workers:
            worker.stop()

class RobotControlState(BaseModel):
    task: str | None = None
    model_loaded: bool = False
    dataset_loaded: bool = False
    environment_loaded: bool = False
    is_recording: bool = False
    follower_source: Literal["model", "teleoperation"] | None = None
    episodes_recorded: int = 0

RECORDING_FPS = 30

class RobotControlOrchestrator:
    environment: EnvironmentIntegration | None = None
    recording: RecordingWorker | None = None

    def __init__(self,
        message_queue: asyncio.Queue,
        robot_client_factory: RobotClientFactory,
        mp_terminate_event: EventClass
    ):
        self.state = RobotControlState()
        self._mp_terminate_event = mp_terminate_event
        self.robot_client_factory = robot_client_factory
        self.message_queue = message_queue

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
                    dataset,
                    self.environment.manifest,
                    self._mp_terminate_event
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


    def load_model(self, model: Model, backend: str) -> None:
        """Load model for inference."""

    async def start_recording(self, task: str) -> None:
        """Start recording of specified task."""
        print(f"start recording... recording ready? {self.recording is not None}")
        if self.recording:
            await self.recording.start_episode(task)
            recording_state = self.recording.get_state()
            self.state.is_recording = recording_state["is_recording"]
            print(self.state)
            self._report_state()

    async def save_episode(self) -> None:
        """Save recording."""
        if self.recording:
            await self.recording.save_episode()
            recording_state = self.recording.get_state()
            self.state.is_recording = recording_state["is_recording"]
            self._report_state()

    async def discard_episode(self) -> None:
        """Discard episode."""
        if self.recording:
            await self.recording.discard_episode()
            recording_state = self.recording.get_state()
            self.state.is_recording = recording_state["is_recording"]
            self._report_state()

    async def start_task(self, task: str) -> None:
        """Start task on model."""

    def stop_task(self) -> None:
        """Stop executing actions from model."""

    def set_follower_source(self, follower_source: Literal["model", "teleoperation"] | None) -> None:
        """Sets teleoperation loop to follow either model or teleoperator."""

    def teardown(self) -> None:
        if self.environment:
            self.environment.teardown()
        if self.recording:
            self.recording.stop()

    def get_observation(self) -> dict | None:
        if self.environment and self.environment.manifest:
            obs = get_observation_from_manifest(self.environment.manifest)
            return format_observation_for_reporting(obs, self.environment.manifest)
        return None

    def _report_state(self):
        self.message_queue.put_nowait({
            "event": "state",
            "data": self.state.model_dump(),
        })

    def _report_error(self, component: str, error: BaseException):
        """Report error in application."""
        self.message_queue.put_nowait({
            "event": "error",
            "component": component,
            "data": str(error),
        })
