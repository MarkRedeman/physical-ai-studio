from src.workers.base import BaseProcessWorker
import base64
import cv2
from internal_datasets.lerobot.lerobot_dataset import InternalLeRobotDataset
import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
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

def get_observation_from_manifest(manifest: EnvironmentDataManifest, timestamp: float = 0):
    raw_actions = list(manifest.robot.actions.get_obj())
    actions = {i: raw_actions[k] for k,i in enumerate(manifest.robot.features)}
    raw_state = list(manifest.robot.state.get_obj())
    state = {i: raw_state[k] for k,i in enumerate(manifest.robot.features)}

    camera_images = {}
    for camera in manifest.cameras:
        frame = CameraWorker.frame_from_buffer(camera.frame_data.get_obj(), camera.width, camera.height)
        _, imagebytes = cv2.imencode(".jpg", frame)
        camera_images[camera.id] = base64.b64encode(imagebytes).decode()
        #TODO  dont convert to jpg just yet, jsut for reporting

    return {
        "state": state,
        "actions": actions,
        "cameras": camera_images,
        "timestamp": timestamp,
    }


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
    dataset: InternalLeRobotDataset | None = None

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

    def load_dataset(self, dataset: Dataset) -> None:
        """Load dataset and setup recording."""
        if self.environment and self.environment.manifest:
            try:
                self.dataset = InternalLeRobotDataset(Path(dataset.path))
                features = build_lerobot_dataset_features(self.environment.manifest)
                self.recording_mutation = self.dataset.start_recording_mutation(
                    fps=RECORDING_FPS,
                    features=features,
                    robot_type=self.environment.manifest.robot.type,
                )
                self.state.dataset_loaded = True
            except Exception as e:
                self._report_error("dataset", e)
            finally:
                self._report_state()
        else:
            self._report_error("dataset", ValueError("Cannot load dataset without environment."))


    def load_model(self, model: Model) -> None:
        """Load model for inference."""

    def start_recording(self, task: str) -> None:
        """Start recording of specified task."""

    def start_task(self, task: str) -> None:
        """Start task on model."""

    def stop_task(self) -> None:
        """Stop executing actions from model."""

    def set_follower_source(self, follower_source: Literal["model", "teleoperation"] | None) -> None:
        """Sets teleoperation loop to follow either model or teleoperator."""

    def teardown(self) -> None:
        if self.environment:
            self.environment.teardown()

    def get_observation(self) -> dict | None:
        if self.environment and self.environment.manifest:
            return get_observation_from_manifest(self.environment.manifest)
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
