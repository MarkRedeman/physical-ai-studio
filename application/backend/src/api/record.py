import asyncio
import multiprocessing as mp
from queue import Empty

from fastapi import APIRouter, WebSocket
from fastapi.responses import Response
from loguru import logger

from api.dependencies import ModelRegistryDep, RecordingLockedCamerasDep, RobotClientFactoryDep, SchedulerDep
from schemas import Dataset, InferenceDevice, Model
from schemas.environment import EnvironmentWithRelations
from workers.robot_control_worker import RobotControlWorker

router = APIRouter(prefix="/api/record")


@router.get("/robot_control/ws", tags=["WebSocket"], summary="Robot Control (WebSocket)", status_code=426)
async def robot_control_websocket_openapi() -> Response:
    """This endpoint requires a WebSocket connection. Use `wss://` to connect."""
    return Response(status_code=426)


async def handle_incoming(
    websocket: WebSocket,
    process: RobotControlWorker,
    locked_camera_fingerprints: set[str],
) -> None:
    """Handle incoming messages for robot control."""
    try:
        while True:
            data = await websocket.receive_json("text")
            payload = data.get("data", {})
            match data["event"]:
                case "load_environment":
                    environment = EnvironmentWithRelations.model_validate(payload["environment"])
                    locked_camera_fingerprints.clear()
                    locked_camera_fingerprints.update(camera.fingerprint for camera in environment.cameras)
                    process.load_environment(environment)
                case "load_model":
                    process.load_model(
                        Model.model_validate(payload["model"]),
                        InferenceDevice.model_validate(payload["inference_device"]),
                    )
                case "load_dataset":
                    process.load_dataset(Dataset.model_validate(payload["dataset"]))
                case "set_follower_source":
                    process.set_follower_source(payload["follower_source"])
                case "start_recording":
                    process.start_recording(payload["task"])
                case "save_episode":
                    process.save_episode()
                case "discard_episode":
                    process.discard_episode()
                case "start_task":
                    process.start_task(payload["task"])
                case "stop_task":
                    process.stop()
                case "disconnect":
                    process.disconnect()
                    break
    except Exception as e:
        logger.error(f"Incoming task stopped: {e}")
        logger.info("Except: disconnected!")


CAMERA_ID_FRAME_LENGTH = 36  # fixed width of a dashed UUID string


def camera_frame_payload(camera_id: str, jpeg: bytes) -> bytes:
    """Encode a camera frame as a multiplexed binary WebSocket payload.

    Payload layout: ``[36-byte camera id][JPEG bytes]``.

    Camera ids are always dashed UUIDs, so the header is a fixed width and
    carries no length prefix. A client slices the first ``CAMERA_ID_FRAME_LENGTH``
    bytes off the shared binary stream before decoding the JPEG.
    """
    cam_id = camera_id.encode()
    if len(cam_id) != CAMERA_ID_FRAME_LENGTH:
        msg = f"Camera id must be a fixed {CAMERA_ID_FRAME_LENGTH}-char UUID, got {len(cam_id)}-char {camera_id!r}"
        raise ValueError(msg)
    return cam_id + jpeg


async def handle_outgoing(websocket: WebSocket, queue: mp.Queue) -> None:
    """Handle outgoing messages for robot control."""
    try:
        while True:
            try:
                loop = asyncio.get_running_loop()

                message = await loop.run_in_executor(None, queue.get)
                # Camera frames travel as binary WebSocket frames rather than
                # base64 embedded in JSON. Separate them out if present.
                cameras = None
                if isinstance(message.get("data"), dict):
                    cameras = message["data"].pop("cameras", None)
                # Send control data (state/actions/timestamp) as JSON.
                await websocket.send_json(message)
                # WebSocket binary frames (multiplexed by a per-camera header).
                if cameras:
                    for camera_id, jpeg in cameras.items():
                        await websocket.send_bytes(camera_frame_payload(camera_id, jpeg))
            except Empty:
                await asyncio.sleep(0.05)
    except Exception as e:
        logger.error(f"Outgoing camera task stopped: {e}")


@router.websocket("/robot_control/ws")
async def robot_control_websocket(
    websocket: WebSocket,
    robot_client_factory: RobotClientFactoryDep,
    scheduler: SchedulerDep,
    model_registry: ModelRegistryDep,
    locked_camera_fingerprints: RecordingLockedCamerasDep,
) -> None:
    """Robot control websocket."""
    await websocket.accept()
    queue: mp.Queue = mp.Queue()
    process = RobotControlWorker(
        stop_event=scheduler.mp_stop_event,
        robot_client_factory=robot_client_factory,
        queue=queue,
        model_worker_registry=model_registry,
    )
    process.start()

    incoming_task = asyncio.create_task(handle_incoming(websocket, process, locked_camera_fingerprints))
    outgoing_task = asyncio.create_task(handle_outgoing(websocket, queue))

    try:
        _, pending = await asyncio.wait(
            {incoming_task, outgoing_task},
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
            task.cancel()

        if process is not None:
            process.disconnect()
            process.join(10)
    finally:
        queue.close()
        # NOTE: this clears ALL locks, not just this session's. Safe today
        # because only one recording websocket runs at a time. If concurrent
        # sessions are added, scope locks per session instead.
        locked_camera_fingerprints.clear()
    logger.info("websocket handling done...")
