import asyncio
import json
import time
from contextlib import asynccontextmanager
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, Query, WebSocket
from fastapi.responses import Response
from fastapi.websockets import WebSocketDisconnect
from frame_source import FrameSourceFactory

from api.dependencies import SchedulerDep
from schemas.camera import SupportedCameraFormat
from schemas.project_camera import Camera as ProjectCamera
from schemas.project_camera import CameraAdapter
from workers.camera_worker import CameraWorker

router = APIRouter(prefix="/api/cameras", tags=["Cameras"])


@router.get("/supported_formats/{driver}")
async def get_supported_formats(
    driver: str,
    fingerprint: str,
) -> list[SupportedCameraFormat]:
    """Returns the supported camera resolution and fps associated to the camera"""
    camera = FrameSourceFactory.create(driver if driver != "usb_camera" else "webcam", source=fingerprint)
    formats = camera.get_supported_formats()

    if formats is None:
        return []

    return [
        SupportedCameraFormat(width=format["width"], height=format["height"], fps=format["fps"]) for format in formats
    ]


def get_camera_from_query(websocket: WebSocket) -> ProjectCamera:
    """Parse camera from query parameters."""
    camera_param = websocket.query_params.get("camera")
    if not camera_param:
        raise ValueError("Missing 'camera' query parameter")

    try:
        camera_data = json.loads(camera_param)
        return CameraAdapter.validate_python(camera_data)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in camera parameter: {e}")
    except Exception as e:
        raise ValueError(f"Invalid camera configuration: {e}")


@router.get("/ws", tags=["WebSocket"], summary="Camera streaming (WebSocket)", status_code=426)
async def camera_websocket_openapi(
    camera: Annotated[str | None, Query(description="JSON-serialized ProjectCamera configuration")] = None,  # noqa: ARG001
) -> Response:
    """This endpoint requires a WebSocket connection. Use `wss://` to connect."""
    return Response(status_code=426)


@asynccontextmanager
async def run_at_target_fps(target_dt):
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    sleep_time = target_dt - elapsed
    if sleep_time > 0:
        await asyncio.sleep(sleep_time)


@router.websocket("/ws")
async def camera_websocket(
    websocket: WebSocket,
    scheduler: SchedulerDep,
    camera: Annotated[ProjectCamera, Depends(get_camera_from_query)],
) -> None:
    """
    WebSocket endpoint for camera streaming.

    Query Parameters:
        camera: JSON serialized ProjectCamera

    Protocol:
        Client sends JSON messages:
            {"event": "disconnect"} - Request graceful disconnect
            {"event": "ping"} - Keep-alive check

        Server sends JSON-encoded messages with status updates:
            {"event": "status", "state": "running", ...}
    """
    await websocket.accept()

    worker = None
    try:
        worker = CameraWorker(camera, scheduler.mp_stop_event)
        worker.start()
        while True:
            async with run_at_target_fps(1 / camera.payload.fps):
                if not worker.frame_queue.empty():
                    message = worker.frame_queue.get_nowait()
                    await websocket.send_bytes(message)
    except WebSocketDisconnect:
        pass
    finally:
        if worker:
            worker.stop()
