"""Form-scoped direct robot observations; no persistence, actions, or calibration."""

import math
import time
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID

import anyio
from fastapi import APIRouter, WebSocket
from fastapi.exceptions import RequestValidationError
from fastapi.responses import Response
from fastapi.websockets import WebSocketDisconnect
from loguru import logger
from physicalai.robot import Robot, RobotTransportError
from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from api.dependencies import ProjectServiceDep, RobotCatalogServiceDep, RobotClientFactoryDep
from robots.draft_validation import validate_draft_payload
from robots.shared_robot_errors import translate_robot_error
from runtime.features import observation_to_dict

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

    from robots.robot_client_factory import RobotClientFactory
    from services.robot_catalog_service import RobotCatalogService

router = APIRouter(prefix="/api/projects/{project_id}/robots", tags=["Robot Setup"])
FPS = 15
MAX_STATE_AGE_S = 2.0


class DraftRobot(BaseModel):
    """Open payload contract supports plugins installed after UI compilation."""

    id: UUID
    name: str
    type: str
    payload: dict[str, Any]


class ConnectCommand(BaseModel):
    command: Literal["connect"]
    request_id: str = Field(min_length=1, max_length=128)
    robot: DraftRobot


class DisconnectCommand(BaseModel):
    command: Literal["disconnect"]
    request_id: str = Field(min_length=1, max_length=128)


COMMAND = TypeAdapter(ConnectCommand | DisconnectCommand)


class DirectRobotSession:
    """Serial I/O and hardware claims outlive async cancellation until cleanup."""

    def __init__(self) -> None:
        self.driver: Robot | None = None
        self.claim: AbstractContextManager[None] | None = None

    def connect(self, driver: Robot) -> None:
        """Claim before connect, retaining partial connections for cleanup."""
        # Lazy import keeps older installations usable; this feature fails closed
        # until the public device-claim helper is installed, never using private locks.
        from physicalai.robot.transport import claim_devices

        claim = claim_devices(driver.device_ids)
        claim.__enter__()
        self.claim = claim
        self.driver = driver
        driver.connect()

    def observe(self) -> dict[str, float]:
        """Reject stale, malformed, and nonfinite measurements."""
        if self.driver is None:
            raise RuntimeError("No live robot connection")
        observation = self.driver.get_observation()
        age = time.monotonic() - observation.timestamp
        if not math.isfinite(age) or age < -0.1 or age > MAX_STATE_AGE_S:
            raise RobotTransportError("Robot stopped providing fresh observations", phase="observation_stale")
        data = observation_to_dict(self.driver.joint_names, observation)
        if not data or not all(math.isfinite(value) for value in data.values()):
            raise RobotTransportError("Robot returned invalid joint readings", phase="observation_invalid")
        return data

    def disconnect(self) -> None:
        """Release claims only after the driver's disconnect completes successfully."""
        if self.driver is not None:
            self.driver.disconnect()
            self.driver = None
        if self.claim is not None:
            self.claim.__exit__(None, None, None)
            self.claim = None


# Failed driver cleanup is not proof of release. Retain its claim until restart,
# rather than let a new form/runtime open possibly still-active hardware.
_failed_cleanup: list[DirectRobotSession] = []


def error_event(exc: Exception, request_id: str | None) -> dict[str, Any]:
    """Reuse structured runtime errors and retain validation field locations."""
    if isinstance(exc, (ValidationError, RequestValidationError)):
        fields = [{"location": ".".join(map(str, error["loc"])), "message": error["msg"]} for error in exc.errors()]
        return {
            "event": "error",
            "request_id": request_id,
            "error_code": "invalid_config",
            "message": "Invalid robot configuration",
            "errors": fields,
        }
    if isinstance(exc, ImportError):
        return {
            "event": "error",
            "request_id": request_id,
            "error_code": "runtime_upgrade_required",
            "message": "Update PhysicalAI to a version with public device claims before using Live connect.",
        }
    translated = translate_robot_error(exc)
    fallback_code = "serial_permission_denied" if isinstance(exc, PermissionError) else "robot_connection_failed"
    return {
        "event": "error",
        "request_id": request_id,
        "error_code": getattr(translated, "error_code", fallback_code),
        "message": getattr(translated, "message", str(exc) or "Failed to connect to the robot"),
    }


class LiveRobotWorker:
    """One serialized command/read loop, with a separate disconnect watcher."""

    def __init__(self, websocket: WebSocket, catalog: "RobotCatalogService", factory: "RobotClientFactory") -> None:
        self.websocket = websocket
        self.catalog = catalog
        self.factory = factory
        self.session = DirectRobotSession()
        self.request_id: str | None = None

    async def handle_command(self, data: Any) -> None:
        """Disconnect explicitly before accepting a new configuration."""
        command = COMMAND.validate_python(data)
        self.request_id = command.request_id
        if isinstance(command, DisconnectCommand):
            await anyio.to_thread.run_sync(self.session.disconnect)
            await self.websocket.send_json({"event": "disconnected", "request_id": self.request_id})
            return
        if self.session.driver is not None:
            raise RuntimeError("Disconnect the previous configuration before connecting")
        definition = self.catalog.get_definition(command.robot.type)
        payload = validate_draft_payload(definition, command.robot.payload)
        robot = self.catalog.registry.get_robot_adapter().validate_python(
            {**command.robot.model_dump(), "payload": payload}
        )
        if robot.type in {"SO101_Follower", "SO101_Leader"} and not payload.get("calibration"):
            raise ValueError("Complete SO101 setup or upload calibration before Live connect")
        driver, _definition = await self.factory.build_robot_driver(robot, self.factory)
        await self.websocket.send_json({"event": "connecting", "request_id": self.request_id})
        await anyio.to_thread.run_sync(self.session.connect, driver)

    async def broadcast(self) -> None:
        """Emit a verified observation rather than websocket-open readiness."""
        if self.session.driver is not None:
            observation = await anyio.to_thread.run_sync(self.session.observe)
            await self.websocket.send_json({"event": "observation", "request_id": self.request_id, "data": observation})

    async def run(self) -> None:
        """Watch browser disconnects while shielded hardware calls finish."""
        send, receive = anyio.create_memory_object_stream[Any](8)
        async with anyio.create_task_group() as group:

            async def receive_commands() -> None:
                try:
                    while True:
                        data = await self.websocket.receive_json()
                        try:
                            send.send_nowait(data)
                        except anyio.WouldBlock:
                            await self.websocket.close(code=1008, reason="Too many setup commands")
                            break
                except WebSocketDisconnect:
                    pass
                finally:
                    group.cancel_scope.cancel()

            group.start_soon(receive_commands)
            while True:
                data = None
                with anyio.move_on_after(1 / FPS):
                    data = await receive.receive()
                try:
                    if data is not None:
                        await self.handle_command(data)
                    await self.broadcast()
                except Exception as exc:
                    logger.warning("Live robot setup error: {}", exc)
                    await self.websocket.send_json(error_event(exc, self.request_id))
                    try:
                        await anyio.to_thread.run_sync(self.session.disconnect)
                    except Exception as cleanup_exc:
                        await self.websocket.send_json(
                            {
                                "event": "error",
                                "request_id": self.request_id,
                                "error_code": "robot_disconnect_failed",
                                "message": (
                                    f"Could not safely disconnect: {cleanup_exc}. Restart Studio before reconnecting."
                                ),
                            }
                        )
                        group.cancel_scope.cancel()
                        break


@router.get("/draft/observations/ws", tags=["WebSocket"], status_code=426)
async def live_robot_websocket_openapi(project_id: UUID) -> Response:  # noqa: ARG001
    """Requires a WebSocket connection for a validated unsaved robot."""
    return Response(status_code=426)


@router.websocket("/draft/observations/ws")
async def live_robot_websocket(
    websocket: WebSocket,
    project_id: UUID,
    project_service: ProjectServiceDep,
    catalog_service: RobotCatalogServiceDep,
    robot_client_factory: RobotClientFactoryDep,
) -> None:
    """Connect on command; acknowledge disconnect only after releasing hardware."""
    await websocket.accept()
    worker = LiveRobotWorker(websocket, catalog_service, robot_client_factory)
    try:
        await project_service.get_project_by_id(project_id)
        await worker.run()
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.exception("Live robot websocket failed")
        try:
            await websocket.send_json(error_event(exc, worker.request_id))
        except Exception:
            logger.debug("Browser closed before setup error could be sent")
    finally:
        with anyio.CancelScope(shield=True):
            try:
                await anyio.to_thread.run_sync(worker.session.disconnect)
            except Exception:
                logger.exception("Live robot cleanup failed; retaining device claim")
                _failed_cleanup.append(worker.session)
            try:
                await websocket.close()
            except (RuntimeError, WebSocketDisconnect):
                logger.debug("Live robot websocket already closed")
