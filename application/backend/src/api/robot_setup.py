"""WebSocket endpoint for the robot setup wizard.

Dispatches to the appropriate setup worker based on robot type:
  - SO101 types → SO101SetupWorker (serial_number required)
  - Trossen WidowX AI types → TrossenSetupWorker (connection_string required)
"""

from typing import Annotated

from fastapi import APIRouter, Depends, WebSocket, status
from loguru import logger

from api.dependencies import get_project_id
from schemas.robot import RobotType
from workers.robots.so101_setup_worker import SO101SetupWorker
from workers.robots.trossen_setup_worker import TrossenSetupWorker
from workers.transport.websocket_transport import WebSocketTransport

router = APIRouter(prefix="/api/projects/{project_id}/robots", tags=["Robot Setup"])

SO101_TYPES = {RobotType.SO101_FOLLOWER, RobotType.SO101_LEADER}
TROSSEN_TYPES = {RobotType.TROSSEN_WIDOWXAI_FOLLOWER, RobotType.TROSSEN_WIDOWXAI_LEADER}
SUPPORTED_TYPES = SO101_TYPES | TROSSEN_TYPES


@router.websocket("/setup/ws")
async def robot_setup_websocket(
    _project_id: Annotated[str, Depends(get_project_id)],
    websocket: WebSocket,
    robot_type: str,
    serial_number: str = "",
    connection_string: str = "",
) -> None:
    """Establish a WebSocket connection for the robot setup wizard.

    Query parameters:
        robot_type: One of the supported RobotType values
        serial_number: USB serial number (required for SO101 types)
        connection_string: IP address (required for Trossen types)
    """
    # Validate robot type
    if robot_type not in SUPPORTED_TYPES:
        await websocket.accept()
        await websocket.send_json(
            {
                "event": "error",
                "message": f"Unsupported robot type for setup: {robot_type}",
            }
        )
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    # Validate connection parameters per robot type
    if robot_type in SO101_TYPES and not serial_number:
        await websocket.accept()
        await websocket.send_json(
            {
                "event": "error",
                "message": "serial_number is required for SO101 robots",
            }
        )
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    if robot_type in TROSSEN_TYPES and not connection_string:
        await websocket.accept()
        await websocket.send_json(
            {
                "event": "error",
                "message": "connection_string is required for Trossen robots",
            }
        )
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await websocket.accept()

    try:
        transport = WebSocketTransport(websocket)

        if robot_type in SO101_TYPES:
            worker = SO101SetupWorker(
                transport=transport,
                robot_type=robot_type,
                serial_number=serial_number,
            )
        else:
            worker = TrossenSetupWorker(
                transport=transport,
                robot_type=robot_type,
                connection_string=connection_string,
            )

        await worker.run()

    except Exception as e:
        logger.exception(f"Unexpected error in robot setup websocket: {e}")
        try:
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        except Exception as close_err:
            logger.error(f"Could not close websocket after error: {close_err}")
