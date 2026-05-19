from workers.base import run_at_frequency
from starlette.websockets import WebSocketDisconnect
import asyncio
import multiprocessing as mp
from queue import Empty
from typing import Annotated

from fastapi import APIRouter, Depends, WebSocket
from fastapi.responses import Response
from loguru import logger

from api.dependencies import RobotCalibrationServiceDep, RobotConnectionManagerDep, get_scheduler_ws
from core.scheduler import Scheduler
from robots.robot_client_factory import RobotClientFactory
from schemas import Dataset, Model
from schemas.environment import EnvironmentWithRelations
from workers.robot_control_worker import RobotControlOrchestrator

router = APIRouter(prefix="/api/record")

@router.get("/robot_control/ws", tags=["WebSocket"], summary="Robot Control (WebSocket)", status_code=426)
async def robot_control_websocket_openapi() -> Response:
    """This endpoint requires a WebSocket connection. Use `wss://` to connect."""
    return Response(status_code=426)


async def handle_incoming(websocket: WebSocket, robot_control: RobotControlOrchestrator) -> None:
    """Handle incoming messages for robot control."""
    try:
        while True:
            data = await websocket.receive_json("text")
            payload = data.get("data", {})
            match data["event"]:
                case "load_environment":
                    await robot_control.load_environment(EnvironmentWithRelations.model_validate(payload["environment"]))
                case "load_model":
                    robot_control.load_model(Model.model_validate(payload["model"]), payload["backend"])
                case "load_dataset":
                    await robot_control.load_dataset(Dataset.model_validate(payload["dataset"]))
                case "set_follower_source":
                    robot_control.set_follower_source(payload["follower_source"])
                case "start_recording":
                    await robot_control.start_recording(payload["task"])
                case "save_episode":
                    await robot_control.save_episode()
                case "discard_episode":
                    await robot_control.discard_episode()
                #case "start_task":
                #    robot_control.start_task(payload["task"])
                #case "stop_task":
                #    robot_control.stop()
                #    break
            await asyncio.sleep(0.005)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"Incoming task stopped: {e}")
        logger.info("Except: disconnected!")

async def handle_outgoing(websocket: WebSocket, queue: asyncio.Queue) -> None:
    """Handle outgoing messages for robot control."""
    try:
        while True:
            try:
              data = await queue.get()
              if data is None:
                  break
              await websocket.send_json(data)
            except Empty:
                await asyncio.sleep(0.05)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"Outgoing task stopped: {e}")

async def observation_update_loop(websocket: WebSocket, robot_control: RobotControlOrchestrator) -> None:
    """Handle outgoing messages for robot control."""
    try:
        while True:
            async with run_at_frequency(30):
                observation = robot_control.get_observation()
                if observation:
                    await websocket.send_json(
                        {
                            "event": "observations",
                            "data": observation,
                        })
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"Observation update loop stopped: {e}")
        raise

@router.websocket("/robot_control/ws")
async def robot_control_websocket(
    websocket: WebSocket,
    robot_manager: RobotConnectionManagerDep,
    calibration_service: RobotCalibrationServiceDep,
    scheduler: Annotated[Scheduler, Depends(get_scheduler_ws)],
):
    await websocket.accept()
    queue = asyncio.Queue()
    robot_control = RobotControlOrchestrator(
        message_queue=queue,
        robot_client_factory=RobotClientFactory(
            robot_manager=robot_manager,
            calibration_service=calibration_service
        ),
        mp_terminate_event=scheduler.mp_stop_event,
    )
    try:
        incoming_task = asyncio.create_task(handle_incoming(websocket, robot_control))
        outgoing_task = asyncio.create_task(handle_outgoing(websocket, queue))
        observation_update_task = asyncio.create_task(observation_update_loop(websocket, robot_control))

        _, pending = await asyncio.wait(
            {incoming_task, outgoing_task, observation_update_task},
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
            task.cancel()
    finally:
        robot_control.teardown()
#
# TODO: Implement
#@router.websocket("/robot_control/ws")
#async def robot_control_websocket(
#    websocket: WebSocket,
#    robot_manager: RobotConnectionManagerDep,
#    calibration_service: RobotCalibrationServiceDep,
#    scheduler: Annotated[Scheduler, Depends(get_scheduler_ws)],
#    worker_pool: WorkerPoolDep,
#) -> None:
#    """Robot control websocket."""
#    await websocket.accept()
#    queue: mp.Queue = mp.Queue()
#    process = RobotControlWorker(
#        stop_event=scheduler.mp_stop_event,
#        robot_client_factory=RobotClientFactory(
#            robot_manager=robot_manager,
#            calibration_service=calibration_service,
#        ),
#        queue=queue,
#        model_worker_registry=model_registry,
#        teleoperate_worker_registry=teleoperate_registry
#    )
#    process.start()
#
#    incoming_task = asyncio.create_task(handle_incoming(websocket, process))
#    outgoing_task = asyncio.create_task(handle_outgoing(websocket, queue))
#
#    _, pending = await asyncio.wait(
#        {incoming_task, outgoing_task},
#        return_when=asyncio.FIRST_COMPLETED,
#    )
#
#    for task in pending:
#        task.cancel()
#
#    if process is not None:
#        process.disconnect()
#        process.join(10)
#
#    queue.close()
#    logger.info("websocket handling done...")
