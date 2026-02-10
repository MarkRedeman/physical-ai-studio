from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends
from lerobot.motors import MotorCalibration

from api.dependencies import (
    RobotCalibrationServiceDep,
    get_calibration_id,
    get_project_id,
    get_robot_id,
    get_robot_service,
)
from schemas.calibration import Calibration
from services import RobotService

router = APIRouter(prefix="/api/projects/{project_id}/robots", tags=["Robot Calibration"])


@router.post("/{robot_id}/calibrations")
async def save_project_robot_calibration(
    _project_id: Annotated[UUID, Depends(get_project_id)],
    robot_id: Annotated[UUID, Depends(get_robot_id)],
    robot_calibration_service: RobotCalibrationServiceDep,
    calibration_data: Calibration,
) -> Calibration:
    """Save calibration for robot."""
    return await robot_calibration_service.save_calibration(robot_id, calibration_data)


@router.get("/{robot_id}/calibrations")
async def get_project_robot_calibrations(
    project_id: Annotated[UUID, Depends(get_project_id)],
    robot_id: Annotated[UUID, Depends(get_robot_id)],
    robot_service: Annotated[RobotService, Depends(get_robot_service)],
    robot_calibration_service: RobotCalibrationServiceDep,
) -> list[Calibration]:
    """Get available calibrations for robot."""
    robot = await robot_service.get_robot_by_id(project_id, robot_id)

    return await robot_calibration_service.get_robot_calibration(robot)


@router.get("/{robot_id}/calibrations/motor")
async def get_project_robot_motor_calibration(
    project_id: Annotated[UUID, Depends(get_project_id)],
    robot_id: Annotated[UUID, Depends(get_robot_id)],
    robot_service: Annotated[RobotService, Depends(get_robot_service)],
    robot_calibration_service: RobotCalibrationServiceDep,
) -> dict[str, MotorCalibration]:
    """Get available calibrations for robot."""
    robot = await robot_service.get_robot_by_id(project_id, robot_id)

    return await robot_calibration_service.get_robot_motor_calibration(robot)

@router.put("/{robot_id}/calibrations/motor")
async def set_project_robot_motor_calibration(
    project_id: Annotated[UUID, Depends(get_project_id)],
    robot_id: Annotated[UUID, Depends(get_robot_id)],
    robot_service: Annotated[RobotService, Depends(get_robot_service)],
    robot_calibration_service: RobotCalibrationServiceDep,
) -> None:
    """Store the active calibration to the robot's motor."""
    robot = await robot_service.get_robot_by_id(project_id, robot_id)
    if (robot.active_calibration_id is None):
        return None

    active_calibration = await robot_calibration_service.get_calibration(robot.active_calibration_id)

    return await robot_calibration_service.set_robot_motor_calibration(
        robot,
        active_calibration.to_motor_calibrations()
        )

@router.get("/{robot_id}/calibrations/{calibration_id}")
async def get_project_robot_calibration(
    _project_id: Annotated[UUID, Depends(get_project_id)],
    _robot_id: Annotated[UUID, Depends(get_robot_id)],
    calibration_id: Annotated[UUID, Depends(get_calibration_id)],
    robot_calibration_service: RobotCalibrationServiceDep,
) -> Calibration:
    """Get robot calibration."""
    return await robot_calibration_service.get_calibration(calibration_id)
