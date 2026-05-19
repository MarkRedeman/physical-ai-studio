from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger

from core.logging import setup_logging, setup_uvicorn_logging
from services.event_processor import EventProcessor
from settings import get_settings
from utils.multiprocessing import ensure_spawn_start_method
from utils.serial_robot_tools import RobotConnectionManager

from .scheduler import Scheduler


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """FastAPI lifespan context manager"""
    # Startup
    setup_logging()
    setup_uvicorn_logging()

    settings = get_settings()
    app.state.settings = settings

    logger.info("Starting %s application...", settings.app_name)
    ensure_spawn_start_method()
    app_scheduler = Scheduler()
    app_scheduler.start_workers()

    app.state.scheduler = app_scheduler
    app.state.event_processor = EventProcessor(app_scheduler.event_queue)
    logger.info("Application startup completed")

    # Initialize RobotHardwareManager
    app.state.robot_manager = RobotConnectionManager()
    await app.state.robot_manager.find_robots()

    yield

    # Shutdown
    logger.info("Shutting down %s application...", settings.app_name)

    # We might want to shutdown the hardware manager too, though releasing workers should handle it.
    # But a global cleanup is safe.
    # Ideally RobotHardwareManager would have a shutdown_all method too.
    # For now, we assume active workers unregistering will trigger releases.

    app_scheduler.shutdown()
    app.state.event_processor.shutdown()
    logger.info("Application shutdown completed")
