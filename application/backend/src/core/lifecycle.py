import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger

from core.logging import setup_logging, setup_uvicorn_logging
from services.event_processor import EventProcessor
from settings import get_settings
from utils.serial_robot_tools import RobotConnectionManager
from workers.camera_worker_registry import CameraWorkerRegistry
from workers.robot_worker_registry import RobotWorkerRegistry

from .scheduler import Scheduler


def _warmup_imports() -> None:
    """Eagerly import heavy modules in a background thread.

    After deferring ML/CV imports out of the module-level code paths, this
    function pre-loads them so they are cached in ``sys.modules`` before a
    user request needs them.  Python's import lock ensures that if a request
    handler triggers the same import concurrently, it simply blocks until
    the import finishes — no duplicate work or races.
    """
    import importlib

    modules = [
        # Heavy third-party libraries
        "torch",
        "cv2",
        "lerobot",
        "trossen_arm",
        # Internal modules that pull in the above
        "frame_source",
        "workers.training_worker",
        "workers.camera_worker",
        "workers.robot_control_worker",
        "workers.robots.robot_worker",
        "workers.robots.so101_setup_worker",
        "robots.robot_client_factory",
        "robots.so101.so101",
        "robots.widowxai.trossen_widowx_ai_follower",
        "robots.widowxai.trossen_widowx_ai_leader",
        "internal_datasets.lerobot.lerobot_dataset",
        "utils.calibration",
        "utils.trossen_robot_tools",
        "utils.serial_robot_tools",
    ]
    for mod in modules:
        try:
            importlib.import_module(mod)
            logger.debug("Warmup: imported {}", mod)
        except Exception:
            logger.opt(exception=True).warning("Warmup: failed to import {}", mod)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """FastAPI lifespan context manager"""
    # Startup
    setup_logging()
    setup_uvicorn_logging()

    settings = get_settings()
    app.state.settings = settings

    app.state.camera_registry = CameraWorkerRegistry(
        max_workers=10,
        shutdown_timeout_s=10.0,
    )
    app.state.robot_registry = RobotWorkerRegistry(
        max_workers=10,
        shutdown_timeout_s=10.0,
    )

    logger.info("Starting %s application...", settings.app_name)
    app_scheduler = Scheduler()
    app_scheduler.start_workers()
    app.state.scheduler = app_scheduler
    app.state.event_processor = EventProcessor(app_scheduler.event_queue)
    logger.info("Application startup completed")

    # Initialize RobotHardwareManager
    app.state.robot_manager = RobotConnectionManager()
    await app.state.robot_manager.find_robots()

    # Pre-load heavy ML/CV modules in a background thread so the first
    # user request doesn't pay the import cost.
    warmup_task = asyncio.create_task(asyncio.to_thread(_warmup_imports))

    yield

    # Shutdown
    logger.info("Shutting down %s application...", settings.app_name)

    # Cancel warmup if still running
    if not warmup_task.done():
        warmup_task.cancel()
        try:
            await warmup_task
        except asyncio.CancelledError:
            pass

    camera_registry: CameraWorkerRegistry = app.state.camera_registry
    await camera_registry.shutdown_all()

    robot_registry: RobotWorkerRegistry = app.state.robot_registry
    await robot_registry.shutdown_all()

    # We might want to shutdown the hardware manager too, though releasing workers should handle it.
    # But a global cleanup is safe.
    # Ideally RobotHardwareManager would have a shutdown_all method too.
    # For now, we assume active workers unregistering will trigger releases.

    app_scheduler.shutdown()
    app.state.event_processor.shutdown()
    logger.info("Application shutdown completed")
