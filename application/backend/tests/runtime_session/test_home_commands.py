"""Exercise session routing with a fake follower and in-process home RPC responders."""

import time
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest

from api.runtime_ws import _request
from exceptions import RuntimeSessionCommandError
from runtime_session.home_position import HOME_EVENTS, HomeReturn, home_key
from runtime_session.store import Sample
from schemas.home_position import (
    CancelHomeReturnCommand,
    CaptureHomePositionCommand,
    HomeCapability,
    HomeJointLimits,
    HomeMotion,
    ReturnHomeCommand,
)
from schemas.runtime_session import SessionDataset, TeleopControlConfig
from workers import dataset_worker

from .fixtures import make_environment


async def prepare(new_session):
    session = await new_session()
    await session.load_environment(make_environment(cameras=0))
    capability = HomeCapability(
        robot_id=session.environment.robot_id,
        fingerprint="c" * 64,
        unit="degrees",
        joints={
            "shoulder_pan.pos": HomeJointLimits(
                minimum=-10, maximum=10, velocity=5, acceleration=10, tolerance=0.1, tracking_error=1
            )
        },
    )
    session.environment.home_capability = capability
    engine = HomeReturn(
        capability, ("shoulder_pan.pos", "gripper.pos"), lambda motion: session.bus.publish(HOME_EVENTS, motion)
    )
    now = time.monotonic()
    for step in range(41):
        at = now - (40 - step) / 100
        engine.observe(Sample(np.array([1.0, 2.0]), engine.names, at), at)
    session.bus.serve(
        home_key(capability.robot_id, "capture"),
        CaptureHomePositionCommand,
        lambda _request: engine.capture(time.monotonic()),
    )
    session.bus.serve(
        home_key(capability.robot_id, "return"),
        ReturnHomeCommand,
        lambda request: engine.start(request.operation_id, request.home, time.monotonic()),
    )
    session.bus.serve(
        home_key(capability.robot_id, "cancel"),
        CancelHomeReturnCommand,
        lambda request: engine.cancel(request.operation_id),
    )
    return session, engine


async def test_capture_return_takeover_and_stop(new_session):
    session, engine = await prepare(new_session)
    pose = await session.capture_home(
        CaptureHomePositionCommand(request_id="capture", robot_id=session.environment.robot_id)
    )
    assert pose.positions == {"shoulder_pan.pos": 1}
    await session.set_control(TeleopControlConfig())
    pose = pose.model_copy(update={"positions": {"shoulder_pan.pos": 3}})
    result = await session.return_home(
        ReturnHomeCommand(request_id="start", operation_id=uuid4(), home=pose, path_confirmed=True)
    )
    assert result.state == "moving"
    assert session.state().control is None  # A takeover never resumes the previous control.
    ack = await _request(
        session.bus, '{"command":"start_episode","request_id":"episode","task":"Pick"}', session=session
    )
    assert not ack.ok and ack.error.error_code == "home_return_active"
    with pytest.raises(RuntimeSessionCommandError, match="Stop the home"):
        await session.set_control(TeleopControlConfig())
    stopped = await session.cancel_home(CancelHomeReturnCommand(request_id="stop", operation_id=result.operation_id))
    assert stopped.state == "cancelled" and not engine.moving
    session.require_home_idle()


async def test_home_rejects_open_episode_without_discarding_it(new_session):
    session, engine = await prepare(new_session)
    pose = engine.capture(time.monotonic())
    session._dataset.value = SimpleNamespace(
        status=dataset_worker.RecordingStatus(is_recording=True, episodes_recorded=0),
        state=lambda: SessionDataset(
            dataset_id=uuid4(), name="Test", fps=30, fps_fixed=True, is_recording=True, episodes_recorded=0
        ),
    )
    session.bus.serve(dataset_worker.STATUS, dataset_worker.Empty, lambda _request: session._dataset.value.status)
    with pytest.raises(RuntimeSessionCommandError, match="Save or discard"):
        await session.return_home(
            ReturnHomeCommand(request_id="start", operation_id=uuid4(), home=pose, path_confirmed=True)
        )
    assert session._dataset.value.status.is_recording
    assert not engine.moving


async def test_old_progress_does_not_overwrite_a_stopped_return(new_session):
    session, _ = await prepare(new_session)
    operation = uuid4()
    session._home_operation_id = operation
    stopped = HomeMotion(operation_id=operation, robot_id=session.environment.robot_id, state="cancelled", sequence=5)
    session._on_home_motion(stopped)
    session._on_home_motion(stopped.model_copy(update={"state": "moving", "sequence": 4}))
    assert session.state().home_motion.state == "cancelled"
    session._on_home_motion(stopped.model_copy(update={"operation_id": uuid4(), "state": "moving", "sequence": 20}))
    assert session.state().home_motion.operation_id == operation
