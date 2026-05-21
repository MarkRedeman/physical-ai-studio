# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import asyncio
import multiprocessing as mp
from unittest.mock import AsyncMock, MagicMock

import pytest

from robots.robot_client import RobotClient
from workers.teleoperate_worker import ActionWriteState, TeleoperateWorker

FEATURES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


def _make_robot_client():
    client = MagicMock(spec=RobotClient)
    client.features.return_value = FEATURES
    client.home_position = [0.0] * len(FEATURES)
    client.connect = AsyncMock()
    client.disconnect = AsyncMock()
    client.read_state.return_value = {"state": dict.fromkeys(FEATURES, 0.0)}
    client.set_joints_state.return_value = {"event": "joints_state_was_set"}
    return client


def _make_worker(follower=None, leader=None, frequency=100.0):
    follower = follower or _make_robot_client()
    return TeleoperateWorker(follower, leader, frequency, mp.Event())


def _run_one_iteration(worker):
    """Connect, execute one loop body, then stop."""
    original_return = worker.follower.read_state.return_value
    call_count = [0]

    def stop_after_first():
        call_count[0] += 1
        if call_count[0] >= 1:
            worker._stop_event.set()
        return original_return

    worker.follower.read_state.side_effect = stop_after_first
    asyncio.run(worker.run_loop())
    worker.follower.read_state.side_effect = None


class TestInit:
    def test_action_source_initialized_to_none(self):
        worker = _make_worker()
        assert worker.get_action_source() == ActionWriteState.NONE

    def test_output_actions_initialized_to_home_position(self):
        follower = _make_robot_client()
        follower.home_position = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        worker = _make_worker(follower=follower)
        assert worker.get_actions() == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    def test_output_state_initialized_to_zeros(self):
        worker = _make_worker()
        assert worker.get_state() == pytest.approx([0.0] * len(FEATURES))


class TestSharedMemory:
    def test_set_and_get_state(self):
        worker = _make_worker()
        worker._set_state([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        assert worker.get_state() == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    def test_set_and_get_actions(self):
        worker = _make_worker()
        worker._set_actions([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        assert worker.get_actions() == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    def test_set_and_get_action_source(self):
        worker = _make_worker()
        worker.set_action_source(ActionWriteState.FROM_LEADER)
        assert worker.get_action_source() == ActionWriteState.FROM_LEADER

    def test_action_source_roundtrip_from_actions(self):
        worker = _make_worker()
        worker.set_action_source(ActionWriteState.FROM_ACTIONS)
        assert worker.get_action_source() == ActionWriteState.FROM_ACTIONS


class TestRunLoop:
    def test_loaded_event_set_after_connect(self):
        worker = _make_worker()
        worker._stop_event.set()  # stop immediately after robots connect
        asyncio.run(worker.run_loop())
        assert worker.loaded_event.is_set()

    def test_follower_connected_and_disconnected(self):
        follower = _make_robot_client()
        worker = _make_worker(follower=follower)
        worker._stop_event.set()
        asyncio.run(worker.run_loop())
        follower.connect.assert_awaited_once()
        follower.disconnect.assert_awaited_once()

    def test_leader_connected_and_disconnected_when_provided(self):
        follower = _make_robot_client()
        leader = _make_robot_client()
        worker = _make_worker(follower=follower, leader=leader)
        worker._stop_event.set()
        asyncio.run(worker.run_loop())
        leader.connect.assert_awaited_once()
        leader.disconnect.assert_awaited_once()

    def test_no_action_source_does_not_write_to_follower(self):
        follower = _make_robot_client()
        worker = _make_worker(follower=follower)
        worker.set_action_source(ActionWriteState.NONE)
        _run_one_iteration(worker)
        follower.set_joints_state.assert_not_called()

    def test_from_actions_writes_buffered_actions_to_follower(self):
        follower = _make_robot_client()
        worker = _make_worker(follower=follower)
        worker._set_actions([1.0] * len(FEATURES))
        worker.set_action_source(ActionWriteState.FROM_ACTIONS)
        _run_one_iteration(worker)
        follower.set_joints_state.assert_called_once()

    def test_from_leader_reads_leader_and_writes_follower(self):
        follower = _make_robot_client()
        leader = _make_robot_client()
        leader.read_state.return_value = {"state": dict.fromkeys(FEATURES, 1.0)}
        worker = _make_worker(follower=follower, leader=leader)
        worker.set_action_source(ActionWriteState.FROM_LEADER)
        _run_one_iteration(worker)
        leader.read_state.assert_called()
        follower.set_joints_state.assert_called_once()

    def test_follower_state_updated_from_read(self):
        follower = _make_robot_client()
        expected_state = {k: float(i) for i, k in enumerate(FEATURES)}
        follower.read_state.return_value = {"state": expected_state}
        worker = _make_worker(follower=follower)
        _run_one_iteration(worker)
        assert worker.get_state() == pytest.approx(list(range(len(FEATURES))))
