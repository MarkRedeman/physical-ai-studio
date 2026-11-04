import asyncio
from contextlib import suppress
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from api import runtime_ws
from runtime_session.controls.model import ModelControl
from runtime_session.controls.teleop import TeleopControl
from runtime_session.store import Sample
from schemas.runtime_session import AckMessage


async def test_stop_ack_is_not_queued_behind_a_starting_return(monkeypatch):
    queue = asyncio.Queue()
    entered, finish, stopped = asyncio.Event(), asyncio.Event(), asyncio.Event()
    replies = []

    async def request(_bus, raw, **_kwargs):
        if '"start"' in raw:
            entered.set()
            await finish.wait()
            return AckMessage(request_id="start", ok=True)
        stopped.set()
        return AckMessage(request_id="stop", ok=True)

    async def send(reply):
        replies.append(reply)

    monkeypatch.setattr(runtime_ws, "_request", request)
    socket = SimpleNamespace(receive_text=queue.get, send_json=send)
    forwarding = asyncio.create_task(runtime_ws._forward_commands(socket, SimpleNamespace(bus=object())))
    try:
        await queue.put('{"command":"return_home","request_id":"start"}')
        await asyncio.wait_for(entered.wait(), 1)
        await queue.put('{"command":"cancel_home_return","request_id":"stop"}')
        await asyncio.wait_for(stopped.wait(), 1)
        await asyncio.sleep(0)
        assert [reply["request_id"] for reply in replies] == ["stop"]
        finish.set()
    finally:
        forwarding.cancel()
        with suppress(asyncio.CancelledError):
            await forwarding


def test_stopped_controls_do_not_publish_a_late_action():
    model = ModelControl.__new__(ModelControl)
    model.store = Mock()
    model.store.read.return_value = Sample(np.array([0.0]), ("joint.pos",), 1)
    model.source = Mock()
    model.source.update.return_value = np.array([99.0])
    model.step = 0
    model.should_stop = lambda: True
    model.on_tick()
    model.store.write.assert_not_called()
    teleop = TeleopControl.__new__(TeleopControl)
    teleop.leader = Mock()
    teleop.store = Mock()
    teleop.should_stop = lambda: True
    teleop.on_tick()
    teleop.store.write.assert_not_called()
