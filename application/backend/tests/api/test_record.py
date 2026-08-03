import asyncio
import queue
from unittest.mock import MagicMock

from api.record import camera_frame_payload, handle_incoming, handle_outgoing
from schemas import InferenceBackend, InferenceDevice, Model


class FakeWebSocket:
    def __init__(self, messages: list[dict]) -> None:
        self._messages = messages

    async def receive_json(self, _: str) -> dict:
        if not self._messages:
            raise RuntimeError("No more messages")
        return self._messages.pop(0)


class RecordingWebSocket:
    def __init__(self) -> None:
        self.json_messages: list[dict] = []
        self.binary_messages: list[bytes] = []

    async def send_json(self, message) -> None:
        self.json_messages.append(message)

    async def send_bytes(self, payload: bytes) -> None:
        self.binary_messages.append(payload)


def test_handle_incoming_load_model_requires_inference_device(test_model) -> None:
    process = MagicMock()
    websocket = FakeWebSocket(
        [
            {
                "event": "load_model",
                "data": {
                    "model": test_model.model_dump(mode="json"),
                    "inference_device": {"backend": "openvino", "device": "GPU"},
                },
            },
            {"event": "disconnect", "data": {}},
        ]
    )

    asyncio.run(handle_incoming(websocket, process, set()))

    process.load_model.assert_called_once_with(
        Model.model_validate(test_model.model_dump(mode="json")),
        InferenceDevice(backend=InferenceBackend.OPENVINO, device="GPU"),
    )
    process.disconnect.assert_called_once()


def test_camera_frame_payload_layout() -> None:
    payload = camera_frame_payload("0823b0fd-5c9f-4c1a-9dd1-5f4e105aebe9", b"\xff\xd8JPEGDATA")
    assert payload[0] == 36  # UUID length
    assert payload[1 : 1 + 36].decode() == "0823b0fd-5c9f-4c1a-9dd1-5f4e105aebe9"
    assert payload[1 + 36 :] == b"\xff\xd8JPEGDATA"


def test_camera_frame_payload_non_ascii_camera_id() -> None:
    good = "Bob's view"
    payload = camera_frame_payload(good, b"\xff\xd8")
    assert payload[0] == len(good.encode())
    assert payload[1 : 1 + len(good)] == good.encode()


def test_handle_outgoing_splits_cameras_into_binary_frames() -> None:
    class NonBlockingGetQueue:
        """Queue whose get() never blocks, so the executor thread can free
        during asyncio teardown instead of hanging on an empty queue."""

        def __init__(self) -> None:
            self._items: list = []

        def put(self, item) -> None:
            self._items.append(item)

        def get(self):
            if not self._items:
                raise queue.Empty
            return self._items.pop(0)

    async def scenario() -> RecordingWebSocket:
        websocket = RecordingWebSocket()
        q = NonBlockingGetQueue()
        task = asyncio.create_task(handle_outgoing(websocket, q))
        q.put(
            {
                "event": "observations",
                "data": {"state": {"a": 0.5}, "timestamp": 1, "cameras": {"cam1": b"\xff\xd8AAA"}},
            }
        )
        q.put({"event": "state", "data": {"model_loaded": True}})
        await asyncio.sleep(0.1)
        if not task.done():
            task.cancel()
        return websocket

    websocket = asyncio.run(scenario())

    assert websocket.json_messages == [
        {"event": "observations", "data": {"state": {"a": 0.5}, "timestamp": 1}},
        {"event": "state", "data": {"model_loaded": True}},
    ]
    assert websocket.binary_messages == [camera_frame_payload("cam1", b"\xff\xd8AAA")]
