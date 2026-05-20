import asyncio
import os
from multiprocessing.managers import BaseManager

from schemas import Model

_DEFAULT_HOST = os.environ.get("INFERENCE_HOST", "127.0.0.1")
_DEFAULT_PORT = int(os.environ.get("INFERENCE_PORT", "50051"))
_DEFAULT_AUTHKEY = os.environ.get("INFERENCE_AUTHKEY", "geti-inference").encode()


class _InferenceManagerClient(BaseManager):
    pass


_InferenceManagerClient.register("observation_queue")
_InferenceManagerClient.register("output_queue")
_InferenceManagerClient.register("command_queue")
_InferenceManagerClient.register("model_loaded_event")


class RemoteModelWorker:
    """Connects to a running async_inference.py server and exposes the same
    observation_queue / output_queue interface as ModelWorker."""

    def __init__(
        self,
        host: str = _DEFAULT_HOST,
        port: int = _DEFAULT_PORT,
        authkey: bytes = _DEFAULT_AUTHKEY,
    ) -> None:
        self._client = _InferenceManagerClient(address=(host, port), authkey=authkey)

    def connect(self) -> None:
        """Connect to the inference server. Raises ConnectionRefusedError if not running."""
        self._client.connect()
        self.observation_queue = self._client.observation_queue()  # type: ignore[attr-defined]
        self.output_queue = self._client.output_queue()  # type: ignore[attr-defined]
        self._command_queue = self._client.command_queue()  # type: ignore[attr-defined]
        self._model_loaded_event = self._client.model_loaded_event()  # type: ignore[attr-defined]

    def load_model(self, model: Model, backend: str) -> None:
        self._command_queue.put(("load", model.model_dump_json(), backend))

    async def wait_for_loading_to_complete(self) -> None:
        await asyncio.to_thread(self._model_loaded_event.wait)
