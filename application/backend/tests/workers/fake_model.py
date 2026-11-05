"""A model worker whose model is an oracle instead of an export, for the model tests.

Module-level so the spawned child can import it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from physicalai.inference import ExecutionCapabilities, ExecutionMode

from workers.model_worker import ModelWorker

CHUNK = 10


class Oracle:
    """Commands ``state + 1 .. state + CHUNK``, so a follower that follows it keeps climbing."""

    execution_capabilities = ExecutionCapabilities()

    def configure_execution(self, mode: ExecutionMode) -> None:
        if mode not in self.execution_capabilities.supported_execution_modes:
            raise ValueError(f"Unsupported mode: {mode}")
        self.reset()

    def __init__(self, images: list[str]) -> None:
        self.images = images

    def predict_action_chunk(self, observation: dict[str, Any]) -> np.ndarray:
        for name in self.images:
            if observation[name].shape[0] != 1:
                raise ValueError(f"{name} is not one frame")
        state = observation["state"][0]
        return np.stack([state + step for step in range(1, CHUNK + 1)])

    def reset(self) -> None:
        pass


class FakeModelWorker(ModelWorker):
    def open_model(self) -> Oracle:  # type: ignore[override]
        if (self.export_dir / "broken").exists():
            raise RuntimeError("the export does not load")
        return Oracle(list(self.images))
