# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest

from workers.remote_model_worker import RemoteModelWorker


class TestConnect:
    def test_connect_raises_when_server_not_running(self):
        worker = RemoteModelWorker(host="127.0.0.1", port=9)  # port 9 (discard) is never open
        with pytest.raises(ConnectionRefusedError):
            worker.connect()


class TestLoadModel:
    def test_load_model_puts_load_command_on_queue(self, test_model):
        worker = RemoteModelWorker()
        worker._command_queue = MagicMock()
        worker.load_model(test_model, "torch")
        worker._command_queue.put.assert_called_once()
        command, model_json, backend = worker._command_queue.put.call_args[0][0]
        assert command == "load"
        assert backend == "torch"

    def test_load_model_serializes_model_as_json(self, test_model):
        worker = RemoteModelWorker()
        worker._command_queue = MagicMock()
        worker.load_model(test_model, "openvino")
        _, model_json, _ = worker._command_queue.put.call_args[0][0]
        import json
        parsed = json.loads(model_json)
        assert parsed["name"] == test_model.name
