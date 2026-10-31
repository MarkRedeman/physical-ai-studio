"""Draft setup exercises real validation/claims with fake robot hardware."""

import math
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import numpy as np
import pytest
from fastapi.testclient import TestClient
from physicalai.robot.transport import claim_devices
from physicalai_studio_plugin import RobotCatalogDefinition
from pydantic import BaseModel, Field, model_validator

from api.dependencies import get_project_service, get_robot_catalog_service, get_robot_client_factory
from api.robot_live_connect import DirectRobotSession
from main import app
from services.robot_catalog_service import RobotCatalogService

PROJECT_ID = uuid4()
ROBOT_ID = uuid4()
ROBOT_TYPE = "LiveTest_Follower"
URL = f"/api/projects/{PROJECT_ID}/robots/draft/observations/ws"


class LivePayload(BaseModel):
    connection: str
    timeout: float = 5.0
    label: str = Field(default="", json_schema_extra={"x-physicalai-ui": {"required": True}})

    @model_validator(mode="after")
    def check_connection(self):
        if self.connection == "invalid":
            raise ValueError("Invalid test connection")
        return self


class FakeDriver:
    def __init__(self) -> None:
        self.device_ids = (f"fake:{uuid4().hex}",)
        self.joint_names = ["shoulder", "gripper"]
        self.calls: list[str] = []
        self.value = 15.0
        self.timestamp_offset = 0.0
        self.connect_error: Exception | None = None
        self.connect_gate: threading.Event | None = None
        self.entered = threading.Event()

    def connect(self) -> None:
        self.calls.append("connect")
        self.entered.set()
        if self.connect_gate is not None:
            self.connect_gate.wait(timeout=5)
        if self.connect_error is not None:
            raise self.connect_error

    def get_observation(self):
        self.calls.append("read")
        return SimpleNamespace(
            joint_positions=np.array([self.value, 0.01]), timestamp=time.monotonic() - self.timestamp_offset
        )

    def disconnect(self) -> None:
        self.calls.append("disconnect")

    def send_action(self, *_args, **_kwargs) -> None:
        pytest.fail("Live setup must never send actions")


@pytest.fixture
def live_driver():
    return FakeDriver()


@pytest.fixture
def live_client(live_driver, monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    catalog = RobotCatalogService()
    registry = catalog.registry
    builder = AsyncMock(return_value=live_driver)
    registry.register_robot(
        RobotCatalogDefinition(
            type=ROBOT_TYPE, display_name="Test arm", role="follower", robot_payload=LivePayload, robot_builder=builder
        )
    )
    definition = registry.get_definition(ROBOT_TYPE)
    factory = SimpleNamespace(build_robot_driver=AsyncMock(return_value=(live_driver, definition)))
    project = SimpleNamespace(get_project_by_id=AsyncMock(return_value=SimpleNamespace(id=PROJECT_ID)))
    app.dependency_overrides[get_robot_catalog_service] = lambda: catalog
    app.dependency_overrides[get_robot_client_factory] = lambda: factory
    app.dependency_overrides[get_project_service] = lambda: project
    try:
        yield TestClient(app), factory
    finally:
        app.dependency_overrides.clear()


def connect_command(request_id="first", **payload):
    return {
        "command": "connect",
        "request_id": request_id,
        "robot": {
            "id": str(ROBOT_ID),
            "name": "Test arm",
            "type": ROBOT_TYPE,
            "payload": {"connection": "fake", "label": "Arm", **payload},
        },
    }


def observation(websocket):
    event = websocket.receive_json()
    if event["event"] == "connecting":
        event = websocket.receive_json()
    return event


def test_validation_applies_defaults_without_building(live_client):
    client, factory = live_client
    response = client.post(f"/api/robots/catalog/{ROBOT_TYPE}/validate", json={"connection": "fake", "label": "Arm"})
    assert response.status_code == 200
    assert response.json()["timeout"] == 5.0
    factory.build_robot_driver.assert_not_called()


def test_assetless_catalog_entry_is_explicit(live_client):
    client, _factory = live_client
    entries = client.get("/api/robots/catalog").json()
    entry = next(item for item in entries if item["type"] == ROBOT_TYPE)
    assert entry["urdf_path"] is None


@pytest.mark.parametrize("payload", [{"connection": "invalid", "label": "Arm"}, {"connection": "fake"}, {}])
def test_validation_rejects_custom_and_ui_required_errors(live_client, payload):
    client, factory = live_client
    response = client.post(f"/api/robots/catalog/{ROBOT_TYPE}/validate", json=payload)
    assert response.status_code == 400
    factory.build_robot_driver.assert_not_called()


def test_stream_and_acknowledged_disconnect(live_client, live_driver):
    client, factory = live_client
    with client.websocket_connect(URL) as websocket:
        assert live_driver.calls == []  # Opening a socket is not consent to connect.
        websocket.send_json(connect_command())
        event = observation(websocket)
        assert event == {
            "event": "observation",
            "request_id": "first",
            "data": {"shoulder.pos": 15.0, "gripper.pos": 0.01},
        }
        websocket.send_json({"command": "disconnect", "request_id": "release"})
        while websocket.receive_json()["event"] != "disconnected":
            pass
        assert live_driver.calls[-1] == "disconnect"
        with claim_devices(live_driver.device_ids):
            pass  # Release acknowledgement means the claim is available.
    factory.build_robot_driver.assert_awaited_once()


@pytest.mark.parametrize("offset,value", [(10.0, 15.0), (0.0, math.nan)])
def test_invalid_observations_fail_and_clean_up(live_client, live_driver, offset, value):
    live_driver.timestamp_offset = offset
    live_driver.value = value
    client, _factory = live_client
    with client.websocket_connect(URL) as websocket:
        websocket.send_json(connect_command())
        event = observation(websocket)
        assert event["event"] == "error"
    assert live_driver.calls[-1] == "disconnect"


def test_partial_connect_failure_is_disconnected(live_client, live_driver):
    live_driver.connect_error = PermissionError("serial permission denied")
    client, _factory = live_client
    with client.websocket_connect(URL) as websocket:
        websocket.send_json(connect_command())
        assert observation(websocket)["event"] == "error"
    assert live_driver.calls == ["connect", "disconnect"]


def test_device_in_use_is_rejected_before_driver_connect(live_client, live_driver):
    client, _factory = live_client
    with claim_devices(live_driver.device_ids), client.websocket_connect(URL) as websocket:
        websocket.send_json(connect_command())
        event = observation(websocket)
        assert event["error_code"] == "robot_device_already_owned"
    assert live_driver.calls == []


def test_reconfiguration_releases_before_reconnecting(live_client, live_driver):
    client, _factory = live_client
    with client.websocket_connect(URL) as websocket:
        websocket.send_json(connect_command())
        assert observation(websocket)["event"] == "observation"
        websocket.send_json({"command": "disconnect", "request_id": "release"})
        while websocket.receive_json()["event"] != "disconnected":
            pass
        websocket.send_json(connect_command("second", connection="new-fake"))
        assert observation(websocket)["request_id"] == "second"
    first_release = live_driver.calls.index("disconnect")
    assert live_driver.calls.index("connect", first_release) > first_release


def test_browser_close_during_connect_cleans_up_late_completion(live_client, live_driver):
    client, _factory = live_client
    live_driver.connect_gate = threading.Event()
    with client.websocket_connect(URL) as websocket:
        websocket.send_json(connect_command())
        assert live_driver.entered.wait(timeout=2)
        websocket.close()
        live_driver.connect_gate.set()
    assert live_driver.calls[-1] == "disconnect"


def test_unknown_or_invalid_draft_never_builds(live_client):
    client, factory = live_client
    with client.websocket_connect(URL) as websocket:
        websocket.send_json(connect_command(connection="invalid"))
        assert websocket.receive_json()["error_code"] == "invalid_config"
    factory.build_robot_driver.assert_not_called()


def test_unknown_robot_type_is_rejected_before_building(live_client):
    client, factory = live_client
    command = connect_command()
    command["robot"]["type"] = "Unknown_Follower"
    with client.websocket_connect(URL) as websocket:
        websocket.send_json(command)
        assert websocket.receive_json()["error_code"] == "Robot_not_found"
    factory.build_robot_driver.assert_not_called()


def test_direct_session_retains_claim_if_disconnect_fails(monkeypatch, live_driver):
    session = DirectRobotSession()
    session.connect(live_driver)

    def fail_disconnect():
        raise RuntimeError("driver cleanup failed")

    with monkeypatch.context() as patch:
        patch.setattr(live_driver, "disconnect", fail_disconnect)
        with pytest.raises(RuntimeError):
            session.disconnect()
        with pytest.raises(Exception, match="already in use"), claim_devices(live_driver.device_ids):
            pytest.fail("Failed cleanup must retain the claim")
    session.disconnect()
