"""Studio recipes are static, documented, and valid training payloads."""

import subprocess
import sys
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from main import app
from schemas.job import LocalTrainJobPayload
from training.policy_catalog import _STUDIO_POLICIES, POLICY_TRAINING_CONFIGURATIONS


def test_every_studio_policy_has_a_configuration() -> None:
    assert POLICY_TRAINING_CONFIGURATIONS.keys() == _STUDIO_POLICIES.keys()


@pytest.mark.parametrize("policy", POLICY_TRAINING_CONFIGURATIONS)
def test_defaults_are_valid_persistable_payloads(policy: str) -> None:
    response = TestClient(app).get(f"/api/policies/{policy}/default-training-config")
    assert response.status_code == 200
    defaults = response.json()["defaults"]
    payload = LocalTrainJobPayload(project_id=uuid4(), dataset_id=uuid4(), policy=policy, model_name=policy, **defaults)
    assert LocalTrainJobPayload.model_validate_json(payload.model_dump_json()) == payload
    assert all(payload.model_dump()[key] == value for key, value in defaults.items())


def test_unknown_policy_is_404() -> None:
    assert TestClient(app).get("/api/policies/unknown/default-training-config").status_code == 404


def test_lookup_does_not_load_policy_dependencies() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import asyncio, sys; from api.policies import get_default_training_configuration; "
                "assert asyncio.run(get_default_training_configuration('pi05')).defaults.lora_enabled; "
                "assert 'physicalai.policies' not in sys.modules"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_camera_slots_are_ordered_and_documented() -> None:
    configuration = POLICY_TRAINING_CONFIGURATIONS["smolvla"]
    assert [slot.id for slot in configuration.camera_slots] == ["camera1", "camera2", "camera3"]
    assert [slot.is_required for slot in configuration.camera_slots] == [True, False, False]
    assert all(slot.description for slot in configuration.camera_slots)


def test_openapi_documents_defaults() -> None:
    schema = app.openapi()["components"]["schemas"]["PolicyTrainingDefaults"]
    assert all(field.get("description") for field in schema["properties"].values())


def test_unsupported_checkpointing_override_is_rejected() -> None:
    with pytest.raises(ValidationError, match="Gradient checkpointing"):
        LocalTrainJobPayload(
            project_id=uuid4(), dataset_id=uuid4(), policy="act", model_name="act", gradient_checkpointing=True
        )
