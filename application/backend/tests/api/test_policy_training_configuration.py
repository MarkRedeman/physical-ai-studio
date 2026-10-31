"""Studio recipes are static, documented, and valid training payloads."""

import subprocess
import sys
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from main import app
from schemas.job import LocalTrainJobPayload
from training.policy_catalog import (
    _STUDIO_POLICIES,
    POLICY_TRAINING_CONFIGURATIONS,
    LoraDefaults,
    PolicyTrainingConfiguration,
)


def test_every_studio_policy_has_a_configuration() -> None:
    assert POLICY_TRAINING_CONFIGURATIONS.keys() == _STUDIO_POLICIES.keys()


@pytest.mark.parametrize("policy", POLICY_TRAINING_CONFIGURATIONS)
def test_defaults_are_valid_persistable_payloads(policy: str) -> None:
    response = TestClient(app).get(f"/api/policies/{policy}/default-training-config")
    assert response.status_code == 200
    defaults = PolicyTrainingConfiguration.model_validate(response.json()).defaults
    values = defaults.model_dump(exclude={"validation", "lora", "snapflow"})
    lora = defaults.lora or LoraDefaults()
    payload = LocalTrainJobPayload(
        project_id=uuid4(),
        dataset_id=uuid4(),
        policy=policy,
        model_name=policy,
        **values,
        val_split=defaults.validation.split,
        val_split_seed=defaults.validation.seed,
        lora_enabled=defaults.lora is not None,
        lora_rank=lora.rank,
        lora_alpha=lora.alpha,
        lora_dropout=lora.dropout,
        lora_use_dora=lora.use_dora,
        snapflow_enabled=defaults.snapflow is not None,
        snapflow_distill_epochs=defaults.snapflow.distill_epochs if defaults.snapflow else 3,
    )
    assert LocalTrainJobPayload.model_validate_json(payload.model_dump_json()) == payload
    assert all(payload.model_dump()[key] == value for key, value in values.items())
    assert payload.lora_enabled == (defaults.lora is not None)
    assert payload.snapflow_enabled == (defaults.snapflow is not None)


def test_unknown_policy_is_404() -> None:
    assert TestClient(app).get("/api/policies/unknown/default-training-config").status_code == 404


def test_lookup_does_not_load_policy_dependencies() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import asyncio, sys; from api.policies import get_default_training_configuration; "
                "assert asyncio.run(get_default_training_configuration('pi05')).defaults.lora is not None; "
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


@pytest.mark.parametrize("model", ["PolicyTrainingDefaults", "LoraDefaults", "SnapflowDefaults", "ValidationDefaults"])
def test_openapi_documents_defaults(model: str) -> None:
    schema = app.openapi()["components"]["schemas"][model]
    assert all(field.get("description") for field in schema["properties"].values())
    assert set(schema["required"]) == set(schema["properties"])


def test_optional_methods_are_disabled_by_null_not_an_enabled_flag() -> None:
    defaults = POLICY_TRAINING_CONFIGURATIONS["act"].defaults
    assert defaults.lora is None
    assert defaults.snapflow is None
    with pytest.raises(ValidationError):
        LoraDefaults(enabled=False)


def test_unsupported_checkpointing_override_is_rejected() -> None:
    with pytest.raises(ValidationError, match="Gradient checkpointing"):
        LocalTrainJobPayload(
            project_id=uuid4(), dataset_id=uuid4(), policy="act", model_name="act", gradient_checkpointing=True
        )
