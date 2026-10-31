"""Studio recipes are static, documented, and valid training payloads."""

import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import pytest
import yaml
from fastapi.testclient import TestClient
from pydantic import ValidationError

from main import app
from schemas.job import LocalTrainJobPayload
from training.job import PRETRAINED_BASE_CHECKPOINTS
from training.policy_catalog import _STUDIO_POLICIES, POLICY_TRAINING_CONFIGURATIONS
from training.policy_training_configuration import (
    LoraDefaults,
    PolicyTrainingConfiguration,
    PolicyTrainingDefaults,
    SnapflowDefaults,
)


def test_every_studio_policy_has_a_configuration() -> None:
    assert POLICY_TRAINING_CONFIGURATIONS.keys() == _STUDIO_POLICIES.keys()


def test_recipe_can_select_steps_without_default_epochs() -> None:
    defaults = PolicyTrainingDefaults(max_steps=100, snapflow=SnapflowDefaults(distill_steps=25))
    assert defaults.max_epochs is None
    assert defaults.max_steps == 100
    assert defaults.snapflow.distill_epochs is None
    assert defaults.snapflow.distill_steps == 25


def test_recipe_distillation_requires_matching_units() -> None:
    with pytest.raises(ValidationError, match="same unit"):
        PolicyTrainingDefaults(max_steps=100, snapflow=SnapflowDefaults())


@pytest.mark.parametrize(
    ("policy", "recipe"),
    [
        ("act", "act/so101/default.yaml"),
        ("smolvla", "smolvla/so100/default.yaml"),
        ("molmoact2", "molmoact2/single_robot.yaml"),
    ],
)
def test_single_robot_recipe_matches_studio_defaults(policy: str, recipe: str) -> None:
    root = Path(__file__).resolve().parents[4] / "library/configs/physicalai"
    configuration = yaml.safe_load((root / recipe).read_text())
    defaults = POLICY_TRAINING_CONFIGURATIONS[policy].defaults
    model = configuration["model"]["init_args"]
    data = configuration["data"]["init_args"]
    trainer = configuration["trainer"]

    if checkpoint := model.get("pretrained_name_or_path"):
        assert checkpoint == PRETRAINED_BASE_CHECKPOINTS[policy]
    assert defaults.batch_size == data["train_batch_size"]
    assert defaults.num_workers == data["num_workers"]
    assert defaults.max_epochs == trainer.get("max_epochs")
    assert defaults.max_steps == trainer.get("max_steps")
    assert defaults.precision == trainer["precision"]
    assert defaults.validation.split == data["val_split"]
    assert defaults.validation.seed == data["val_split_seed"]
    assert defaults.compile_model == model["compile_model"]
    assert defaults.gradient_checkpointing == model.get("gradient_checkpointing")
    assert defaults.augment_images == ("image_transforms" in data)
    assert (defaults.lora is not None) == model.get("lora_enabled", False)
    if defaults.lora is not None:
        assert defaults.lora.model_dump() == {key: model[f"lora_{key}"] for key in defaults.lora.model_dump()}
    assert defaults.snapflow is None


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
        snapflow_distill_steps=defaults.snapflow.distill_steps if defaults.snapflow else None,
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
