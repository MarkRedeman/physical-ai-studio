import json
from uuid import uuid4

from schemas.calibration import Calibration, CalibrationValue
from schemas.dataset import Dataset
from schemas.environment import EnvironmentWithRelations
from schemas.model import Model
from services.model_manifest_service import ModelManifestService


def test_write_root_manifest_copies_torch_manifest_with_root_relative_artifacts(tmp_path) -> None:
    torch_dir = tmp_path / "exports" / "torch"
    torch_dir.mkdir(parents=True)
    (torch_dir / "act.pt").write_text("weights", encoding="utf-8")
    (torch_dir / "manifest.json").write_text(
        json.dumps(
            {
                "format": "policy_package",
                "version": "1.0",
                "policy": {"name": "act"},
                "model": {
                    "artifacts": {"torch": "act.pt"},
                    "input_features": [
                        {
                            "class_path": "physicalai.inference.data.features.InferenceFeature",
                            "init_args": {
                                "ftype": "STATE",
                                "shape": [6],
                                "name": "state",
                                "dtype": "float32",
                            },
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    manifest_path = ModelManifestService.write_root_manifest(tmp_path)

    assert manifest_path == tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["format"] == "policy_package"
    assert manifest["policy"] == {"name": "act"}
    assert manifest["model"]["artifacts"] == {"torch": "exports/torch/act.pt"}
    assert manifest["model"]["input_features"][0]["init_args"]["name"] == "state"


def test_write_root_manifest_returns_none_when_torch_manifest_missing(tmp_path) -> None:
    assert ModelManifestService.write_root_manifest(tmp_path) is None
    assert not (tmp_path / "manifest.json").exists()


def test_write_root_manifest_returns_none_for_invalid_torch_manifest(tmp_path) -> None:
    torch_dir = tmp_path / "exports" / "torch"
    torch_dir.mkdir(parents=True)
    (torch_dir / "manifest.json").write_text("{", encoding="utf-8")

    assert ModelManifestService.write_root_manifest(tmp_path) is None
    assert not (tmp_path / "manifest.json").exists()


def test_write_root_manifest_returns_none_for_non_policy_package_manifest(tmp_path) -> None:
    torch_dir = tmp_path / "exports" / "torch"
    torch_dir.mkdir(parents=True)
    (torch_dir / "manifest.json").write_text(json.dumps({"format": "other"}), encoding="utf-8")

    assert ModelManifestService.write_root_manifest(tmp_path) is None
    assert not (tmp_path / "manifest.json").exists()


def test_write_model_card_from_root_manifest(tmp_path) -> None:
    openvino_dir = tmp_path / "exports" / "openvino"
    openvino_dir.mkdir(parents=True)
    (openvino_dir / "manifest.json").write_text(
        json.dumps(
            {
                "format": "policy_package",
                "version": "1.0",
                "policy": {"name": "act"},
                "model": {
                    "artifacts": {"openvino": "act.xml"},
                    "input_features": [
                        {
                            "class_path": "physicalai.inference.data.features.InferenceFeature",
                            "init_args": {
                                "ftype": "STATE",
                                "shape": [6],
                                "name": "state",
                                "dtype": "float32",
                            }
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "format": "policy_package",
                "version": "1.0",
                "policy": {"name": "act"},
                "model": {
                    "artifacts": {"torch": "exports/torch/act.pt"},
                    "input_features": [
                        {
                            "class_path": "physicalai.inference.data.features.InferenceFeature",
                            "init_args": {
                                "ftype": "STATE",
                                "shape": [6],
                                "name": "state",
                                "dtype": "float32",
                            }
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    model = Model(
        id=uuid4(),
        project_id=uuid4(),
        dataset_id=uuid4(),
        path=str(tmp_path),
        name="Block Pickup ACT",
        snapshot_id=uuid4(),
        policy="act",
        properties={},
        train_job_id=uuid4(),
        parent_model_id=None,
        version=1,
        created_at=None,
    )
    dataset = Dataset(
        id=uuid4(),
        name="Block pickup demos",
        path="/tmp/dataset",
        default_task="pick up the block",
        project_id=uuid4(),
        environment_id=uuid4(),
    )
    environment, calibrations = _environment_with_calibration()

    readme_path = ModelManifestService.write_model_card(
        tmp_path,
        model=model,
        dataset=dataset,
        environment=environment,
        calibrations=calibrations,
    )

    assert readme_path == tmp_path / "README.md"
    readme = readme_path.read_text(encoding="utf-8")
    assert "library_name: physicalai" in readme
    assert "pipeline_tag: robotics" in readme
    assert "vision-language-action" in readme
    assert "physical-ai-studio" in readme
    assert "# Action Chunking Transformer (ACT)" in readme
    assert "Model Card for" not in readme
    assert "**Policy:** act" in readme
    assert "**Block pickup demos**" in readme
    assert "## Model Package" in readme
    assert "| Backend | Artifact | Intended Use |" in readme
    assert "| torch | `exports/torch/act.pt` | Canonical checkpoint and Python inference |" in readme
    assert "| openvino | `exports/openvino/act.xml` | CPU, Intel GPU, and NPU inference |" in readme
    assert "### Shared By `openvino`, `torch`" in readme
    assert "### openvino" not in readme
    assert "### torch" not in readme
    assert "| state" in readme
    assert "| STATE" in readme
    assert "| [6]" in readme
    assert "| float32" in readme
    assert 'model = InferenceModel.load(MODEL_PATH, device="CPU")' in readme
    assert '"state": np.random.rand(1, 6).astype(np.float32)' in readme
    assert "python examples/runtime/sync_inference.py" in readme
    assert "--robot so101" in readme
    assert "--camera overhead:uvc:/dev/video0" in readme
    assert '--task "pick up the block"' in readme
    assert "Local device handles are placeholders" in readme
    assert "## Training Environment" in readme
    assert "SO101 follower" in readme
    assert "teleoperator" not in readme
    assert "SO101 leader" not in readme
    assert "Logitech C920" in readme
    assert "width: 640" in readme
    assert "height: 480" in readme
    assert "```yaml" in readme
    assert "calibration:" in readme
    assert "shoulder_pan:" in readme
    assert "homing_offset: 10" in readme
    assert "\n\n\n" not in readme
    assert "| --- | --- | --- | --- |\n| state" in readme
    assert 'observation = {\n    "state"' in readme
    assert '"state": np.random.rand(1, 6).astype(np.float32),\n}' in readme


def test_write_model_card_returns_none_when_root_manifest_missing(tmp_path) -> None:
    assert ModelManifestService.write_model_card(tmp_path) is None
    assert not (tmp_path / "README.md").exists()


def test_write_environment_description_sanitizes_and_includes_calibration(tmp_path) -> None:
    environment, calibrations = _environment_with_calibration()

    environment_path = ModelManifestService.write_environment_description(tmp_path, environment, calibrations)

    assert environment_path == tmp_path / "environment.json"
    environment_json = environment_path.read_text(encoding="utf-8")
    environment_data = json.loads(environment_json)
    assert environment_data["format"] == "physical_ai_studio_environment"
    assert environment_data["name"] == "SO101 tabletop"
    assert environment_data["robots"][0] == {
        "name": "SO101 follower",
        "type": "SO101_Follower",
        "calibration": {
            "shoulder_pan": {
                "id": 1,
                "drive_mode": 1,
                "homing_offset": 10,
                "range_min": 100,
                "range_max": 3900,
            }
        },
        "teleoperator": {
            "type": "robot",
            "name": "SO101 leader",
            "robot_type": "SO101_Leader",
        },
    }
    assert environment_data["cameras"][0] == {
        "name": "overhead",
        "driver": "usb_camera",
        "hardware_name": "Logitech C920",
        "width": 640,
        "height": 480,
        "fps": 30,
    }
    assert "active_calibration_id" not in environment_json
    assert "calibration-id" not in environment_json
    assert "calibration.json" not in environment_json
    assert "/dev/ttyACM0" not in environment_json
    assert "SO101-2024-001" not in environment_json
    assert "USB\\VID" not in environment_json
    assert "fingerprint" not in environment_json


def test_write_runtime_calibration_exports_primary_robot_calibration(tmp_path) -> None:
    environment, calibrations = _environment_with_calibration()

    calibration_path = ModelManifestService.write_runtime_calibration(tmp_path, environment, calibrations)

    assert calibration_path == tmp_path / "calibration.json"
    calibration_json = calibration_path.read_text(encoding="utf-8")
    assert json.loads(calibration_json) == {
        "shoulder_pan": {
            "id": 1,
            "drive_mode": 1,
            "homing_offset": 10,
            "range_min": 100,
            "range_max": 3900,
        }
    }
    assert "robot_id" not in calibration_json
    assert "calibration.json" not in calibration_json
    assert "file_path" not in calibration_json


def test_write_runtime_calibration_returns_none_without_calibration(tmp_path) -> None:
    environment, _ = _environment_with_calibration()

    assert ModelManifestService.write_runtime_calibration(tmp_path, environment, {}) is None
    assert not (tmp_path / "calibration.json").exists()


def test_write_model_card_without_environment(tmp_path) -> None:
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "format": "policy_package",
                "version": "1.0",
                "policy": {"name": "smolvla"},
                "model": {"artifacts": {"torch": "exports/torch/model.pt"}},
            }
        ),
        encoding="utf-8",
    )

    readme_path = ModelManifestService.write_model_card(tmp_path)

    readme = readme_path.read_text(encoding="utf-8")
    assert "# SmolVLA" in readme
    assert "No training environment description was exported with this model." in readme


def _environment_with_calibration() -> tuple[EnvironmentWithRelations, dict]:
    follower_id = uuid4()
    leader_id = uuid4()
    calibration_id = uuid4()
    environment = EnvironmentWithRelations.model_validate(
        {
            "id": uuid4(),
            "name": "SO101 tabletop",
            "robots": [
                {
                    "robot": {
                        "id": follower_id,
                        "name": "SO101 follower",
                        "type": "SO101_Follower",
                        "active_calibration_id": calibration_id,
                        "payload": {
                            "connection_string": "/dev/ttyACM0",
                            "serial_number": "SO101-2024-001",
                        },
                    },
                    "tele_operator": {
                        "type": "robot",
                        "robot_id": leader_id,
                        "robot": {
                            "id": leader_id,
                            "name": "SO101 leader",
                            "type": "SO101_Leader",
                            "active_calibration_id": None,
                            "payload": {
                                "connection_string": "/dev/ttyACM1",
                                "serial_number": "SO101-2024-002",
                            },
                        },
                    },
                }
            ],
            "cameras": [
                {
                    "id": uuid4(),
                    "name": "overhead",
                    "driver": "usb_camera",
                    "fingerprint": "USB\\VID_1234&PID_5678:0",
                    "hardware_name": "Logitech C920",
                    "payload": {"width": 640, "height": 480, "fps": 30},
                }
            ],
        }
    )
    calibration = Calibration.model_validate(
        {
            "id": calibration_id,
            "robot_id": follower_id,
            "file_path": "/home/user/calibration.json",
            "values": {
                "shoulder_pan": CalibrationValue.model_validate(
                    {
                        "id": 1,
                        "joint_name": "shoulder_pan",
                        "drive_mode": 1,
                        "homing_offset": 10,
                        "range_min": 100,
                        "range_max": 3900,
                    }
                )
            },
        }
    )
    return environment, {follower_id: calibration}
