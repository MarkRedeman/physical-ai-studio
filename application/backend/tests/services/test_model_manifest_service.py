import json

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
