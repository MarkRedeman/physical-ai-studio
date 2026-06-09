"""Unit tests for ModelService."""

import json
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from schemas.model import Model
from services.model_service import ModelService


def _make_model(snapshot_id=None) -> Model:
    return Model.model_validate(
        {
            "id": str(uuid4()),
            "name": "test-model",
            "policy": "act",
            "path": "/tmp/test-model",
            "project_id": str(uuid4()),
            "dataset_id": str(uuid4()),
            "snapshot_id": str(snapshot_id) if snapshot_id else None,
            "properties": {},
        }
    )


def test_get_backend_io_spec_reads_manifest(tmp_path) -> None:
    manifest = {
        "model": {
            "input_features": [
                {
                    "class_path": "physicalai.inference.data.features.InferenceFeature",
                    "init_args": {
                        "ftype": "STATE",
                        "shape": [6],
                        "name": "state",
                        "dtype": "float32",
                    },
                },
                {
                    "class_path": "physicalai.inference.data.features.InferenceFeature",
                    "init_args": {
                        "ftype": "VISUAL",
                        "shape": [3, 480, 640],
                        "name": "images.wrist",
                        "dtype": "float32",
                    },
                },
            ]
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    io_spec = ModelService.get_backend_io_spec(tmp_path)

    assert io_spec is not None
    assert [feature.name for feature in io_spec.input_features] == ["state", "images.wrist"]
    assert io_spec.input_features[1].shape == [3, 480, 640]
    assert io_spec.input_features[1].ftype == "VISUAL"
    assert io_spec.input_features[1].dtype == "float32"


def test_get_backend_io_spec_returns_none_for_invalid_manifest(tmp_path) -> None:
    (tmp_path / "manifest.json").write_text("{", encoding="utf-8")

    assert ModelService.get_backend_io_spec(tmp_path) is None


def test_get_backend_details_includes_io_spec(tmp_path) -> None:
    model = _make_model()
    model.path = str(tmp_path)
    backend_dir = tmp_path / "exports" / "torch"
    backend_dir.mkdir(parents=True)
    (backend_dir / "model.pt").write_text("weights", encoding="utf-8")
    (backend_dir / "manifest.json").write_text(
        json.dumps(
            {
                "model": {
                    "input_features": [
                        {
                            "init_args": {
                                "ftype": "STATE",
                                "shape": [6],
                                "name": "state",
                                "dtype": "float32",
                            }
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    details = ModelService.get_backend_details(model)

    assert len(details) == 1
    assert details[0].type == "torch"
    assert details[0].io_spec is not None
    assert details[0].io_spec.input_features[0].name == "state"


@pytest.mark.anyio
async def test_delete_model_deletes_snapshot_when_snapshot_id_set() -> None:
    """When model.snapshot_id is set, delete_model should also delete the snapshot row."""
    snapshot_id = uuid4()
    model = _make_model(snapshot_id=snapshot_id)

    mock_model_repo = AsyncMock()
    mock_snapshot_repo = AsyncMock()

    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("services.model_service.get_async_db_session_ctx", return_value=mock_session),
        patch("services.model_service.ModelRepository", return_value=mock_model_repo),
        patch("services.model_service.SnapshotRepository", return_value=mock_snapshot_repo),
        patch("services.model_service.shutil.rmtree"),
    ):
        await ModelService.delete_model(model)

    mock_model_repo.delete_by_id.assert_awaited_once_with(model.id)
    mock_snapshot_repo.delete_by_id.assert_awaited_once_with(model.snapshot_id)


@pytest.mark.anyio
async def test_delete_model_skips_snapshot_delete_when_no_snapshot_id() -> None:
    """When model.snapshot_id is None, snapshot repo delete should NOT be called."""
    model = _make_model(snapshot_id=None)

    mock_model_repo = AsyncMock()
    mock_snapshot_repo = AsyncMock()

    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("services.model_service.get_async_db_session_ctx", return_value=mock_session),
        patch("services.model_service.ModelRepository", return_value=mock_model_repo),
        patch("services.model_service.SnapshotRepository", return_value=mock_snapshot_repo),
        patch("services.model_service.shutil.rmtree"),
    ):
        await ModelService.delete_model(model)

    mock_model_repo.delete_by_id.assert_awaited_once_with(model.id)
    mock_snapshot_repo.delete_by_id.assert_not_awaited()
