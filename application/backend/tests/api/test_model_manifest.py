# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

from api.dependencies import get_model_service
from main import app
from schemas import Model


class _StubModelService:
    def __init__(self, model: Model) -> None:
        self._model = model

    async def get_model_by_id(self, _model_id) -> Model:
        return self._model


def _make_model(path: Path) -> Model:
    return Model(
        id=uuid4(),
        name="test-model",
        path=str(path),
        policy="act",
        properties={},
        project_id=uuid4(),
        dataset_id=uuid4(),
        snapshot_id=uuid4(),
    )


def _write_manifest(backend_dir: Path) -> None:
    backend_dir.mkdir(parents=True)
    (backend_dir / "manifest.json").write_text(
        json.dumps(
            {
                "model": {
                    "input_features": [
                        {
                            "init_args": {
                                "name": "state",
                                "ftype": "STATE",
                                "shape": [6],
                                "dtype": "float32",
                            }
                        }
                    ],
                    "output_features": [
                        {
                            "init_args": {
                                "name": "action",
                                "ftype": "ACTION",
                                "shape": [6],
                                "dtype": "float32",
                            }
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )


def test_model_manifest_returns_features_from_default_backend(tmp_path: Path) -> None:
    model = _make_model(tmp_path)
    _write_manifest(tmp_path / "exports" / "torch")
    app.dependency_overrides[get_model_service] = lambda: _StubModelService(model)

    try:
        response = TestClient(app).get(f"/api/models/{model.id}/manifest")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "backend": "torch",
        "input_features": [{"name": "state", "ftype": "STATE", "shape": [6], "dtype": "float32"}],
        "output_features": [{"name": "action", "ftype": "ACTION", "shape": [6], "dtype": "float32"}],
    }


def test_model_manifest_reads_requested_backend(tmp_path: Path) -> None:
    model = _make_model(tmp_path)
    _write_manifest(tmp_path / "exports" / "openvino")
    app.dependency_overrides[get_model_service] = lambda: _StubModelService(model)

    try:
        response = TestClient(app).get(f"/api/models/{model.id}/manifest?backend=openvino")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["backend"] == "openvino"


def test_model_manifest_returns_not_found_when_missing(tmp_path: Path) -> None:
    model = _make_model(tmp_path)
    app.dependency_overrides[get_model_service] = lambda: _StubModelService(model)

    try:
        response = TestClient(app).get(f"/api/models/{model.id}/manifest")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 404


def test_model_manifest_returns_not_found_when_invalid(tmp_path: Path) -> None:
    model = _make_model(tmp_path)
    backend_dir = tmp_path / "exports" / "torch"
    backend_dir.mkdir(parents=True)
    (backend_dir / "manifest.json").write_text("{", encoding="utf-8")
    app.dependency_overrides[get_model_service] = lambda: _StubModelService(model)

    try:
        response = TestClient(app).get(f"/api/models/{model.id}/manifest")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 404
