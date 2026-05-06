import io
import zipfile
from uuid import uuid4

from fastapi.testclient import TestClient

from api.dependencies import get_model_import_service
from main import app
from schemas import Model


class _StubModelImportService:
    def __init__(self, model: Model):
        self.model = model
        self.calls: list[dict] = []

    async def import_model_archive(
        self,
        *,
        archive_path,
        project_id,
        dataset_id,
        policy,
        model_name,
        base_model_id,
        version,
    ) -> Model:
        self.calls.append(
            {
                "archive_path": archive_path,
                "project_id": project_id,
                "dataset_id": dataset_id,
                "policy": policy,
                "model_name": model_name,
                "base_model_id": base_model_id,
                "version": version,
            }
        )
        return self.model


def _make_zip_bytes() -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("model.ckpt", b"checkpoint")
    return stream.getvalue()


def test_direct_model_import_endpoint_accepts_zip_and_returns_model() -> None:
    project_id = uuid4()
    dataset_id = uuid4()
    model = Model.model_validate(
        {
            "id": str(uuid4()),
            "name": "Imported model",
            "path": "/tmp/imported",
            "policy": "pi05",
            "properties": {},
            "project_id": str(project_id),
            "dataset_id": str(dataset_id),
            "snapshot_id": str(uuid4()),
        }
    )
    stub = _StubModelImportService(model)
    app.dependency_overrides[get_model_import_service] = lambda: stub

    try:
        client = TestClient(app)
        response = client.post(
            f"/api/projects/{project_id}/imports/models:import",
            data={
                "dataset_id": str(dataset_id),
                "policy": "pi05",
                "model_name": "Imported model",
                "version": "1",
            },
            files={"archive": ("model.zip", _make_zip_bytes(), "application/zip")},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 201
    assert response.json()["id"] == str(model.id)
    assert len(stub.calls) == 1
    assert stub.calls[0]["project_id"] == project_id


def test_direct_model_import_endpoint_rejects_non_zip_archive() -> None:
    project_id = uuid4()
    dataset_id = uuid4()
    model = Model.model_validate(
        {
            "id": str(uuid4()),
            "name": "Imported model",
            "path": "/tmp/imported",
            "policy": "pi05",
            "properties": {},
            "project_id": str(project_id),
            "dataset_id": str(dataset_id),
            "snapshot_id": str(uuid4()),
        }
    )
    stub = _StubModelImportService(model)
    app.dependency_overrides[get_model_import_service] = lambda: stub

    try:
        client = TestClient(app)
        response = client.post(
            f"/api/projects/{project_id}/imports/models:import",
            data={
                "dataset_id": str(dataset_id),
                "policy": "pi05",
                "model_name": "Imported model",
            },
            files={"archive": ("model.txt", b"not-a-zip", "text/plain")},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 415
    assert len(stub.calls) == 0
