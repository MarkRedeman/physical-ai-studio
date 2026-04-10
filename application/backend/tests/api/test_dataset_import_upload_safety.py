import io
import zipfile
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient

from api.dependencies import get_dataset_import_service
from main import app
from schemas.base_job import JobStatus, JobType
from schemas.import_job import DatasetImportJobPayload, ImportStep
from schemas.job import DatasetImportJob


class _StubDatasetImportService:
    def __init__(self, project_id):
        self.project_id = project_id
        self.calls: list[dict] = []

    async def attach_dataset_import_archive(self, project_id, job_id, archive_staging_id, uploaded_archive_name):
        self.calls.append(
            {
                "project_id": project_id,
                "job_id": job_id,
                "archive_staging_id": archive_staging_id,
                "uploaded_archive_name": uploaded_archive_name,
            }
        )
        return DatasetImportJob(
            id=job_id,
            project_id=project_id,
            status=JobStatus.PENDING,
            progress=5,
            message="Dataset archive uploaded and queued",
            payload=DatasetImportJobPayload(
                type=JobType.DATASET_IMPORT,
                step=ImportStep.UPLOADED,
                archive_staging_id=archive_staging_id,
                uploaded_archive_name=uploaded_archive_name,
                source_hint="auto",
            ),
        )


def _make_zip_bytes(files: dict[str, bytes], *, compression=zipfile.ZIP_DEFLATED) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, mode="w", compression=compression) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return stream.getvalue()


def test_upload_rejects_nested_zip_and_does_not_attach_archive() -> None:
    project_id = uuid4()
    job_id = uuid4()
    stub = _StubDatasetImportService(project_id)
    app.dependency_overrides[get_dataset_import_service] = lambda: stub

    nested_zip_bytes = _make_zip_bytes({"payload.txt": b"hello"})
    outer_zip_bytes = _make_zip_bytes(
        {
            "meta/info.json": b"{}",
            "data/episode.parquet": b"parquet",
            "nested/payload.zip": nested_zip_bytes,
        }
    )

    try:
        client = TestClient(app)
        response = client.put(
            f"/api/projects/{project_id}/imports/datasets/{job_id}:upload",
            files={"archive": ("dataset.zip", outer_zip_bytes, "application/zip")},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 413
    body = response.json()
    assert body["error_code"] == "zip_bomb_detected"
    assert "nested zip entry" in body["message"]
    assert stub.calls == []


def test_upload_rejects_archive_with_too_large_uncompressed_size_and_does_not_attach_archive() -> None:
    project_id = uuid4()
    job_id = uuid4()
    stub = _StubDatasetImportService(project_id)
    app.dependency_overrides[get_dataset_import_service] = lambda: stub

    large_payload = b"A" * 10_000
    archive_bytes = _make_zip_bytes(
        {
            "meta/info.json": b"{}",
            "data/episode.parquet": large_payload,
        },
        compression=zipfile.ZIP_STORED,
    )

    with patch("api.imports.get_settings") as mock_get_settings:
        settings = mock_get_settings.return_value
        settings.cache_dir = Path("~/.cache/physicalai").expanduser() / "cache"
        settings.data_import_max_upload_bytes = 100 * 1024 * 1024 * 1024
        settings.data_import_min_free_bytes = 0
        settings.data_import_max_uncompressed_bytes = 2_000

        try:
            client = TestClient(app)
            response = client.put(
                f"/api/projects/{project_id}/imports/datasets/{job_id}:upload",
                files={"archive": ("dataset.zip", archive_bytes, "application/zip")},
            )
        finally:
            app.dependency_overrides.clear()

    assert response.status_code == 413
    body = response.json()
    assert body["error_code"] == "zip_bomb_detected"
    assert "uncompressed size exceeds allowed limit" in body["message"]
    assert stub.calls == []


def test_upload_accepts_valid_zip_and_attaches_archive() -> None:
    project_id = uuid4()
    job_id = uuid4()
    stub = _StubDatasetImportService(project_id)
    app.dependency_overrides[get_dataset_import_service] = lambda: stub

    archive_bytes = _make_zip_bytes(
        {
            "meta/info.json": b"{}",
            "data/episode.parquet": b"small-data",
        },
        compression=zipfile.ZIP_STORED,
    )

    try:
        client = TestClient(app)
        response = client.put(
            f"/api/projects/{project_id}/imports/datasets/{job_id}:upload",
            files={"archive": ("dataset.zip", archive_bytes, "application/zip")},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 202
    assert len(stub.calls) == 1
    # The API now passes archive_staging_id (a UUID string) to the service.
    staging_id = stub.calls[0]["archive_staging_id"]
    assert staging_id is not None
    assert stub.calls[0]["uploaded_archive_name"] == "dataset.zip"
    # The file should have been written to the staging path derived from that id.
    from services.dataset_import.staging import staging_path_for_id

    staged_path = staging_path_for_id(staging_id)
    assert staged_path.exists()
    staged_path.unlink(missing_ok=True)


def test_upload_rejects_archive_with_too_many_entries() -> None:
    project_id = uuid4()
    job_id = uuid4()
    stub = _StubDatasetImportService(project_id)
    app.dependency_overrides[get_dataset_import_service] = lambda: stub

    archive_bytes = _make_zip_bytes(
        {
            **{f"data/file_{index}.txt": b"x" for index in range(101)},
            "meta/info.json": b"{}",
        },
        compression=zipfile.ZIP_STORED,
    )

    with (
        patch("api.imports.get_settings") as mock_get_settings,
        patch("services.archive_safety.DEFAULT_MAX_FILE_COUNT", 100),
    ):
        settings = mock_get_settings.return_value
        settings.cache_dir = Path("~/.cache/physicalai").expanduser() / "cache"
        settings.data_import_max_upload_bytes = 100 * 1024 * 1024 * 1024
        settings.data_import_min_free_bytes = 0
        settings.data_import_max_uncompressed_bytes = 200 * 1024 * 1024 * 1024

        try:
            client = TestClient(app)
            response = client.put(
                f"/api/projects/{project_id}/imports/datasets/{job_id}:upload",
                files={"archive": ("dataset.zip", archive_bytes, "application/zip")},
            )
        finally:
            app.dependency_overrides.clear()

    assert response.status_code == 413
    body = response.json()
    assert body["error_code"] == "zip_bomb_detected"
    assert "too many entries" in body["message"]
    assert stub.calls == []


# ---------------------------------------------------------------------------
# Large-upload guardrail tests
# ---------------------------------------------------------------------------


def test_upload_rejects_when_content_length_exceeds_max() -> None:
    """HTTP guard: reject before reading the body when Content-Length > limit."""
    project_id = uuid4()
    job_id = uuid4()
    stub = _StubDatasetImportService(project_id)
    app.dependency_overrides[get_dataset_import_service] = lambda: stub

    # A tiny but valid ZIP - the rejection must happen purely on the header value.
    archive_bytes = _make_zip_bytes(
        {"meta/info.json": b"{}"},
        compression=zipfile.ZIP_STORED,
    )

    # Patch settings so the threshold is lower than any real upload.
    # We set max_upload_bytes to 1 byte so the declared Content-Length (which
    # TestClient derives from the multipart body length) is always over the cap.
    with patch("middleware.upload_size_guard.get_settings") as mock_get_settings:
        settings = mock_get_settings.return_value
        settings.data_import_max_upload_bytes = 1  # 1 byte - always exceeded

        try:
            client = TestClient(app)
            response = client.put(
                f"/api/projects/{project_id}/imports/datasets/{job_id}:upload",
                files={"archive": ("dataset.zip", archive_bytes, "application/zip")},
            )
        finally:
            app.dependency_overrides.clear()

    assert response.status_code == 413
    body = response.json()
    assert body["error_code"] == "upload_too_large"
    assert "exceeds" in body["message"]
    assert stub.calls == []


def test_upload_rejects_when_cache_dir_has_insufficient_free_space() -> None:
    """Disk guard: reject upload when cache dir has insufficient free space."""
    project_id = uuid4()
    job_id = uuid4()
    stub = _StubDatasetImportService(project_id)
    app.dependency_overrides[get_dataset_import_service] = lambda: stub

    archive_bytes = _make_zip_bytes(
        {"meta/info.json": b"{}"},
        compression=zipfile.ZIP_STORED,
    )

    import shutil

    # Build a fake disk_usage namedtuple that always reports zero free bytes.
    _fake_usage = shutil.disk_usage("/")._replace(free=0)

    with (
        patch("api.imports.get_settings") as mock_get_settings,
        patch("services.archive_safety.shutil.disk_usage", return_value=_fake_usage),
    ):
        settings = mock_get_settings.return_value
        settings.cache_dir = Path("~/.cache/physicalai").expanduser() / "cache"
        settings.datasets_dir = Path("~/.cache/physicalai").expanduser() / "datasets"
        settings.data_import_max_upload_bytes = 10 * 1024 * 1024 * 1024  # huge - no header rejection
        settings.data_import_min_free_bytes = 1  # any positive headroom will be unmet
        settings.data_import_max_uncompressed_bytes = 5 * 1024 * 1024 * 1024

        try:
            client = TestClient(app)
            response = client.put(
                f"/api/projects/{project_id}/imports/datasets/{job_id}:upload",
                files={"archive": ("dataset.zip", archive_bytes, "application/zip")},
            )
        finally:
            app.dependency_overrides.clear()

    assert response.status_code == 507
    body = response.json()
    assert body["error_code"] == "insufficient_disk_space"
    assert stub.calls == []
