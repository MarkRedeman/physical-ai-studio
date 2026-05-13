from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from exceptions import InvalidArchiveError
from schemas import Model
from schemas.dataset import Dataset, Snapshot
from schemas.job import TrainJob
from services.model_import_service import ModelImportService


@pytest.mark.anyio
async def test_import_model_archive_creates_model_and_job(tmp_path: Path) -> None:
    archive_path = tmp_path / "model.zip"
    archive_path.write_bytes(b"zip")

    project_id = uuid4()
    dataset_id = uuid4()

    settings = SimpleNamespace(
        models_dir=tmp_path / "models",
        data_import_max_uncompressed_bytes=1024 * 1024,
        data_import_min_free_bytes=0,
    )

    dataset = Dataset.model_validate(
        {
            "id": str(dataset_id),
            "name": "dataset",
            "path": str(tmp_path / "dataset"),
            "default_task": "task",
            "project_id": str(project_id),
            "environment_id": str(uuid4()),
        }
    )
    snapshot = Snapshot.model_validate(
        {
            "id": str(uuid4()),
            "dataset_id": str(dataset_id),
            "path": str(tmp_path / "snapshot"),
        }
    )
    job = TrainJob.model_validate(
        {
            "id": str(uuid4()),
            "project_id": str(project_id),
            "type": "training",
            "status": "completed",
            "payload": {
                "project_id": str(project_id),
                "dataset_id": str(dataset_id),
                "policy": "pi05",
                "model_name": "imported",
            },
        }
    )

    def _extract_to(destination: Path, *, min_free_bytes: int = 0) -> int:
        (destination / "model.ckpt").write_text("checkpoint")
        return 1

    safe_archive = SimpleNamespace(validate=lambda: None, extract_to=_extract_to)

    async def _create_model(model: Model) -> Model:
        return model

    with (
        patch("services.model_import_service.get_settings", return_value=settings),
        patch("services.model_import_service.DatasetService.get_dataset_by_id", AsyncMock(return_value=dataset)),
        patch("services.model_import_service.SafeZipArchive", return_value=safe_archive),
        patch("services.model_import_service.flatten_single_root_directory"),
        patch(
            "services.model_import_service.SnapshotService.create_snapshot_for_dataset",
            AsyncMock(return_value=snapshot),
        ),
        patch("services.model_import_service.JobService.create_job", AsyncMock(return_value=job)),
        patch("services.model_import_service.ModelService.create_model", AsyncMock(side_effect=_create_model)),
        patch(
            "services.model_import_service.asyncio.to_thread", AsyncMock(side_effect=lambda fn, *a, **k: fn(*a, **k))
        ),
    ):
        model = await ModelImportService().import_model_archive(
            archive_path=archive_path,
            project_id=project_id,
            dataset_id=dataset_id,
            policy="pi05",
            model_name="imported",
        )

    assert model.project_id == project_id
    assert model.dataset_id == dataset_id
    assert model.train_job_id == job.id
    assert Path(model.path).is_dir()


@pytest.mark.anyio
async def test_import_model_archive_accepts_archive_without_model_ckpt(tmp_path: Path) -> None:
    archive_path = tmp_path / "model.zip"
    archive_path.write_bytes(b"zip")

    project_id = uuid4()
    dataset_id = uuid4()

    settings = SimpleNamespace(
        models_dir=tmp_path / "models",
        data_import_max_uncompressed_bytes=1024 * 1024,
        data_import_min_free_bytes=0,
    )

    dataset = Dataset.model_validate(
        {
            "id": str(dataset_id),
            "name": "dataset",
            "path": str(tmp_path / "dataset"),
            "default_task": "task",
            "project_id": str(project_id),
            "environment_id": str(uuid4()),
        }
    )
    snapshot = Snapshot.model_validate(
        {
            "id": str(uuid4()),
            "dataset_id": str(dataset_id),
            "path": str(tmp_path / "snapshot"),
        }
    )
    job = TrainJob.model_validate(
        {
            "id": str(uuid4()),
            "project_id": str(project_id),
            "type": "training",
            "status": "completed",
            "payload": {
                "project_id": str(project_id),
                "dataset_id": str(dataset_id),
                "policy": "pi05",
                "model_name": "imported",
            },
        }
    )

    def _extract_to(destination: Path, *, min_free_bytes: int = 0) -> int:
        (destination / "weights.safetensors").write_text("weights")
        return 1

    safe_archive = SimpleNamespace(validate=lambda: None, extract_to=_extract_to)

    async def _create_model(model: Model) -> Model:
        return model

    with (
        patch("services.model_import_service.get_settings", return_value=settings),
        patch("services.model_import_service.DatasetService.get_dataset_by_id", AsyncMock(return_value=dataset)),
        patch("services.model_import_service.SafeZipArchive", return_value=safe_archive),
        patch("services.model_import_service.flatten_single_root_directory"),
        patch(
            "services.model_import_service.SnapshotService.create_snapshot_for_dataset",
            AsyncMock(return_value=snapshot),
        ),
        patch("services.model_import_service.JobService.create_job", AsyncMock(return_value=job)),
        patch("services.model_import_service.ModelService.create_model", AsyncMock(side_effect=_create_model)),
        patch(
            "services.model_import_service.asyncio.to_thread", AsyncMock(side_effect=lambda fn, *a, **k: fn(*a, **k))
        ),
    ):
        model = await ModelImportService().import_model_archive(
            archive_path=archive_path,
            project_id=project_id,
            dataset_id=dataset_id,
            policy="pi05",
            model_name="imported",
        )

    assert model.project_id == project_id
    assert model.dataset_id == dataset_id
    assert model.train_job_id == job.id


@pytest.mark.anyio
async def test_import_model_archive_removes_model_dir_when_snapshot_creation_fails(tmp_path: Path) -> None:
    archive_path = tmp_path / "model.zip"
    archive_path.write_bytes(b"zip")

    project_id = uuid4()
    dataset_id = uuid4()

    settings = SimpleNamespace(
        models_dir=tmp_path / "models",
        data_import_max_uncompressed_bytes=1024 * 1024,
        data_import_min_free_bytes=0,
    )

    dataset = Dataset.model_validate(
        {
            "id": str(dataset_id),
            "name": "dataset",
            "path": str(tmp_path / "dataset"),
            "default_task": "task",
            "project_id": str(project_id),
            "environment_id": str(uuid4()),
        }
    )

    created_model_dir: Path | None = None

    def _extract_to(destination: Path, *, min_free_bytes: int = 0) -> int:
        nonlocal created_model_dir
        created_model_dir = destination
        (destination / "weights.safetensors").write_text("weights")
        return 1

    safe_archive = SimpleNamespace(validate=lambda: None, extract_to=_extract_to)

    with (
        patch("services.model_import_service.get_settings", return_value=settings),
        patch("services.model_import_service.DatasetService.get_dataset_by_id", AsyncMock(return_value=dataset)),
        patch("services.model_import_service.SafeZipArchive", return_value=safe_archive),
        patch("services.model_import_service.flatten_single_root_directory"),
        patch(
            "services.model_import_service.SnapshotService.create_snapshot_for_dataset",
            AsyncMock(side_effect=InvalidArchiveError("snapshot failed")),
        ),
        patch(
            "services.model_import_service.asyncio.to_thread", AsyncMock(side_effect=lambda fn, *a, **k: fn(*a, **k))
        ),
    ):
        with pytest.raises(InvalidArchiveError, match="snapshot failed"):
            await ModelImportService().import_model_archive(
                archive_path=archive_path,
                project_id=project_id,
                dataset_id=dataset_id,
                policy="pi05",
                model_name="imported",
            )

    assert created_model_dir is not None
    assert not created_model_dir.exists()
