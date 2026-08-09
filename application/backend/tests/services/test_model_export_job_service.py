"""Unit tests for model export job submission."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from schemas.export_job import ModelExportJob, ModelExportJobPayload
from schemas.model import Model
from services.job_service import JobService

MODULE = "services.job_service"


def _session_context() -> AsyncMock:
    session = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    return session


def _make_model(*, project_id) -> Model:
    return Model.model_validate(
        {
            "id": str(uuid4()),
            "name": "source-model",
            "policy": "act",
            "path": "/tmp/source-model",
            "project_id": str(project_id),
            "dataset_id": str(uuid4()),
            "snapshot_id": None,
            "properties": {},
        }
    )


@pytest.mark.anyio
async def test_submit_model_export_job_uses_the_model_project() -> None:
    session = _session_context()
    repository = MagicMock()
    repository.save = AsyncMock(side_effect=lambda job: job)
    project_id = uuid4()
    model = _make_model(project_id=project_id)
    model_service = MagicMock()
    model_service.get_model_by_id = AsyncMock(return_value=model)

    with (
        patch(f"{MODULE}.JobRepository", return_value=repository),
        patch(f"{MODULE}.ModelService", return_value=model_service),
    ):
        job = await JobService(session).submit_model_export_job(
            ModelExportJobPayload(model_id=model.id, backends=["openvino"], compress=False)
        )

    assert isinstance(job, ModelExportJob)
    assert job.type == "model_export"
    assert job.project_id == project_id
    assert job.payload.model_id == model.id
    assert job.payload.backends == ["openvino"]
    assert job.payload.compress is False
    repository.save.assert_awaited_once()


@pytest.mark.anyio
async def test_get_pending_model_export_jobs() -> None:
    session = _session_context()
    repository = MagicMock()
    repository.get_pending_jobs_by_type = AsyncMock(return_value=[MagicMock()])

    with patch(f"{MODULE}.JobRepository", return_value=repository):
        jobs = await JobService(session).get_pending_model_export_jobs()

    assert len(jobs) == 1
    repository.get_pending_jobs_by_type.assert_awaited_once()
