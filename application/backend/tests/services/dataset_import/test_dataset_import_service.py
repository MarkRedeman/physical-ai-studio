"""Focused unit tests for DatasetImportService step/ownership validation logic.

These tests cover the three public methods that enforce ownership and step
guards without requiring a real database.  Each test patches only the minimal
surface area needed:

* ``db.get_async_db_session_ctx`` - returns a lightweight async context manager
  that yields a sentinel session object (never used directly, because
  ``JobRepository`` is also patched).
* ``services.dataset_import.service.JobRepository`` - replaced by a
  ``_StubJobRepository`` that returns pre-built ``DatasetImportJob`` fixtures.
* ``services.archive_safety.cleanup_staged_archive`` - no-op to avoid
  touching the filesystem.
* ``services.dataset_import.service.EnvironmentService.get_environment_by_id``
  - no-op ``AsyncMock`` for ``finalize_dataset_import_job`` tests that reach
  the environment lookup.
"""

from __future__ import annotations

import contextlib
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from exceptions import InvalidJobStateError, ResourceNotFoundError
from schemas.base_job import JobStatus
from schemas.dataset_import_job import DatasetImportFinalizeInput, DatasetImportJobPayload, ImportStep
from schemas.job import DatasetImportJob
from services.dataset_import.service import DatasetImportService

# ---------------------------------------------------------------------------
# Helpers / shared infrastructure
# ---------------------------------------------------------------------------

_STAGING_ID = "a" * 8 + "-" + "b" * 4 + "-" + "c" * 4 + "-" + "d" * 4 + "-" + "e" * 12


def _make_job(
    *,
    project_id=None,
    step: ImportStep = ImportStep.AWAITING_UPLOAD,
    archive_staging_id: str = _STAGING_ID,
) -> DatasetImportJob:
    """Build a minimal ``DatasetImportJob`` with the given step."""
    return DatasetImportJob(
        id=uuid4(),
        project_id=project_id or uuid4(),
        status=JobStatus.PENDING,
        payload=DatasetImportJobPayload(
            step=step,
            archive_staging_id=archive_staging_id,
        ),
    )


class _StubJobRepository:
    """Minimal stub for JobRepository covering the paths exercised below."""

    def __init__(self, job: DatasetImportJob | None) -> None:
        self._job = job
        self.updated: list[tuple] = []

    async def get_by_id(self, job_id):
        return self._job

    async def update(self, job, updates):
        self.updated.append((job, updates))
        # Return a copy with the updates applied so callers get a valid Job
        return job.model_copy(update=updates, deep=True)


@contextlib.asynccontextmanager
async def _fake_db_session_ctx():
    """Fake async context manager that yields a sentinel session."""
    yield MagicMock()


def _patch_repo(stub: _StubJobRepository):
    """Return a context manager that injects *stub* as JobRepository."""
    return patch(
        "services.dataset_import.service.JobRepository",
        return_value=stub,
    )


def _patch_db():
    """Return a context manager that replaces the DB session factory."""
    return patch(
        "services.dataset_import.service.get_async_db_session_ctx",
        return_value=_fake_db_session_ctx(),
    )


def _patch_cleanup():
    """Return a context manager that suppresses archive cleanup."""
    return patch("services.dataset_import.service.cleanup_staged_archive")


def _patch_env_service():
    """Return a context manager that stubs EnvironmentService."""
    return patch(
        "services.dataset_import.service.EnvironmentService.get_environment_by_id",
        new=AsyncMock(return_value=MagicMock()),
    )


# ---------------------------------------------------------------------------
# attach_dataset_import_archive
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_attach_raises_resource_not_found_when_job_not_found() -> None:
    """Ownership check: job is None -> ResourceNotFoundError."""
    project_id = uuid4()
    stub = _StubJobRepository(job=None)

    with _patch_db(), _patch_repo(stub), pytest.raises(ResourceNotFoundError):
        await DatasetImportService.attach_dataset_import_archive(
            project_id=project_id,
            job_id=uuid4(),
            uploaded_archive_name="dataset.zip",
        )


@pytest.mark.anyio
async def test_attach_raises_resource_not_found_when_project_mismatch() -> None:
    """Ownership check: job.project_id != caller project_id -> ResourceNotFoundError."""
    project_id = uuid4()
    other_project_id = uuid4()
    job = _make_job(project_id=other_project_id, step=ImportStep.AWAITING_UPLOAD)
    stub = _StubJobRepository(job=job)

    with _patch_db(), _patch_repo(stub), pytest.raises(ResourceNotFoundError):
        await DatasetImportService.attach_dataset_import_archive(
            project_id=project_id,
            job_id=job.id,
            uploaded_archive_name="dataset.zip",
        )


@pytest.mark.anyio
async def test_attach_raises_invalid_job_state_when_wrong_step() -> None:
    """Step guard: job not in AWAITING_UPLOAD -> InvalidJobStateError."""
    project_id = uuid4()
    job = _make_job(project_id=project_id, step=ImportStep.UPLOADED)
    stub = _StubJobRepository(job=job)

    with _patch_db(), _patch_repo(stub), pytest.raises(InvalidJobStateError):
        await DatasetImportService.attach_dataset_import_archive(
            project_id=project_id,
            job_id=job.id,
            uploaded_archive_name="dataset.zip",
        )


@pytest.mark.anyio
async def test_attach_succeeds_when_step_is_awaiting_upload() -> None:
    """Happy path: AWAITING_UPLOAD step -> archive attached, step advanced to UPLOADED."""
    project_id = uuid4()
    job = _make_job(project_id=project_id, step=ImportStep.AWAITING_UPLOAD)
    stub = _StubJobRepository(job=job)

    with _patch_db(), _patch_repo(stub):
        result = await DatasetImportService.attach_dataset_import_archive(
            project_id=project_id,
            job_id=job.id,
            uploaded_archive_name="dataset.zip",
        )

    assert result is not None
    assert len(stub.updated) == 1
    _, updates = stub.updated[0]
    assert updates["status"] == JobStatus.PENDING
    # Payload in updates is a dict after model_dump
    assert updates["payload"]["step"] == ImportStep.UPLOADED
    assert updates["payload"]["archive_staging_id"] == _STAGING_ID
    assert updates["payload"]["uploaded_archive_name"] == "dataset.zip"


# ---------------------------------------------------------------------------
# finalize_dataset_import_job
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_finalize_raises_resource_not_found_when_job_not_found() -> None:
    """Ownership check: job is None -> ResourceNotFoundError."""
    project_id = uuid4()
    stub = _StubJobRepository(job=None)
    finalize_input = DatasetImportFinalizeInput(dataset_name="test", environment_id=uuid4())

    with _patch_db(), _patch_repo(stub), pytest.raises(ResourceNotFoundError):
        await DatasetImportService.finalize_dataset_import_job(
            project_id=project_id,
            job_id=uuid4(),
            finalize_input=finalize_input,
        )


@pytest.mark.anyio
async def test_finalize_raises_resource_not_found_when_project_mismatch() -> None:
    """Ownership check: job.project_id != caller project_id -> ResourceNotFoundError."""
    project_id = uuid4()
    other_project_id = uuid4()
    job = _make_job(project_id=other_project_id, step=ImportStep.WAITING_FOR_USER_INPUT)
    stub = _StubJobRepository(job=job)
    finalize_input = DatasetImportFinalizeInput(dataset_name="test", environment_id=uuid4())

    with _patch_db(), _patch_repo(stub), pytest.raises(ResourceNotFoundError):
        await DatasetImportService.finalize_dataset_import_job(
            project_id=project_id,
            job_id=job.id,
            finalize_input=finalize_input,
        )


@pytest.mark.anyio
async def test_finalize_raises_invalid_job_state_when_wrong_step() -> None:
    """Step guard: step is not WAITING_FOR_USER_INPUT -> InvalidJobStateError."""
    project_id = uuid4()
    job = _make_job(project_id=project_id, step=ImportStep.AWAITING_UPLOAD)
    stub = _StubJobRepository(job=job)
    finalize_input = DatasetImportFinalizeInput(dataset_name="test", environment_id=uuid4())

    with _patch_db(), _patch_repo(stub), pytest.raises(InvalidJobStateError):
        await DatasetImportService.finalize_dataset_import_job(
            project_id=project_id,
            job_id=job.id,
            finalize_input=finalize_input,
        )


@pytest.mark.parametrize(
    "bad_step",
    [
        ImportStep.AWAITING_UPLOAD,
        ImportStep.UPLOADED,
        ImportStep.DETECTING_SOURCE,
        ImportStep.GENERATING_DRAFT_MANIFEST,
        ImportStep.READY_TO_COMMIT,
        ImportStep.REGISTERING_RESOURCE,
        ImportStep.IMPORTING_RESOURCE,
        ImportStep.COMPLETED,
    ],
)
@pytest.mark.anyio
async def test_finalize_raises_invalid_job_state_for_every_non_waiting_step(bad_step: ImportStep) -> None:
    """Exhaustive check: all steps except WAITING_FOR_USER_INPUT raise InvalidJobStateError."""
    project_id = uuid4()
    job = _make_job(project_id=project_id, step=bad_step)
    stub = _StubJobRepository(job=job)
    finalize_input = DatasetImportFinalizeInput(dataset_name="test", environment_id=uuid4())

    with _patch_db(), _patch_repo(stub), pytest.raises(InvalidJobStateError):
        await DatasetImportService.finalize_dataset_import_job(
            project_id=project_id,
            job_id=job.id,
            finalize_input=finalize_input,
        )


@pytest.mark.anyio
async def test_finalize_succeeds_when_step_is_waiting_for_user_input() -> None:
    """Happy path: WAITING_FOR_USER_INPUT -> step advanced to READY_TO_COMMIT."""
    project_id = uuid4()
    environment_id = uuid4()
    job = _make_job(project_id=project_id, step=ImportStep.WAITING_FOR_USER_INPUT)
    stub = _StubJobRepository(job=job)
    finalize_input = DatasetImportFinalizeInput(dataset_name="my-dataset", environment_id=environment_id)

    with _patch_db(), _patch_repo(stub), _patch_env_service():
        result = await DatasetImportService.finalize_dataset_import_job(
            project_id=project_id,
            job_id=job.id,
            finalize_input=finalize_input,
        )

    assert result is not None
    assert len(stub.updated) == 1
    _, updates = stub.updated[0]
    assert updates["status"] == JobStatus.PENDING
    assert updates["payload"]["step"] == ImportStep.READY_TO_COMMIT


# ---------------------------------------------------------------------------
# cancel_dataset_import_job
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_cancel_raises_resource_not_found_when_job_not_found() -> None:
    """Ownership check: job is None -> ResourceNotFoundError."""
    project_id = uuid4()
    stub = _StubJobRepository(job=None)

    with _patch_db(), _patch_repo(stub), _patch_cleanup(), pytest.raises(ResourceNotFoundError):
        await DatasetImportService.cancel_dataset_import_job(
            project_id=project_id,
            job_id=uuid4(),
        )


@pytest.mark.anyio
async def test_cancel_raises_resource_not_found_when_project_mismatch() -> None:
    """Ownership check: job.project_id != caller project_id -> ResourceNotFoundError."""
    project_id = uuid4()
    other_project_id = uuid4()
    job = _make_job(project_id=other_project_id, step=ImportStep.AWAITING_UPLOAD)
    stub = _StubJobRepository(job=job)

    with _patch_db(), _patch_repo(stub), _patch_cleanup(), pytest.raises(ResourceNotFoundError):
        await DatasetImportService.cancel_dataset_import_job(
            project_id=project_id,
            job_id=job.id,
        )


@pytest.mark.parametrize(
    "blocked_step",
    [
        ImportStep.IMPORTING_RESOURCE,
        ImportStep.REGISTERING_RESOURCE,
    ],
)
@pytest.mark.anyio
async def test_cancel_raises_invalid_job_state_for_blocked_steps(blocked_step: ImportStep) -> None:
    """Step guard: steps in blocked set raise InvalidJobStateError."""
    project_id = uuid4()
    job = _make_job(project_id=project_id, step=blocked_step)
    stub = _StubJobRepository(job=job)

    with _patch_db(), _patch_repo(stub), _patch_cleanup(), pytest.raises(InvalidJobStateError):
        await DatasetImportService.cancel_dataset_import_job(
            project_id=project_id,
            job_id=job.id,
        )


@pytest.mark.parametrize(
    "allowed_step",
    [
        ImportStep.AWAITING_UPLOAD,
        ImportStep.UPLOADED,
        ImportStep.DETECTING_SOURCE,
        ImportStep.GENERATING_DRAFT_MANIFEST,
        ImportStep.WAITING_FOR_USER_INPUT,
        ImportStep.READY_TO_COMMIT,
    ],
)
@pytest.mark.anyio
async def test_cancel_succeeds_for_non_blocked_steps_with_staging_id(allowed_step: ImportStep) -> None:
    """Happy path: non-blocked steps with archive_staging_id -> job marked CANCELED, archive cleaned up."""
    project_id = uuid4()
    job = _make_job(project_id=project_id, step=allowed_step, archive_staging_id=_STAGING_ID)
    stub = _StubJobRepository(job=job)

    with _patch_db(), _patch_repo(stub), _patch_cleanup() as mock_cleanup:
        result = await DatasetImportService.cancel_dataset_import_job(
            project_id=project_id,
            job_id=job.id,
        )

    assert result is not None
    assert len(stub.updated) == 1
    _, updates = stub.updated[0]
    assert updates["status"] == JobStatus.CANCELED
    # resolve_payload_archive_path derives path from staging id -> cleanup called with a Path
    from pathlib import Path

    called_path = mock_cleanup.call_args[0][0]
    assert isinstance(called_path, Path)
    assert called_path.name == f"{_STAGING_ID}.zip"


# ---------------------------------------------------------------------------
# Schema shape and validation_report optionality
# ---------------------------------------------------------------------------


def test_dataset_manifest_slim_shape_no_legacy_fields() -> None:
    """DatasetManifest no longer carries manifest_version, resource_type, integrity,
    warnings, or missing_fields."""
    from schemas.dataset_import_job import DatasetImportSource, DatasetManifest, DatasetManifestStatistics

    manifest = DatasetManifest(
        source_type=DatasetImportSource.UNKNOWN,
        suggested_name="test",
        statistics=DatasetManifestStatistics(episode_count=5, frame_count=150),
    )
    dumped = manifest.model_dump(by_alias=True)

    assert "manifest_version" not in dumped
    assert "resource_type" not in dumped
    assert "integrity" not in dumped
    assert "warnings" not in dumped
    assert "missing_fields" not in dumped
    # dataset_schema must still be present
    assert "dataset_schema" in dumped


def test_dataset_manifest_source_slim_shape_no_legacy_fields() -> None:
    """DatasetManifest.source_type is a direct field; no nested DatasetManifestSource wrapper."""
    from schemas.dataset_import_job import DatasetImportSource, DatasetManifest

    manifest = DatasetManifest(source_type=DatasetImportSource.LEROBOT_V3)
    dumped = manifest.model_dump()

    assert "source_format_version" not in dumped
    assert "source_identifier" not in dumped
    assert "original_dataset_uuid" not in dumped
    assert dumped["source_type"] == DatasetImportSource.LEROBOT_V3


def test_dataset_manifest_identity_slim_shape_no_default_task() -> None:
    """DatasetManifest.suggested_name is a direct field; no nested DatasetManifestIdentity wrapper."""
    from schemas.dataset_import_job import DatasetManifest

    manifest = DatasetManifest(suggested_name="my-dataset")
    dumped = manifest.model_dump()

    assert "default_task" not in dumped
    assert dumped["suggested_name"] == "my-dataset"


def test_finalize_input_slim_shape_no_user_overrides() -> None:
    """DatasetImportFinalizeInput no longer carries user_overrides."""
    env_id = uuid4()
    finalize_input = DatasetImportFinalizeInput(dataset_name="test", environment_id=env_id)
    dumped = finalize_input.model_dump(mode="json")

    assert "user_overrides" not in dumped
    assert dumped["dataset_name"] == "test"


def test_import_step_includes_completed() -> None:
    """ImportStep now includes COMPLETED as a terminal enum value."""
    from schemas.dataset_import_job import ImportStep

    assert hasattr(ImportStep, "COMPLETED")
    assert ImportStep.COMPLETED == "completed"


# ---------------------------------------------------------------------------
# validation_report optionality
# ---------------------------------------------------------------------------


def test_dataset_import_job_payload_validation_report_defaults_to_none() -> None:
    """A freshly created DatasetImportJobPayload has validation_report=None."""
    payload = DatasetImportJobPayload(step=ImportStep.AWAITING_UPLOAD, archive_staging_id=_STAGING_ID)
    assert payload.validation_report is None


def test_dataset_import_job_payload_validation_report_is_none_when_no_issues() -> None:
    """validation_report should be set to None when there are no issues (waiting for user input)."""
    from schemas.dataset_import_job import ImportValidationReport

    payload = DatasetImportJobPayload(step=ImportStep.WAITING_FOR_USER_INPUT, archive_staging_id=_STAGING_ID)
    # Simulate the worker behaviour: clean report (no messages) -> None
    report = ImportValidationReport()
    has_issues = bool(report.messages)
    payload.validation_report = report if has_issues else None

    assert payload.validation_report is None


def test_dataset_import_job_payload_validation_report_is_set_when_issues_present() -> None:
    """validation_report is preserved when there are messages (e.g. errors)."""
    from schemas.dataset_import_job import ImportValidationReport, ImportValidationSeverity

    payload = DatasetImportJobPayload(step=ImportStep.WAITING_FOR_USER_INPUT, archive_staging_id=_STAGING_ID)
    report = ImportValidationReport()
    report.add_error("env is required")
    has_issues = bool(report.messages)
    payload.validation_report = report if has_issues else None

    assert payload.validation_report is not None
    assert len(payload.validation_report.messages) == 1
    assert payload.validation_report.messages[0].severity == ImportValidationSeverity.ERROR


# ---------------------------------------------------------------------------
# Step consistency: completed job must have COMPLETED step
# ---------------------------------------------------------------------------


def test_completed_payload_step_is_completed() -> None:
    """A completed import job payload must use ImportStep.COMPLETED, not REGISTERING_RESOURCE."""
    from schemas.dataset_import_job import ImportStep

    payload = DatasetImportJobPayload(step=ImportStep.COMPLETED, archive_staging_id=_STAGING_ID)
    assert payload.step == ImportStep.COMPLETED
    assert payload.step != ImportStep.REGISTERING_RESOURCE


# ---------------------------------------------------------------------------
# end_time consistency: pending/running updates must not set end_time
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_attach_archive_does_not_set_end_time_for_pending_status() -> None:
    """Transitioning to PENDING (attach archive) must not populate end_time."""
    project_id = uuid4()
    job = _make_job(project_id=project_id, step=ImportStep.AWAITING_UPLOAD)
    stub = _StubJobRepository(job=job)

    with _patch_db(), _patch_repo(stub):
        await DatasetImportService.attach_dataset_import_archive(
            project_id=project_id,
            job_id=job.id,
            uploaded_archive_name="dataset.zip",
        )

    _, updates = stub.updated[0]
    assert "end_time" not in updates


@pytest.mark.anyio
async def test_finalize_does_not_set_end_time_for_pending_status() -> None:
    """Transitioning to PENDING (finalize) must not populate end_time."""
    project_id = uuid4()
    environment_id = uuid4()
    job = _make_job(project_id=project_id, step=ImportStep.WAITING_FOR_USER_INPUT)
    stub = _StubJobRepository(job=job)
    finalize_input = DatasetImportFinalizeInput(dataset_name="my-dataset", environment_id=environment_id)

    with _patch_db(), _patch_repo(stub), _patch_env_service():
        await DatasetImportService.finalize_dataset_import_job(
            project_id=project_id,
            job_id=job.id,
            finalize_input=finalize_input,
        )

    _, updates = stub.updated[0]
    assert "end_time" not in updates


@pytest.mark.anyio
async def test_cancel_sets_end_time() -> None:
    """Canceling an import job (terminal status CANCELED) must set end_time."""
    project_id = uuid4()
    job = _make_job(project_id=project_id, step=ImportStep.AWAITING_UPLOAD, archive_staging_id=_STAGING_ID)
    stub = _StubJobRepository(job=job)

    with _patch_db(), _patch_repo(stub), _patch_cleanup():
        await DatasetImportService.cancel_dataset_import_job(
            project_id=project_id,
            job_id=job.id,
        )

    _, updates = stub.updated[0]
    assert "end_time" in updates
    assert updates["end_time"] is not None
