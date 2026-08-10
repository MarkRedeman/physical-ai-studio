# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ModelExportService."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path  # noqa: TC003
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from schemas.model import Model
from services.model_export_service import ModelExportError, ModelExportService


def _make_model(*, path: str, properties: dict | None = None) -> Model:
    return Model.model_validate(
        {
            "id": str(uuid4()),
            "name": "test-model",
            "policy": "act",
            "path": path,
            "project_id": str(uuid4()),
            "dataset_id": str(uuid4()),
            "snapshot_id": None,
            "properties": properties or {},
        }
    )


def run(coro):
    return asyncio.run(coro)


def _mock_policy(backends: list[str]) -> MagicMock:
    """Return a fake loaded policy advertising the given export backends."""
    policy = MagicMock()
    policy.get_supported_export_backends.return_value = backends
    return policy


def _patch_db(model: Model):
    """Patch get_async_db_session_ctx + ModelService to serve a fake model."""
    session = MagicMock()

    @asynccontextmanager
    async def fake_session_ctx():
        yield session

    service = MagicMock()
    service.get_model_by_id = AsyncMock(return_value=model)
    service.create_model = AsyncMock()

    ctx_patch = patch("services.model_export_service.get_async_db_session_ctx", fake_session_ctx)
    svc_patch = patch("services.model_export_service.ModelService", return_value=service)
    return ctx_patch, svc_patch, service


class TestModelExportService:
    def test_export_adds_backend_to_existing_model(self, tmp_path: Path) -> None:
        src = tmp_path / "source"
        (src / "exports" / "torch").mkdir(parents=True)
        (src / "model.ckpt").write_bytes(b"ckpt")
        (src / "exports" / "torch" / "existing.pt").write_bytes(b"torch")
        model = _make_model(path=str(src))

        def write_backend(_policy, _model, exports_dir: Path, backend) -> None:
            backend_dir = exports_dir / backend.value
            backend_dir.mkdir(parents=True)
            (backend_dir / f"{backend.value}.artifact").write_bytes(b"export")

        ctx_patch, svc_patch, service = _patch_db(model)
        with (
            ctx_patch,
            svc_patch,
            patch(
                "services.model_export_service.ModelExportService._load_policy",
                return_value=_mock_policy(["torch", "openvino"]),
            ) as load_policy,
            patch(
                "services.model_export_service.ModelExportService._export_policy",
                side_effect=write_backend,
            ) as export_policy,
            patch("services.model_export_service.ModelCompressionService._compress_openvino_dir") as compress,
        ):
            result = run(ModelExportService.export_model(model.id, backends=["openvino"]))

        load_policy.assert_called_once()
        export_policy.assert_called_once()
        compress.assert_called_once()
        service.create_model.assert_not_awaited()

        assert result is model
        assert (src / "model.ckpt").is_file()
        assert (src / "exports" / "torch" / "existing.pt").is_file()
        assert (src / "exports" / "openvino" / "openvino.artifact").is_file()

    def test_export_preserves_existing_backend_when_export_fails(self, tmp_path: Path) -> None:
        src = tmp_path / "source"
        (src / "exports" / "openvino").mkdir(parents=True)
        (src / "model.ckpt").write_bytes(b"ckpt")
        (src / "exports" / "openvino" / "existing.xml").write_text("old export")
        model = _make_model(path=str(src))

        ctx_patch, svc_patch, _ = _patch_db(model)
        with (
            ctx_patch,
            svc_patch,
            patch(
                "services.model_export_service.ModelExportService._load_policy",
                return_value=_mock_policy(["torch", "openvino"]),
            ),
            patch("services.model_export_service.ModelExportService._export_policy", side_effect=RuntimeError("boom")),
            pytest.raises(ModelExportError, match="boom"),
        ):
            run(ModelExportService.export_model(model.id, backends=["openvino"]))

        assert (src / "exports" / "openvino" / "existing.xml").read_text() == "old export"
        assert not list(src.glob(".tmp-export-*"))

    def test_export_requires_checkpoint(self, tmp_path: Path) -> None:
        src = tmp_path / "source"
        src.mkdir(parents=True)
        model = _make_model(path=str(src))

        ctx_patch, svc_patch, _ = _patch_db(model)
        with (
            ctx_patch,
            svc_patch,
            pytest.raises(ModelExportError, match="no checkpoint"),
        ):
            run(ModelExportService.export_model(model.id))

    def test_export_rejects_backend_not_supported_by_policy(self, tmp_path: Path) -> None:
        src = tmp_path / "source"
        src.mkdir(parents=True)
        (src / "model.ckpt").write_bytes(b"ckpt")
        model = _make_model(path=str(src))

        ctx_patch, svc_patch, _ = _patch_db(model)
        with (
            ctx_patch,
            svc_patch,
            patch(
                "services.model_export_service.ModelExportService._load_policy",
                return_value=_mock_policy(["torch"]),
            ),
            pytest.raises(ModelExportError, match="openvino"),
        ):
            run(ModelExportService.export_model(model.id, backends=["openvino"]))

    def test_is_lerobot_model(self) -> None:
        assert ModelExportService._is_lerobot_model(_make_model(path="/tmp", properties={"training_engine": "lerobot"}))
        assert not ModelExportService._is_lerobot_model(_make_model(path="/tmp"))
