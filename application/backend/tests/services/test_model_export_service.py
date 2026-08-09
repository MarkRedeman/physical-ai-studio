# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ModelExportService."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
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


@pytest.fixture
def settings(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(models_dir=tmp_path / "models")


def run(coro):
    return asyncio.run(coro)


def _patch_db(model: Model):
    """Patch get_async_db_session_ctx + ModelService to serve a fake model."""
    session = MagicMock()

    @asynccontextmanager
    async def fake_session_ctx():
        yield session

    created: list[Model] = []

    def _create(model: Model) -> Model:
        created.append(model)
        return model

    service = MagicMock()
    service.get_model_by_id = AsyncMock(return_value=model)
    service.create_model = AsyncMock(side_effect=_create)

    ctx_patch = patch("services.model_export_service.get_async_db_session_ctx", fake_session_ctx)
    svc_patch = patch("services.model_export_service.ModelService", return_value=service)
    return ctx_patch, svc_patch, created


class TestModelExportService:
    def test_export_creates_complete_model(self, tmp_path: Path, settings) -> None:
        src = tmp_path / "source"
        (src / "version_0").mkdir(parents=True)
        (src / "model.ckpt").write_bytes(b"ckpt")
        (src / "version_0" / "metrics.csv").write_text("epoch,step\n")
        model = _make_model(path=str(src))

        ctx_patch, svc_patch, created = _patch_db(model)
        with (
            ctx_patch,
            svc_patch,
            patch("services.model_export_service.get_settings", return_value=settings),
            patch("services.model_export_service.ModelExportService._load_policy") as load_policy,
            patch(
                "services.model_export_service.ModelExportService._export_policy",
                side_effect=lambda _policy, _model, exports_dir, backends: (exports_dir / "openvino").mkdir(
                    parents=True, exist_ok=True
                ),
            ) as export_policy,
            patch("services.model_export_service.ModelCompressionService._compress_openvino_dir") as compress,
        ):
            result = run(ModelExportService.export_model(model.id))

        load_policy.assert_called_once()
        export_policy.assert_called_once()
        compress.assert_called_once()

        assert created and created[0].id == result.id
        new_dir = Path(result.path)
        assert (new_dir / "model.ckpt").is_file()
        assert (new_dir / "version_0" / "metrics.csv").is_file()
        assert result.parent_model_id == model.id
        assert result.version == model.version + 1
        assert result.properties["source_model_id"] == str(model.id)
        assert "training_engine" not in result.properties

    def test_export_preserves_training_engine_for_lerobot(self, tmp_path: Path, settings) -> None:
        src = tmp_path / "source"
        src.mkdir(parents=True)
        (src / "model.ckpt").write_bytes(b"ckpt")
        model = _make_model(path=str(src), properties={"training_engine": "lerobot"})

        ctx_patch, svc_patch, created = _patch_db(model)
        with (
            ctx_patch,
            svc_patch,
            patch("services.model_export_service.get_settings", return_value=settings),
            patch("services.model_export_service.ModelExportService._load_policy"),
            patch("services.model_export_service.ModelExportService._export_policy"),
            patch("services.model_export_service.ModelCompressionService._compress_openvino_dir"),
        ):
            result = run(ModelExportService.export_model(model.id))

        assert result.properties["training_engine"] == "lerobot"

    def test_export_requires_checkpoint(self, tmp_path: Path, settings) -> None:
        src = tmp_path / "source"
        src.mkdir(parents=True)
        model = _make_model(path=str(src))

        ctx_patch, svc_patch, _ = _patch_db(model)
        with (
            ctx_patch,
            svc_patch,
            patch("services.model_export_service.get_settings", return_value=settings),
            pytest.raises(ModelExportError, match="no checkpoint"),
        ):
            run(ModelExportService.export_model(model.id))

    def test_export_rejects_unsupported_backend(self, tmp_path: Path, settings) -> None:
        src = tmp_path / "source"
        src.mkdir(parents=True)
        (src / "model.ckpt").write_bytes(b"ckpt")
        model = _make_model(path=str(src))

        ctx_patch, svc_patch, _ = _patch_db(model)
        with (
            ctx_patch,
            svc_patch,
            patch("services.model_export_service.get_settings", return_value=settings),
            pytest.raises(ModelExportError, match="onnx"),
        ):
            run(ModelExportService.export_model(model.id, backends=["onnx"]))

    def test_export_cleans_up_on_failure(self, tmp_path: Path, settings) -> None:
        src = tmp_path / "source"
        src.mkdir(parents=True)
        (src / "model.ckpt").write_bytes(b"ckpt")
        model = _make_model(path=str(src))

        ctx_patch, svc_patch, _ = _patch_db(model)
        with (
            ctx_patch,
            svc_patch,
            patch("services.model_export_service.get_settings", return_value=settings),
            patch("services.model_export_service.ModelExportService._load_policy", side_effect=RuntimeError("boom")),
            pytest.raises(ModelExportError, match="boom"),
        ):
            run(ModelExportService.export_model(model.id))

        leftovers = list((settings.models_dir).iterdir()) if settings.models_dir.exists() else []
        assert leftovers == []

    def test_is_lerobot_model(self) -> None:
        assert ModelExportService._is_lerobot_model(_make_model(path="/tmp", properties={"training_engine": "lerobot"}))
        assert not ModelExportService._is_lerobot_model(_make_model(path="/tmp"))
