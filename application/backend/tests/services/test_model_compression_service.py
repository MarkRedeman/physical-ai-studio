# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ModelCompressionService."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from schemas.model import Model
from services.model_compression_service import ModelCompressionError, ModelCompressionService


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

    ctx_patch = patch("services.model_compression_service.get_async_db_session_ctx", fake_session_ctx)
    svc_patch = patch("services.model_compression_service.ModelService", return_value=service)
    return ctx_patch, svc_patch, created


class TestFindMainOpenvinoModel:
    def test_skips_tokenizer(self, tmp_path: Path) -> None:
        (tmp_path / "tokenizer.xml").write_text("<xml/>")
        (tmp_path / "tokenizer.bin").write_bytes(b"t")
        (tmp_path / "model.xml").write_text("<xml/>")
        (tmp_path / "model.bin").write_bytes(b"m")

        assert ModelCompressionService._find_main_openvino_model(tmp_path).name == "model.xml"

    def test_raises_without_weights(self, tmp_path: Path) -> None:
        (tmp_path / "model.xml").write_text("<xml/>")
        with pytest.raises(ModelCompressionError, match="No OpenVINO IR"):
            ModelCompressionService._find_main_openvino_model(tmp_path)


class TestCompressModel:
    def test_compress_creates_new_model(self, tmp_path: Path, settings) -> None:
        src = tmp_path / "source"
        ov_dir = src / "exports" / "openvino"
        ov_dir.mkdir(parents=True)
        (ov_dir / "model.xml").write_text("<xml/>")
        (ov_dir / "model.bin").write_bytes(b"w")
        (ov_dir / "manifest.json").write_text("{}")
        model = _make_model(path=str(src), properties={"training_engine": "lerobot"})

        ctx_patch, svc_patch, created = _patch_db(model)
        with (
            ctx_patch,
            svc_patch,
            patch("services.model_compression_service.get_settings", return_value=settings),
            patch("services.model_compression_service.ModelCompressionService._compress_openvino_dir") as compress,
        ):
            result = run(ModelCompressionService.compress_model(model.id))

        compress.assert_called_once()
        assert created and created[0].id == result.id
        new_ov = Path(result.path) / "exports" / "openvino"
        assert (new_ov / "model.xml").is_file()
        assert (new_ov / "model.bin").is_file()
        assert (new_ov / "manifest.json").is_file()
        assert result.parent_model_id == model.id
        assert result.properties["compression"] == "INT8_SYM"
        assert result.properties["training_engine"] == "lerobot"

    def test_compress_requires_openvino_export(self, tmp_path: Path, settings) -> None:
        src = tmp_path / "source"
        src.mkdir(parents=True)
        model = _make_model(path=str(src))

        ctx_patch, svc_patch, _ = _patch_db(model)
        with (
            ctx_patch,
            svc_patch,
            patch("services.model_compression_service.get_settings", return_value=settings),
        ):
            with pytest.raises(ModelCompressionError, match="no OpenVINO export"):
                run(ModelCompressionService.compress_model(model.id))

    def test_compress_requires_nncf(self, tmp_path: Path, settings) -> None:
        src = tmp_path / "source"
        ov_dir = src / "exports" / "openvino"
        ov_dir.mkdir(parents=True)
        (ov_dir / "model.xml").write_text("<xml/>")
        (ov_dir / "model.bin").write_bytes(b"w")
        model = _make_model(path=str(src))

        ctx_patch, svc_patch, _ = _patch_db(model)
        with (
            ctx_patch,
            svc_patch,
            patch("services.model_compression_service.get_settings", return_value=settings),
            patch("builtins.__import__", side_effect=ImportError("nncf")),
        ):
            with pytest.raises(ModelCompressionError, match="nncf"):
                run(ModelCompressionService.compress_model(model.id))
