# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""NNCF weight compression for exported OpenVINO models.

Self-contained: this service only reads an existing ``exports/openvino/``
directory and writes a compressed copy, so it works for any model that has an
OpenVINO export (physicalai or LeRobot engine). It is easy to remove if the
compression feature is dropped.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from uuid import UUID, uuid4

from db import get_async_db_session_ctx
from schemas import Model
from services.model_service import ModelService
from settings import get_settings

logger = logging.getLogger(__name__)


class ModelCompressionError(Exception):
    """Raised when model compression fails."""


class ModelCompressionService:
    """Compress an exported OpenVINO model's weights to INT8 with NNCF."""

    @staticmethod
    def _find_main_openvino_model(openvino_dir: Path) -> Path:
        """Find the main OpenVINO IR model with weights, skipping the tokenizer."""
        for f in sorted(openvino_dir.glob("*.xml")):
            if f.stem.lower() == "tokenizer":
                continue
            if f.with_suffix(".bin").is_file():
                return f
        msg = f"No OpenVINO IR with weights (.xml + .bin) found in '{openvino_dir}'"
        raise ModelCompressionError(msg)

    @staticmethod
    def _copy_supporting_files(openvino_dir: Path, new_ov_dir: Path) -> None:
        """Copy tokenizer and manifest files required by InferenceModel."""
        tokenizer_xml = openvino_dir / "tokenizer.xml"
        if tokenizer_xml.is_file():
            shutil.copy2(str(tokenizer_xml), str(new_ov_dir / "tokenizer.xml"))
            tokenizer_bin = openvino_dir / "tokenizer.bin"
            if tokenizer_bin.is_file():
                shutil.copy2(str(tokenizer_bin), str(new_ov_dir / "tokenizer.bin"))
        for manifest_name in ("manifest.json", "metadata.yaml"):
            src = openvino_dir / manifest_name
            if src.is_file():
                shutil.copy2(str(src), str(new_ov_dir / manifest_name))

    @classmethod
    def _compress_openvino_dir(cls, openvino_dir: Path) -> None:
        """Compress an OpenVINO IR directory's weights to INT8_SYM in place.

        Replaces the main ``*.xml``/``*.bin`` pair with NNCF INT8_SYM compressed
        weights, keeping tokenizer and manifest files untouched.

        Raises:
            ModelCompressionError: If NNCF is not installed or no OpenVINO IR
                with weights is found.
        """
        try:
            import nncf
        except ImportError as e:
            msg = "nncf is required for weight compression. Install with: pip install nncf>=2.7.0"
            raise ModelCompressionError(msg) from e

        import openvino
        import openvino_tokenizers  # noqa: F401  # registers custom ops (e.g. SpecialTokensSplit)

        src_xml = cls._find_main_openvino_model(openvino_dir)
        src_bin = src_xml.with_suffix(".bin")

        logger.info("Compressing weights to INT8_SYM: %s", src_xml)
        core = openvino.Core()
        ov_model = core.read_model(str(src_xml))
        compressed_model = nncf.compress_weights(ov_model, mode=nncf.CompressWeightsMode.INT8_SYM)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_xml = Path(tmp_dir) / src_xml.name
            openvino.save_model(compressed_model, str(tmp_xml))
            del compressed_model, ov_model, core

            src_xml.unlink()
            if src_bin.exists():
                src_bin.unlink()

            tmp_bin = tmp_xml.with_suffix(".bin")
            shutil.move(str(tmp_xml), str(src_xml))
            shutil.move(str(tmp_bin), str(src_bin))

        logger.info("INT8_SYM weight compression complete: %s", src_xml)

    @classmethod
    async def compress_model(cls, model_id: UUID, name: str | None = None) -> Model:
        """Compress an exported OpenVINO model's weights to INT8 using NNCF.

        Creates a new model record with compressed weights, linked to the
        original via ``parent_model_id``. Only the ``exports/openvino/``
        directory is included in the new model directory (checkpoints and
        other export backends are omitted to save space).

        Args:
            model_id: UUID of the existing model to compress.
            name: Optional display name for the compressed model. Defaults to
                ``"<original> (INT8)"``.

        Returns:
            The newly created ``Model`` record.

        Raises:
            ModelCompressionError: If the model has no OpenVINO export or if
                NNCF is not installed.
        """
        settings = get_settings()

        async with get_async_db_session_ctx() as session:
            original = await ModelService(session).get_model_by_id(model_id)

        openvino_dir = Path(original.path) / "exports" / "openvino"
        if not openvino_dir.is_dir():
            msg = f"Model '{model_id}' has no OpenVINO export at '{openvino_dir}'. Export the model to OpenVINO first."
            raise ModelCompressionError(msg)

        new_model_id = uuid4()
        new_model_dir = settings.models_dir / str(new_model_id)
        new_ov_dir = new_model_dir / "exports" / "openvino"
        new_ov_dir.mkdir(parents=True, exist_ok=True)

        for f in openvino_dir.iterdir():
            if f.is_file():
                shutil.copy2(str(f), str(new_ov_dir / f.name))

        cls._compress_openvino_dir(new_ov_dir)

        model_name = name or f"{original.name} (INT8)"
        properties: dict[str, str] = {
            "compression": "INT8_SYM",
            "compression_tool": "nncf",
            "source_model_id": str(original.id),
            "source_model_name": original.name,
        }
        if isinstance(original.properties, dict) and original.properties.get("training_engine"):
            properties["training_engine"] = str(original.properties["training_engine"])

        compressed = Model(
            id=new_model_id,
            project_id=original.project_id,
            dataset_id=original.dataset_id,
            path=str(new_model_dir),
            name=model_name,
            policy=original.policy,
            properties=properties,
            snapshot_id=None,
            train_job_id=None,
            parent_model_id=original.id,
            version=original.version + 1,
            created_at=None,
        )

        async with get_async_db_session_ctx() as session:
            return await ModelService(session).create_model(compressed)
