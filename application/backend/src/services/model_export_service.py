# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Re-export trained models to their supported backends.

Creates a new, complete model record (checkpoint + ``version_0/`` logs +
fresh ``exports/{backend}``) from an existing model, for both the
physicalai-train engine and the LeRobot engine. The requested backends are
validated against the loaded policy's own supported backends. Optionally
applies NNCF INT8 weight compression to an OpenVINO export.

Self-contained: the heavy lifting is delegated to the policy export machinery
(:mod:`physicalai.export`) and :class:`ModelCompressionService`. Removing this
feature later is a one-file deletion.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, cast
from uuid import UUID, uuid4

from physicalai.export.backends import ExportBackend

from db import get_async_db_session_ctx
from schemas import Model
from services.model_compression_service import ModelCompressionService
from services.model_service import ModelService
from settings import get_settings
from training.job import CHECKPOINT_NAME

if TYPE_CHECKING:
    from physicalai.export.mixin_policy import ExportablePolicyMixin
    from physicalai.policies.base import Policy

logger = logging.getLogger(__name__)

_WEIGHTS_ONLY_POLICIES = frozenset({"pi0"})
_LEROBOT_ENGINE = "lerobot"


class ModelExportError(Exception):
    """Raised when model re-export fails."""


class ModelExportService:
    """Re-export an existing model to torch and/or OpenVINO."""

    @staticmethod
    def _is_lerobot_model(model: Model) -> bool:
        """Return True when the model was trained by the LeRobot engine."""
        properties = model.properties if isinstance(model.properties, dict) else {}
        return properties.get("training_engine") == _LEROBOT_ENGINE

    @classmethod
    def _load_policy(cls, model: Model, checkpoint: Path) -> Policy:
        """Load a trained policy from its checkpoint, engine-aware."""
        if cls._is_lerobot_model(model):
            from physicalai.policies.lerobot.export import ExportableLeRobotPolicy

            return ExportableLeRobotPolicy.load_from_checkpoint(checkpoint, map_location="cpu")

        from physicalai.policies import get_physicalai_policy_class

        policy_class = get_physicalai_policy_class(model.policy)
        kwargs: dict = {"weights_only": True} if model.policy.lower() in _WEIGHTS_ONLY_POLICIES else {}
        return policy_class.load_from_checkpoint(str(checkpoint), **kwargs)

    @staticmethod
    def _export_policy(policy: Policy, model: Model, exports_dir: Path, backends: list[ExportBackend]) -> None:
        """Export a policy to the requested backends under ``exports_dir``."""
        policy = policy.to("cpu")
        policy.eval()

        if ModelExportService._is_lerobot_model(model):
            from physicalai.policies.lerobot.export import ExportableLeRobotPolicy, export_lerobot_policy

            exportable = cast("ExportableLeRobotPolicy", policy)
            backend_args: list[str | ExportBackend] = list(backends)
            export_lerobot_policy(exportable, str(exports_dir), backends=backend_args)
            return

        exportable = cast("ExportablePolicyMixin", policy)
        for backend in backends:
            exportable.export(exports_dir / backend.value, backend=backend)

    @classmethod
    async def export_model(
        cls,
        model_id: UUID,
        name: str | None = None,
        *,
        backends: list[str | ExportBackend] | None = None,
        compress: bool = True,
    ) -> Model:
        """Re-export a model from its checkpoint to its supported backends.

        Creates a new, complete model record: the checkpoint and training logs
        are copied over, and fresh exports are produced under ``exports/``. When
        *compress* is True and OpenVINO is exported, NNCF INT8 weight
        compression is applied on top.

        Args:
            model_id: UUID of the existing model to re-export.
            name: Optional display name for the exported model. Defaults to
                ``"<original> (re-exported)"``.
            backends: Backends to export to. Defaults to torch and OpenVINO.
            compress: Whether to apply NNCF INT8 weight compression to the
                OpenVINO export after re-exporting. Defaults to True.

        Returns:
            The newly created ``Model`` record.

        Raises:
            ModelExportError: If the model has no PyTorch checkpoint, a
                requested backend is not supported by the policy, or export
                fails.
        """
        settings = get_settings()

        async with get_async_db_session_ctx() as session:
            original = await ModelService(session).get_model_by_id(model_id)

        checkpoint = Path(original.path) / CHECKPOINT_NAME
        if not checkpoint.is_file():
            msg = f"Model '{model_id}' has no checkpoint at '{checkpoint}'. Cannot re-export."
            raise ModelExportError(msg)

        selected = [ExportBackend(b) for b in (backends or [ExportBackend.TORCH, ExportBackend.OPENVINO])]

        new_model_id = uuid4()
        new_model_dir = settings.models_dir / str(new_model_id)
        new_model_dir.mkdir(parents=True, exist_ok=True)
        exports_dir = new_model_dir / "exports"

        logger.info("Re-exporting model %s to %s", model_id, [b.value for b in selected])
        try:
            policy = cls._load_policy(original, checkpoint)
            exportable = cast("ExportablePolicyMixin", policy)
            supported = {ExportBackend(b) for b in exportable.get_supported_export_backends()}
            unsupported = set(selected) - supported
            if unsupported:
                msg = f"Model does not support backend(s): {sorted(b.value for b in unsupported)}"
                raise ModelExportError(msg)
            cls._export_policy(policy, original, exports_dir, selected)

            if compress and ExportBackend.OPENVINO in selected:
                ov_dir = exports_dir / ExportBackend.OPENVINO.value
                if ov_dir.is_dir():
                    ModelCompressionService._compress_openvino_dir(ov_dir)

            # Keep the result a complete model: checkpoint + training logs, so
            # the new model can itself be re-exported or retrained.
            shutil.copy2(str(checkpoint), str(new_model_dir / CHECKPOINT_NAME))
            source_logs = Path(original.path) / "version_0"
            if source_logs.is_dir():
                shutil.copytree(str(source_logs), str(new_model_dir / "version_0"), dirs_exist_ok=True)
        except Exception as exc:
            shutil.rmtree(new_model_dir, ignore_errors=True)
            msg = f"Re-export failed for model '{model_id}': {exc}"
            raise ModelExportError(msg) from exc

        model_name = name or f"{original.name} (re-exported)"
        properties: dict[str, object] = {
            "export_tool": "physicalai-reexport",
            "source_model_id": str(original.id),
            "source_model_name": original.name,
            "backends": ",".join(b.value for b in selected),
            "compress": str(compress),
        }
        if ExportBackend.OPENVINO in selected:
            properties["precision"] = "FP16"
        if cls._is_lerobot_model(original):
            properties["training_engine"] = _LEROBOT_ENGINE

        exported = Model(
            id=new_model_id,
            project_id=original.project_id,
            dataset_id=original.dataset_id,
            path=str(new_model_dir),
            name=model_name,
            policy=original.policy,
            properties=properties,
            snapshot_id=original.snapshot_id,
            train_job_id=None,
            parent_model_id=original.id,
            version=original.version + 1,
            created_at=None,
        )

        async with get_async_db_session_ctx() as session:
            return await ModelService(session).create_model(exported)
