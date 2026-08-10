# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Add supported backend exports to an existing trained model.

Exports are written under the model's existing ``exports/{backend}`` directory
for both the physicalai-train engine and the LeRobot engine. The requested
backends are validated against the loaded policy's own supported backends.
Optionally applies NNCF INT8 weight compression to an OpenVINO export.

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
    """Add supported backend exports to an existing model."""

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
    def _export_policy(policy: Policy, model: Model, exports_dir: Path, backend: ExportBackend) -> None:
        """Export one backend under ``exports_dir``."""
        policy = policy.to("cpu")
        policy.eval()

        if ModelExportService._is_lerobot_model(model):
            from physicalai.policies.lerobot.export import ExportableLeRobotPolicy, export_lerobot_policy

            exportable = cast("ExportableLeRobotPolicy", policy)
            export_lerobot_policy(exportable, str(exports_dir), backends=[backend])
            return

        exportable = cast("ExportablePolicyMixin", policy)
        exportable.export(exports_dir / backend.value, backend=backend)

    @classmethod
    async def export_model(
        cls,
        model_id: UUID,
        *,
        backends: list[str | ExportBackend] | None = None,
        compress: bool = True,
    ) -> Model:
        """Re-export a model from its checkpoint to its supported backends.

        Each requested backend is produced in a temporary directory then moved
        atomically into the existing model's ``exports/`` directory. When
        *compress* is True and OpenVINO is exported, NNCF INT8 weight
        compression is applied on top.

        Args:
            model_id: UUID of the existing model to re-export.
            backends: Backends to export to. Defaults to torch and OpenVINO.
            compress: Whether to apply NNCF INT8 weight compression to the
                OpenVINO export after re-exporting. Defaults to True.

        Returns:
            The existing ``Model`` record with new exports on disk.

        Raises:
            ModelExportError: If the model has no PyTorch checkpoint, a
                requested backend is not supported by the policy, or export
                fails.
        """
        async with get_async_db_session_ctx() as session:
            original = await ModelService(session).get_model_by_id(model_id)

        checkpoint = Path(original.path) / CHECKPOINT_NAME
        if not checkpoint.is_file():
            msg = f"Model '{model_id}' has no checkpoint at '{checkpoint}'. Cannot re-export."
            raise ModelExportError(msg)

        selected = [ExportBackend(b) for b in (backends or [ExportBackend.TORCH, ExportBackend.OPENVINO])]

        exports_dir = Path(original.path) / "exports"
        exports_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Re-exporting model %s to %s", model_id, [b.value for b in selected])
        try:
            policy = cls._load_policy(original, checkpoint)
            exportable = cast("ExportablePolicyMixin", policy)
            supported = {ExportBackend(b) for b in exportable.get_supported_export_backends()}
            unsupported = set(selected) - supported
            if unsupported:
                msg = f"Model does not support backend(s): {sorted(b.value for b in unsupported)}"
                raise ModelExportError(msg)

            for backend in selected:
                temporary_dir = Path(original.path) / f".tmp-export-{backend.value}-{uuid4().hex}"
                try:
                    temporary_dir.mkdir(parents=True)
                    cls._export_policy(policy, original, temporary_dir, backend)
                    temporary_backend_dir = temporary_dir / backend.value
                    if not temporary_backend_dir.is_dir():
                        msg = f"Export did not produce a {backend.value} backend directory"
                        raise ModelExportError(msg)
                    target_backend_dir = exports_dir / backend.value
                    if target_backend_dir.exists():
                        shutil.rmtree(target_backend_dir)
                    shutil.move(str(temporary_backend_dir), str(target_backend_dir))
                finally:
                    shutil.rmtree(temporary_dir, ignore_errors=True)

            if compress and ExportBackend.OPENVINO in selected:
                ov_dir = exports_dir / ExportBackend.OPENVINO.value
                if ov_dir.is_dir():
                    ModelCompressionService._compress_openvino_dir(ov_dir)
        except Exception as exc:
            msg = f"Re-export failed for model '{model_id}': {exc}"
            raise ModelExportError(msg) from exc

        return original
