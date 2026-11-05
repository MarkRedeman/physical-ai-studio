# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import fcntl
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from db.schema import JobDB
from exceptions import DuplicateJobException, InsufficientDiskSpaceError, InvalidResourceError, ResourceNotFoundError
from repositories.mappers import JobMapper
from schemas.job import Job, OptimizeModelJob, OptimizeModelJobPayload
from schemas.model import Model
from schemas.model_variant import (
    VariantInfo,
    VariantOperationDetail,
    VariantRequest,
    export_directory,
    package_files,
    read_package,
    target_variant,
    variant_info,
)
from services.model_service import ModelService
from settings import get_settings


class ModelOptimizationService:
    """Build immutable export variants without creating another model."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    @staticmethod
    def operations(model: Model, backend: str, info: VariantInfo, packages: dict) -> list[VariantOperationDetail]:
        """Describe operations using the same validation used at submission."""
        result = []
        for operation in ("int8_sym", "rtc", "snapflow"):
            target = target_variant(info, operation)
            reason = None
            if operation == "int8_sym" and (backend != "openvino" or info.weight_compression):
                reason = "INT8 weight compression requires an uncompressed OpenVINO export"
            elif operation == "rtc" and (model.policy not in {"smolvla", "pi05"} or info.rtc_enabled):
                reason = "RTC requires a supported policy without RTC already enabled"
            elif operation == "snapflow" and (
                model.policy not in {"smolvla", "pi05"} or info.checkpoint_kind == "snapflow"
            ):
                reason = "SnapFlow requires a supported, non-SnapFlow source"
            elif operation != "int8_sym":
                kind = "snapflow" if operation == "snapflow" else info.checkpoint_kind
                checkpoint = Path(model.path) / ("snapflow-model.ckpt" if kind == "snapflow" else "model.ckpt")
                if not checkpoint.is_file():
                    reason = (
                        "A distilled checkpoint is required; run SnapFlow training first"
                        if kind == "snapflow"
                        else "RTC export requires the original checkpoint"
                    )
            if reason is None and (backend, target) in packages:
                reason = "This variant already exists"
            result.append(
                VariantOperationDetail(
                    operation=operation, target_variant=target, available=reason is None, reason=reason
                )
            )
        return result

    async def submit(
        self, model_id: UUID, backend: str = "openvino", request: VariantRequest | None = None
    ) -> OptimizeModelJob:
        request = request or VariantRequest()
        model = await ModelService(self.session).get_model_by_id(model_id)
        source = ModelService.resolve_export(model, backend, request.source_variant)
        info = variant_info(read_package(source, backend), request.source_variant)
        if (
            request.source_variant == "original"
            and model.snapflow_enabled
            and "variant" not in read_package(source, backend).get("studio", {})
        ):
            info.checkpoint_kind = "snapflow"
        if backend == "openvino":
            self.openvino_artifact(source)
        operation = next(
            item
            for item in self.operations(model, backend, info, model.export_packages())
            if item.operation == request.operation
        )
        if not operation.available:
            raise InvalidResourceError("Model", operation.reason or "Unsupported variant operation")
        payload = OptimizeModelJobPayload(
            model_id=model.id,
            backend=backend,
            source_variant=request.source_variant,
            operation=request.operation,
            target_variant=operation.target_variant,
        )
        job = OptimizeModelJob(
            project_id=model.project_id, payload=payload, message=f"Creating {payload.target_variant} variant"
        )
        reservation = self.reservation(model, payload)
        reservation.parent.mkdir(parents=True, exist_ok=True)
        if reservation.is_symlink() or reservation.parent.is_symlink():
            raise InvalidResourceError("Model", "Variant reservations must not contain symbolic links")
        # Keep the inode for the model's lifetime: unlinking a lock file can let
        # another process lock a different inode for the same target. Kernel
        # locks disappear on process death, so uncommitted submissions recover.
        with reservation.open("a+", encoding="utf-8") as file:
            try:
                fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise DuplicateJobException from exc
            file.seek(0)
            owner = file.read().strip()
            if owner:
                status = await self.session.scalar(select(JobDB.status).where(JobDB.id == owner))
                if status in {"pending", "running"}:
                    raise DuplicateJobException
            existing = await self.session.scalar(
                select(JobDB.id)
                .where(
                    JobDB.type == "optimize_model",
                    JobDB.status.in_(("pending", "running")),
                    func.json_extract(JobDB.payload, "$.model_id") == str(model.id),
                    func.coalesce(func.json_extract(JobDB.payload, "$.backend"), "openvino") == backend,
                    func.coalesce(func.json_extract(JobDB.payload, "$.target_variant"), "int8")
                    == payload.target_variant,
                )
                .limit(1)
            )
            if existing is not None:
                raise DuplicateJobException
            file.seek(0)
            file.truncate()
            file.write(str(job.id))
            file.flush()
            self.session.add(JobMapper.to_schema(job))
            try:
                await self.session.commit()
            except Exception:
                await self.session.rollback()
                raise
        return job

    @staticmethod
    def reservation(model: Model, payload: OptimizeModelJobPayload) -> Path:
        return Path(model.path) / ".variant-reservations" / f"{payload.backend}-{payload.target_variant}"

    @staticmethod
    def openvino_artifact(export_dir: Path) -> Path:
        try:
            artifact = read_package(export_dir, "openvino")["model"]["artifacts"]["openvino"]
            path = export_dir / artifact
            if path.suffix != ".xml":
                raise ValueError("OpenVINO policy artifact must be XML")
            return path
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise InvalidResourceError("Model", "A complete OpenVINO export is required for optimization") from exc

    @staticmethod
    def build_variant(model: Model, payload: OptimizeModelJobPayload, destination: Path, job_id: UUID) -> VariantInfo:
        source = ModelService.resolve_export(model, payload.backend, payload.source_variant)
        source_manifest = read_package(source, payload.backend)
        info = variant_info(source_manifest, payload.source_variant)
        if (
            payload.source_variant == "original"
            and model.snapflow_enabled
            and "variant" not in source_manifest.get("studio", {})
        ):
            info.checkpoint_kind = "snapflow"
        required_bytes = sum(path.stat().st_size for path in package_files(source))
        if shutil.disk_usage(destination.parent).free < required_bytes * 3:
            raise InsufficientDiskSpaceError("Not enough disk space to create the export variant")
        destination.mkdir()
        if payload.operation == "int8_sym":
            for path in package_files(source):
                target = destination / path.relative_to(source)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
        else:
            ModelOptimizationService._export_checkpoint(model, payload, info, destination)
        if info.weight_compression or payload.operation == "int8_sym":
            if payload.backend != "openvino":
                raise InvalidResourceError("Model", "Weight compression is only supported for OpenVINO")
            from physicalai.export.hooks import compress_weights_openvino_int8_sym

            compress_weights_openvino_int8_sym(str(ModelOptimizationService.openvino_artifact(destination)))
            info.weight_compression = "int8_sym"
        info.id = payload.target_variant
        info.source_variant = payload.source_variant
        info.created_at = datetime.now(UTC)
        info.job_id = job_id
        manifest = read_package(destination, payload.backend)
        for key, value in source_manifest.items():
            if key not in {"model", "policy", "input_names", "output_names", "rtc", "studio"}:
                manifest.setdefault(key, value)
        studio = manifest.setdefault("studio", {})
        if not isinstance(studio, dict):
            raise InvalidResourceError("Model", "Manifest studio metadata must be an object")
        studio = {**source_manifest.get("studio", {}), **studio}
        manifest["studio"] = studio
        studio.pop("optimization", None)
        studio["variant"] = info.model_dump(mode="json")
        (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return info

    @staticmethod
    def _export_checkpoint(
        model: Model, payload: OptimizeModelJobPayload, info: VariantInfo, destination: Path
    ) -> None:
        from physicalai.export import ExportablePolicyMixin
        from physicalai.policies import get_physicalai_policy_class
        from physicalai.policies.mixins import RTCPolicyMixin

        kind = "snapflow" if payload.operation == "snapflow" else info.checkpoint_kind
        checkpoint = Path(model.path) / ("snapflow-model.ckpt" if kind == "snapflow" else "model.ckpt")
        policy = get_physicalai_policy_class(model.policy).load_from_checkpoint(
            str(checkpoint), map_location="cpu", compile_model=False
        )
        if not isinstance(policy, ExportablePolicyMixin):
            raise InvalidResourceError("Model", "Policy does not support export")
        if kind == "snapflow" and not getattr(getattr(policy, "config", None), "snapflow_enabled", False):
            raise InvalidResourceError("Model", "Checkpoint does not contain SnapFlow-distilled weights")
        rtc = info.rtc_enabled or payload.operation == "rtc"
        config = getattr(policy, "config", None)
        if rtc and getattr(config, "n_action_steps", None) != getattr(config, "chunk_size", None):
            raise InvalidResourceError("Model", "RTC requires n_action_steps to match the full chunk size")
        if isinstance(policy, RTCPolicyMixin):
            policy.rtc_enabled = rtc
        elif rtc:
            raise InvalidResourceError("Model", "Policy does not support RTC")
        policy.eval()
        policy.export(destination, backend=payload.backend)
        info.checkpoint_kind = kind
        info.rtc_enabled = rtc

    @staticmethod
    def staging(job_id: UUID) -> Path:
        return get_settings().models_dir / f".optimize-{job_id}"

    async def cleanup(self, job: Job) -> None:
        """Delete only unpublished output owned by this job, never the source model."""
        if not isinstance(job, OptimizeModelJob):
            raise TypeError("Variant cleanup requires an optimization job")
        shutil.rmtree(self.staging(job.id), ignore_errors=True)
        try:
            model = await ModelService(self.session).get_model_by_id(job.payload.model_id)
        except ResourceNotFoundError:
            return
        destination = export_directory(Path(model.path), job.payload.backend, job.payload.target_variant)
        marker = destination / ".unpublished"
        if marker.is_file() and marker.read_text(encoding="utf-8") == str(job.id):
            shutil.rmtree(destination)
