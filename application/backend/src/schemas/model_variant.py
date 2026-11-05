# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Portable export variants and safe, package-local filesystem access."""

import json
from datetime import datetime
from pathlib import Path
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict

VARIANT_IDS = frozenset(
    {"original", "int8", "rtc", "rtc-int8", "snapflow", "snapflow-int8", "snapflow-rtc", "snapflow-rtc-int8"}
)
BACKENDS = frozenset({"openvino", "onnx", "torch", "executorch"})
VariantOperation = Literal["int8_sym", "rtc", "snapflow"]


class VariantRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_variant: str = "original"
    operation: VariantOperation = "int8_sym"


class VariantInfo(BaseModel):
    schema_version: Literal[1] = 1
    id: str
    source_variant: str = "original"
    checkpoint_kind: Literal["ordinary", "snapflow"] = "ordinary"
    rtc_enabled: bool = False
    weight_compression: Literal["int8_sym"] | None = None
    created_at: datetime | None = None
    job_id: UUID | None = None


class VariantOperationDetail(BaseModel):
    operation: VariantOperation
    target_variant: str
    available: bool
    reason: str | None = None


def export_directory(model_path: Path, backend: str, variant: str = "original") -> Path:
    """Resolve an allowlisted export without following paths outside the model."""
    if backend not in BACKENDS or variant not in VARIANT_IDS:
        raise ValueError("Unknown export backend or variant")
    root = model_path.resolve()
    directory = root / "exports" / backend
    if variant != "original":
        directory = directory / "variants" / variant
    if not directory.resolve().is_relative_to(root) or any(
        parent.is_symlink() for parent in (directory, *directory.parents) if parent.is_relative_to(root)
    ):
        raise ValueError("Export paths must not contain symbolic links")
    return directory


def package_files(directory: Path) -> list[Path]:
    """List a standalone package, excluding sibling variants and private staging."""
    files = []
    for path in directory.rglob("*"):
        relative = path.relative_to(directory)
        if relative.parts[0] == "variants" or any(part.startswith(".") for part in relative.parts):
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError("Export packages must not contain symbolic links")
        if path.is_file():
            files.append(path)
    return files


def read_package(directory: Path, backend: str) -> dict:
    """Validate the manifest and all declared local artifact dependencies."""
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    artifacts = manifest["model"]["artifacts"]
    if not isinstance(artifacts, dict) or backend not in artifacts:
        raise ValueError("Missing backend artifact")
    if backend == "openvino" and (not isinstance(artifacts[backend], str) or Path(artifacts[backend]).suffix != ".xml"):
        raise ValueError("OpenVINO policy artifact must be XML")
    files = {path.relative_to(directory).as_posix() for path in package_files(directory)}

    def require_file(value: str) -> None:
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() not in files:
            raise ValueError(f"Missing or unsafe export artifact: {value}")
        if relative.suffix == ".xml" and relative.with_suffix(".bin").as_posix() not in files:
            raise ValueError(f"Missing weights for export artifact: {value}")

    for artifact in artifacts.values():
        if not isinstance(artifact, str):
            raise ValueError("Artifact paths must be strings")
        require_file(artifact)

    def validate_components(value: object) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "artifact" and isinstance(item, str):
                    require_file(item)
                else:
                    validate_components(item)
        elif isinstance(value, list):
            for item in value:
                validate_components(item)

    validate_components(manifest["model"])
    return manifest


def action_chunk_size(manifest: dict) -> int | None:
    """Read the declared action chunk length without guessing a policy default."""
    for feature in manifest.get("model", {}).get("output_features", []):
        args = feature.get("init_args", feature)
        shape = args.get("shape", [])
        if args.get("name") == "action" and shape and isinstance(shape[0], int) and shape[0] > 0:
            return shape[0]
    return None


def variant_info(manifest: dict, variant: str) -> VariantInfo:
    """Read portable metadata, including exports predating named variants."""
    studio = manifest.get("studio", {})
    if not isinstance(studio, dict):
        raise ValueError("Manifest studio metadata must be an object")
    metadata = studio.get("variant")
    if metadata is not None:
        info = VariantInfo.model_validate(metadata)
        return info.model_copy(update={"id": variant})
    features = manifest.get("model", {}).get("input_features", [])
    rtc = any(
        isinstance(feature, dict) and feature.get("init_args", feature).get("name") == "prev_chunk_left_over"
        for feature in features
    )
    return VariantInfo(
        id=variant,
        checkpoint_kind="snapflow" if "snapflow" in variant else "ordinary",
        rtc_enabled=rtc or "rtc" in variant,
        weight_compression="int8_sym" if "int8" in variant or studio.get("optimization") else None,
    )


def target_variant(info: VariantInfo, operation: VariantOperation) -> str:
    """Canonicalize the complete feature set, independent of operation order."""
    features = []
    if info.checkpoint_kind == "snapflow" or operation == "snapflow":
        features.append("snapflow")
    if info.rtc_enabled or operation == "rtc":
        features.append("rtc")
    if info.weight_compression or operation == "int8_sym":
        features.append("int8")
    return "-".join(features)
