# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pydantic models for raw-video dataset ``manifest.json`` schema.

Expected on-disk layout::

    raw_dataset/
    ├── manifest.json
    ├── episode_000/
    │   ├── data.jsonl        # {"timestamp": 0.0, "state": [...], "action": [...]}
    │   ├── cam_top.mp4
    │   └── cam_gripper.mp4
    ├── episode_001/
    │   └── ...
    └── .cache/
        └── stats.json        # auto-created by adapter
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator


class CameraConfig(BaseModel):
    """Configuration for a single camera stream.

    Attributes:
        name: Camera identifier (e.g. ``"top"``, ``"gripper"``).
              Must be a valid Python-style identifier (alphanumeric + underscore,
              cannot start with a digit).
    """

    name: str

    @field_validator("name")
    @classmethod
    def _name_must_be_identifier(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
            raise ValueError(
                f"Camera name must be a valid identifier "
                f"(alphanumeric + underscore, cannot start with a digit), got {value!r}"
            )
        return value


class EpisodeEntry(BaseModel):
    """Manifest entry describing a single episode on disk.

    Attributes:
        episode_dir: Relative directory name within the dataset root
                     (e.g. ``"episode_000"``).
        data_file: Filename of the JSONL data file inside *episode_dir*.
        video_files: Mapping from camera name to video filename
                     (e.g. ``{"top": "cam_top.mp4"}``).
    """

    episode_dir: str
    data_file: str = "data.jsonl"
    video_files: dict[str, str]


class DatasetManifest(BaseModel):
    """Top-level schema for ``manifest.json``.

    Attributes:
        name: Human-readable dataset name.
        fps: Target frames per second for training.  The video FPS is
             authoritative; if the data FPS differs, nearest-neighbour
             resampling is applied.
        state_dim: Dimensionality of the state vector.
        action_dim: Dimensionality of the action vector.
        state_names: Per-element names for the state vector (e.g.
            ``["shoulder_pan.pos", "shoulder_lift.pos", ...]``).  Must
            have exactly *state_dim* entries.
        action_names: Per-element names for the action vector.  Must
            have exactly *action_dim* entries.
        robot_type: Robot type identifier (e.g. ``"so101_follower"``).
            Defaults to ``"unknown"`` when the source dataset does not
            provide robot type information.
        cameras: Camera configurations used across the dataset.
        episodes: Ordered list of episode entries.
        task_description: Optional human-readable description of the task.
    """

    name: str
    fps: int = Field(..., gt=0)
    state_dim: int = Field(..., gt=0)
    action_dim: int = Field(..., gt=0)
    state_names: list[str]
    action_names: list[str]
    robot_type: str = "unknown"
    cameras: list[CameraConfig]
    episodes: list[EpisodeEntry]
    task_description: str = ""

    @model_validator(mode="after")
    def _validate_names_length(self) -> DatasetManifest:
        """Ensure state_names and action_names match their respective dimensions."""
        if len(self.state_names) != self.state_dim:
            raise ValueError(f"state_names has {len(self.state_names)} entries but state_dim is {self.state_dim}")
        if len(self.action_names) != self.action_dim:
            raise ValueError(f"action_names has {len(self.action_names)} entries but action_dim is {self.action_dim}")
        return self

    @model_validator(mode="after")
    def _validate_episode_camera_names(self) -> DatasetManifest:
        """Ensure every episode references exactly the camera names declared in *cameras*."""
        expected_names = {cam.name for cam in self.cameras}
        for episode in self.episodes:
            episode_names = set(episode.video_files.keys())
            if episode_names != expected_names:
                missing = expected_names - episode_names
                extra = episode_names - expected_names
                parts: list[str] = [f"Episode {episode.episode_dir!r} camera mismatch."]
                if missing:
                    parts.append(f"Missing cameras: {sorted(missing)}")
                if extra:
                    parts.append(f"Unexpected cameras: {sorted(extra)}")
                raise ValueError(" ".join(parts))
        return self


def load_manifest(dataset_root: Path) -> DatasetManifest:
    """Load and validate a ``manifest.json`` from a raw-video dataset.

    The function parses the manifest, then checks that every referenced file
    and directory actually exists on disk.

    Args:
        dataset_root: Path to the top-level dataset directory that contains
                      ``manifest.json``.

    Returns:
        A fully validated :class:`DatasetManifest`.

    Raises:
        FileNotFoundError: If ``manifest.json`` or any file/directory
            referenced by the manifest does not exist.
        ValueError: If the manifest JSON is malformed or fails schema
            validation.
    """
    manifest_path = dataset_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {manifest_path}: {exc}") from exc

    try:
        manifest = DatasetManifest.model_validate(raw)
    except Exception as exc:
        raise ValueError(f"Manifest validation failed for {manifest_path}: {exc}") from exc

    # ------------------------------------------------------------------
    # Verify that all referenced files and directories exist on disk
    # ------------------------------------------------------------------
    missing: list[str] = []

    for episode in manifest.episodes:
        ep_dir = dataset_root / episode.episode_dir
        if not ep_dir.is_dir():
            missing.append(f"Episode directory: {ep_dir}")
            continue  # skip file checks when the parent dir is missing

        data_path = ep_dir / episode.data_file
        if not data_path.is_file():
            missing.append(f"Data file: {data_path}")

        for camera_name, video_filename in episode.video_files.items():
            video_path = ep_dir / video_filename
            if not video_path.is_file():
                missing.append(f"Video file for camera {camera_name!r}: {video_path}")

    if missing:
        details = "\n  - ".join(missing)
        raise FileNotFoundError(
            f"The following files/directories referenced by {manifest_path} are missing:\n  - {details}"
        )

    return manifest


def save_manifest(dataset_root: Path, manifest: DatasetManifest) -> None:
    """Atomically write a :class:`DatasetManifest` to ``manifest.json``.

    The write is performed via a temporary file in the same directory followed
    by :func:`os.replace`, so that concurrent readers never observe a
    partially-written manifest.

    Args:
        dataset_root: Path to the top-level dataset directory.
        manifest: The manifest to persist.

    Raises:
        OSError: If the write or rename fails.
    """
    dataset_root.mkdir(parents=True, exist_ok=True)
    manifest_path = dataset_root / "manifest.json"
    data = manifest.model_dump(mode="json")

    # Write to a temp file in the same directory so os.replace is atomic
    # (same filesystem guarantees).
    fd, tmp_path = tempfile.mkstemp(dir=dataset_root, suffix=".tmp", prefix=".manifest_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp_path, manifest_path)
    except BaseException:
        # Clean up temp file on failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


__all__ = [
    "CameraConfig",
    "DatasetManifest",
    "EpisodeEntry",
    "load_manifest",
    "save_manifest",
]
