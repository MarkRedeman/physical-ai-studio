# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Raw-video dataset package for the GetiAction platform.

This package provides:

- :class:`RawVideoDatasetAdapter` — read-only training adapter implementing
  :class:`getiaction.data.Dataset` for direct training from raw video.
- :class:`RawVideoDatasetClient` — CRUD client implementing
  :class:`DatasetClient` for recording, querying, and managing raw-video
  datasets.
- :class:`VideoWriter` — ffmpeg-based video writer for encoding camera
  frames during recording.
- Bidirectional converters between LeRobot and raw-video formats.

Example usage::

    from pathlib import Path

    # Training (read-only)
    from internal_datasets.raw_video import RawVideoDatasetAdapter
    dataset = RawVideoDatasetAdapter(Path("data/my_raw_dataset"))

    # Recording (CRUD)
    from internal_datasets.raw_video import RawVideoDatasetClient
    client = RawVideoDatasetClient(Path("data/my_raw_dataset"))
"""

from __future__ import annotations

from .adapter import RawVideoDatasetAdapter
from .converters import LeRobotToRawVideoConverter, RawVideoToLeRobotConverter
from .manifest import DatasetManifest, load_manifest, save_manifest
from .raw_video_dataset_client import RawVideoDatasetClient
from .video_writer import VideoWriter

__all__ = [
    "DatasetManifest",
    "LeRobotToRawVideoConverter",
    "RawVideoDatasetAdapter",
    "RawVideoDatasetClient",
    "RawVideoToLeRobotConverter",
    "VideoWriter",
    "load_manifest",
    "save_manifest",
]
