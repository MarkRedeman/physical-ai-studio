# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Raw-video dataset adapter for the GetiAction training pipeline.

This package provides :class:`RawVideoDatasetAdapter`, which implements the
:class:`getiaction.data.Dataset` abstract base class to enable training directly
from raw video files and JSONL telemetry logs without a prior format conversion
step.

Public API
----------
.. autosummary::
    RawVideoDatasetAdapter
    DatasetManifest
    load_manifest

Example usage::

    from pathlib import Path
    from internal_datasets.raw_video import RawVideoDatasetAdapter

    dataset = RawVideoDatasetAdapter(Path("data/my_raw_dataset"))
    print(len(dataset))  # total frames
    obs = dataset[0]     # single Observation
"""

from __future__ import annotations

from .adapter import RawVideoDatasetAdapter
from .converters import LeRobotToRawVideoConverter, RawVideoToLeRobotConverter
from .manifest import DatasetManifest, load_manifest

__all__ = [
    "DatasetManifest",
    "LeRobotToRawVideoConverter",
    "RawVideoDatasetAdapter",
    "RawVideoToLeRobotConverter",
    "load_manifest",
]
