# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Load a disposable export policy without retaining training-only checkpoint tensors."""

from __future__ import annotations

import tempfile
from contextlib import contextmanager
from typing import TYPE_CHECKING, cast
from zipfile import is_zipfile

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from physicalai.policies.base import Policy


def load_export_policy(checkpoint: Path, policy_class: type[Policy]) -> Policy:
    """Load CPU weights for export, preserving Lightning's reconstruction and migration hooks.

    Modern Torch checkpoints are memory-mapped so optimizer tensors never need to
    become resident. Drop optimizer/scheduler state before constructing the policy,
    including for older checkpoints that cannot be mapped. The returned policy
    owns its weights; the mapped checkpoint is released before conversion starts.

    Args:
        checkpoint: Studio's trusted, local training checkpoint.
        policy_class: Policy class already resolved through Studio's policy registry.

    Returns:
        A policy ready for export preparation, with compilation disabled.
    """
    import torch
    from lightning.pytorch.core.saving import _load_state
    from lightning.pytorch.utilities.migration import migrate_checkpoint, pl_legacy_patch

    # Same trusted-checkpoint boundary as Lightning.load_from_checkpoint. The
    # restricted Torch unpickler alone does not omit optimizer tensor storages.
    with pl_legacy_patch():
        data = torch.load(checkpoint, map_location="cpu", mmap=is_zipfile(checkpoint), weights_only=False)
    if not isinstance(data, dict) or not isinstance(data.get("state_dict"), dict):
        raise ValueError("Export requires a Lightning checkpoint containing a state_dict")
    data, _ = migrate_checkpoint(data)
    data.pop("optimizer_states", None)
    data.pop("lr_schedulers", None)
    # Use Lightning's reconstruction step rather than duplicating constructor
    # argument filtering, strict loading, and policy checkpoint migrations.
    return cast("Policy", _load_state(policy_class, data, compile_model=False))


@contextmanager
def export_temporary_files(parent: Path) -> Iterator[None]:
    """Clean ONNX external-weight sidecars along with other export temporary files.

    The shared exporter removes its temporary ``.onnx`` file, but Torch may also
    write a multi-gigabyte ``.onnx.data`` sidecar. Confine both to a directory
    owned by this export. This context is only for the single-export worker,
    since Python's temporary-directory setting is process-global.

    Args:
        parent: Local export staging directory, also removed by failed-job cleanup.

    Yields:
        Control while the exporter owns its temporary directory.
    """
    with tempfile.TemporaryDirectory(prefix=".export-", dir=parent) as directory:
        previous = tempfile.tempdir
        tempfile.tempdir = directory
        try:
            yield
        finally:
            tempfile.tempdir = previous
