# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Util functions for Training."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from physicalai.data.dataset import Dataset

if TYPE_CHECKING:
    from physicalai.data import DataModule
    from physicalai.policies.base.policy import Policy

logger = logging.getLogger(__name__)


def _get_delta_indices(model: Any, attr_name: str) -> list[int] | None:  # noqa: ANN401
    """Get delta indices from a model, handling both first-party and LeRobot policies.

    Args:
        model: The model to extract delta indices from
        attr_name: Name of the delta indices attribute (e.g., 'observation_delta_indices')

    Returns:
        List of delta indices or None if not available/not needed.
    """
    # Try direct attribute access (first-party policies like ACT)
    if hasattr(model, attr_name):
        return getattr(model, attr_name)

    # Try config attribute (LeRobot policies)
    if hasattr(model, "config"):
        config = model.config
        # Convert observation_delta_indices -> n_obs_steps
        # action_delta_indices -> n_action_steps, etc.
        if attr_name == "observation_delta_indices" and hasattr(config, "n_obs_steps"):
            n_steps = config.n_obs_steps
            # Only add time dimension if n_obs_steps > 1 (i.e., we need history)
            # n_obs_steps=1 means just current observation, no time dimension needed
            return list(range(-n_steps + 1, 1)) if n_steps > 1 else None
        if attr_name == "action_delta_indices" and hasattr(config, "n_action_steps"):
            n_steps = config.n_action_steps
            return list(range(n_steps)) if n_steps > 0 else None
        # Try direct config attribute
        if hasattr(config, attr_name):
            return getattr(config, attr_name)

    return None


def _reformat_lerobot_dataset(policy: Policy, dataset: Dataset) -> None:
    """Set delta_indices on a LeRobot dataset via delta_timestamps conversion."""
    from lerobot.datasets.utils import check_delta_timestamps, get_delta_indices

    delta_timestamps: dict[str, list[float]] = {}

    # For policies with lerobot_policy attribute, use that; otherwise use policy.model
    lerobot_model = getattr(policy, "lerobot_policy", None) or policy.model

    for key in dataset.raw_features:
        reward_delta_indices = _get_delta_indices(lerobot_model, "reward_delta_indices")
        if key == "next.reward" and reward_delta_indices is not None:
            delta_timestamps[key] = [i / dataset.fps for i in reward_delta_indices]

        action_delta_indices = _get_delta_indices(lerobot_model, "action_delta_indices")
        if key == "action" and action_delta_indices is not None:
            delta_timestamps[key] = [i / dataset.fps for i in action_delta_indices]

        observation_delta_indices = _get_delta_indices(lerobot_model, "observation_delta_indices")
        if key.startswith("observation.") and observation_delta_indices is not None:
            delta_timestamps[key] = [i / dataset.fps for i in observation_delta_indices]

    if delta_timestamps:
        check_delta_timestamps(delta_timestamps, dataset.fps, dataset.tolerance_s)
        dataset.delta_indices = get_delta_indices(delta_timestamps, dataset.fps)


def _reformat_generic_dataset(policy: Policy, dataset: Dataset) -> None:
    """Set delta_indices on a generic Dataset (e.g. RawVideoDatasetAdapter).

    Builds the delta_indices dict directly from the policy's integer
    offsets, without the LeRobot delta_timestamps intermediate step.
    """
    model = getattr(policy, "lerobot_policy", None) or policy.model

    delta_indices: dict[str, list[int]] = {}
    for key in dataset.raw_features:
        if key == "next.reward":
            indices = _get_delta_indices(model, "reward_delta_indices")
            if indices is not None:
                delta_indices[key] = indices

        elif key == "action":
            indices = _get_delta_indices(model, "action_delta_indices")
            if indices is not None:
                delta_indices[key] = indices

        elif key.startswith("observation."):
            indices = _get_delta_indices(model, "observation_delta_indices")
            if indices is not None:
                delta_indices[key] = indices

    if delta_indices:
        dataset.delta_indices = delta_indices
        logger.info("Set delta_indices on dataset: %s", {k: len(v) for k, v in delta_indices.items()})


def reformat_dataset_to_match_policy(policy: Policy, datamodule: DataModule) -> None:
    """Reformat dataset to have correct deltas and parameters depending on policy.

    Works with any :class:`Dataset` implementation. For LeRobot datasets,
    uses the delta_timestamps -> delta_indices conversion via LeRobot utils.
    For other datasets (e.g. raw-video), sets delta_indices directly from
    the policy's integer offsets.
    """
    from physicalai.data.lerobot.dataset import _LeRobotDatasetAdapter  # noqa: PLC2701

    dataset = datamodule.train_dataset

    if isinstance(dataset, _LeRobotDatasetAdapter):
        _reformat_lerobot_dataset(policy, dataset)
    elif isinstance(dataset, Dataset):
        _reformat_generic_dataset(policy, dataset)
