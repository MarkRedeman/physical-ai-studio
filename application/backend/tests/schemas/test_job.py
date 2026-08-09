# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the training job payload validation."""

from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from schemas.job import TrainJobPayload


def _payload(**overrides: object) -> TrainJobPayload:
    return TrainJobPayload.model_validate(
        {
            "project_id": str(uuid4()),
            "dataset_id": str(uuid4()),
            "policy": "act",
            "model_name": "test model",
            **overrides,
        }
    )


@pytest.mark.parametrize(
    ("policy", "engine"),
    [
        ("act", "physicalai"),
        ("pi0", "physicalai"),
        ("act", "lerobot"),
        ("diffusion", "lerobot"),
    ],
)
def test_supported_policy_for_engine_accepted(policy: str, engine: str) -> None:
    payload = _payload(policy=policy, training_engine=engine)
    assert payload.policy == policy
    assert payload.training_engine == engine


@pytest.mark.parametrize(
    ("policy", "engine"),
    [
        ("diffusion", "physicalai"),
        ("pi0", "lerobot"),
        ("groot", "physicalai"),
        ("gpt", "lerobot"),
    ],
)
def test_unsupported_policy_for_engine_rejected(policy: str, engine: str) -> None:
    with pytest.raises(ValidationError):
        _payload(policy=policy, training_engine=engine)


def test_engine_defaults_to_physicalai() -> None:
    payload = _payload()
    assert payload.training_engine == "physicalai"
