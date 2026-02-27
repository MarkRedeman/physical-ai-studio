# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from physicalai.inference import InferenceModel
from physicalai.policies import ACT, Pi0, Pi05, SmolVLA
from physicalai.policies.base import Policy

from schemas import Model
from utils.device import get_torch_device


def load_policy(model: Model) -> Policy:
    """Load existing model for retraining.

    Tries ``model.ckpt`` first (standard Lightning checkpoint produced by
    training).  Falls back to the exported inference artifact under
    ``exports/torch/{policy}.pt`` so that re-imported Studio exports can
    also be used as a base for retraining.

    Raises:
        ValueError: If the model is a HuggingFace import (not retrainable)
            or the policy type is not implemented.
        FileNotFoundError: If no loadable checkpoint exists.
    """
    if model.properties.get("source") == "huggingface":
        raise ValueError(
            f"Model '{model.name}' was imported from HuggingFace and cannot be retrained. "
            "HuggingFace models are inference-only."
        )

    model_path = _resolve_checkpoint_path(model)
    if model.policy == "act":
        return ACT.load_from_checkpoint(model_path)
    if model.policy == "pi0":
        return Pi0.load_from_checkpoint(model_path, weights_only=True)
    if model.policy == "pi05":
        return Pi05.load_from_checkpoint(model_path)
    if model.policy == "smolvla":
        return SmolVLA.load_from_checkpoint(model_path)
    raise ValueError(f"Policy {model.policy} not implemented.")


def _resolve_checkpoint_path(model: Model) -> str:
    """Return the best available checkpoint path for *model*.

    Prefers the full Lightning checkpoint (``model.ckpt``) when present.
    Falls back to the exported torch artifact (``exports/torch/{policy}.pt``)
    which is valid for re-imported Studio exports of natively-trained models
    (same state_dict key structure as the Lightning checkpoint).

    This function must NOT be called for HuggingFace-imported models —
    their export artifacts use a different state_dict key structure that is
    incompatible with the native policy ``load_from_checkpoint`` methods.
    """
    model_dir = Path(model.path)

    ckpt_path = model_dir / "model.ckpt"
    if ckpt_path.exists():
        return str(ckpt_path)

    export_path = model_dir / "exports" / "torch" / f"{model.policy}.pt"
    if export_path.exists():
        return str(export_path)

    raise FileNotFoundError(
        f"No loadable checkpoint found for model {model.name!r} ({model.id}). Looked for {ckpt_path} and {export_path}."
    )


def load_inference_model(model: Model, backend: str) -> InferenceModel:
    """Loads inference model."""
    inference_device = "auto"
    if backend == "torch":
        inference_device = get_torch_device()

    export_dir = Path(model.path) / "exports" / backend
    return InferenceModel(export_dir=export_dir, policy_name=model.policy, backend=backend, device=inference_device)


def setup_policy(model: Model) -> Policy:
    """Setup policy for Model training."""
    if model.policy == "act":
        return ACT()
    if model.policy == "pi0":
        return Pi0()
    if model.policy == "pi05":
        return Pi05(pretrained_name_or_path="lerobot/pi05_base")
    if model.policy == "smolvla":
        return SmolVLA()

    raise ValueError(f"Policy not implemented yet: {model.policy}")
