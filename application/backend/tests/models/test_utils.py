from pathlib import Path
from unittest.mock import patch

import pytest

from models.utils import _find_checkpoint, load_inference_model
from schemas import InferenceBackend, InferenceDevice


def test_load_inference_model_uses_selected_torch_device(test_model) -> None:
    inference_device = InferenceDevice(backend=InferenceBackend.TORCH, device="cuda:0")

    with patch("physicalai.inference.InferenceModel") as mock_inference_model:
        load_inference_model(test_model, inference_device=inference_device)

    mock_inference_model.assert_called_once_with(
        export_dir=Path(test_model.path) / "exports" / "torch",
        policy_name=test_model.policy,
        backend="torch",
        device="cuda:0",
    )


def test_load_inference_model_uses_selected_openvino_device(test_model) -> None:
    inference_device = InferenceDevice(backend=InferenceBackend.OPENVINO, device="GPU")

    with patch("physicalai.inference.InferenceModel") as mock_inference_model:
        load_inference_model(test_model, inference_device=inference_device)

    mock_inference_model.assert_called_once_with(
        export_dir=Path(test_model.path) / "exports" / "openvino",
        policy_name=test_model.policy,
        backend="openvino",
        device="GPU",
    )


def test_find_checkpoint_prefers_model_ckpt(tmp_path) -> None:
    (tmp_path / "model.ckpt").touch()
    (tmp_path / "last.ckpt").touch()

    result = _find_checkpoint(tmp_path)

    assert result == tmp_path / "model.ckpt"


def test_find_checkpoint_falls_back_to_last_ckpt(tmp_path) -> None:
    (tmp_path / "last.ckpt").touch()

    result = _find_checkpoint(tmp_path)

    assert result == tmp_path / "last.ckpt"


def test_find_checkpoint_raises_when_no_checkpoint(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="No checkpoint found"):
        _find_checkpoint(tmp_path)
