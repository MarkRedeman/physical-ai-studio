import pytest

from models.utils import _find_checkpoint


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
