"""Tests for local model import commands."""

from pathlib import Path

from cli.models import _act_checkpoint_files


def test_act_checkpoint_files_ignores_root_level_last_checkpoint(tmp_path: Path) -> None:
    epoch_checkpoints = [
        tmp_path / "act-None--epoch=005-val" / "loss=0.2345.ckpt",
        tmp_path / "act-None--epoch=007-val" / "loss=0.2336.ckpt",
        tmp_path / "act-None--epoch=008-val" / "loss=0.2347.ckpt",
        tmp_path / "act-None--epoch=011-val" / "loss=0.2349.ckpt",
        tmp_path / "act-None--epoch=012-val" / "loss=0.2357.ckpt",
    ]
    for checkpoint in epoch_checkpoints:
        checkpoint.parent.mkdir()
        checkpoint.touch()
    (tmp_path / "last.ckpt").touch()

    assert _act_checkpoint_files(tmp_path) == sorted(epoch_checkpoints)
