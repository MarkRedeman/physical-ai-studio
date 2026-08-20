# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for building the training-run Lightning logger(s)."""

from pathlib import Path
from unittest.mock import patch

from lightning.pytorch.loggers import CSVLogger

from settings import LoggerSettings
from training.logging import build_training_logger


def test_csv_provider_returns_single_csv_logger(tmp_path: Path) -> None:
    logger = build_training_logger(LoggerSettings(providers=["csv"]), log_root=tmp_path, run_name="run-a")

    assert isinstance(logger, CSVLogger)
    assert logger.name == "run-a"


def test_multiple_providers_return_a_list_of_loggers(tmp_path: Path) -> None:
    with patch("lightning.pytorch.loggers.TensorBoardLogger") as tb_logger_cls:
        logger = build_training_logger(
            LoggerSettings(providers=["csv", "tensorboard"]),
            log_root=tmp_path,
            run_name="run-b",
        )

    assert isinstance(logger, list)
    assert len(logger) == 2
    assert isinstance(logger[0], CSVLogger)
    assert logger[1] is tb_logger_cls.return_value
    tb_logger_cls.assert_called_once_with(save_dir=tmp_path, name="run-b")


def test_duplicate_providers_are_deduplicated(tmp_path: Path) -> None:
    settings = LoggerSettings(providers=["csv", "csv"])

    logger = build_training_logger(settings, log_root=tmp_path, run_name="run-c")

    assert isinstance(logger, CSVLogger)
