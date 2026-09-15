from pathlib import Path
from unittest.mock import patch

from lightning.pytorch.loggers import CSVLogger

from settings import LoggerSettings
from training.logging import build_training_logger


def test_csv_provider_returns_csv_logger(tmp_path: Path) -> None:
    logger = build_training_logger(LoggerSettings(providers=["csv"]), log_root=tmp_path, run_name="run")

    assert isinstance(logger, CSVLogger)
    assert logger.name == "run"


def test_multiple_providers_return_a_list(tmp_path: Path) -> None:
    with patch("lightning.pytorch.loggers.TensorBoardLogger") as tensorboard_logger:
        logger = build_training_logger(
            LoggerSettings(providers=["csv", "tensorboard"]), log_root=tmp_path, run_name="run"
        )

    assert isinstance(logger, list)
    assert isinstance(logger[0], CSVLogger)
    assert logger[1] is tensorboard_logger.return_value
