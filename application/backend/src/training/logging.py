# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Build Lightning loggers from user settings."""

import os
from pathlib import Path
from typing import Any

from settings import LoggerSettings


def build_training_logger(config: LoggerSettings, *, log_root: Path, run_name: str) -> Any:
    """Return one or more configured Lightning loggers."""
    loggers: list[Any] = []
    for provider in config.providers:
        if provider == "csv":
            from lightning.pytorch.loggers import CSVLogger

            loggers.append(CSVLogger(log_root, name=run_name))
        elif provider == "tensorboard":
            from lightning.pytorch.loggers import TensorBoardLogger

            loggers.append(TensorBoardLogger(save_dir=log_root, name=run_name))
        elif provider == "wandb":
            from lightning.pytorch.loggers import WandbLogger

            kwargs: dict[str, Any] = {}
            if config.wandb_api_key is not None:
                os.environ["WANDB_API_KEY"] = config.wandb_api_key.get_secret_value()
            if config.wandb_project is not None:
                kwargs["project"] = config.wandb_project
            if config.wandb_entity is not None:
                kwargs["entity"] = config.wandb_entity
            loggers.append(WandbLogger(**kwargs))
    return loggers[0] if len(loggers) == 1 else loggers
