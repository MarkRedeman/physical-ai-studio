# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Build the Lightning logger(s) for a training run from user settings.

Kept separate from ``training/job.py`` so the shared runner stays agnostic of
the logger configuration and receives already-constructed logger instances.
Lightning's ``Trainer`` accepts either a single logger or a list of loggers, so
multiple providers (e.g. CSV **and** wandb) are enabled simultaneously.
"""

import os
from pathlib import Path
from typing import Any

from settings import LoggerSettings


def build_training_logger(config: LoggerSettings, *, log_root: Path, run_name: str) -> Any:
    """Return the Lightning logger(s) for a training run.

    One logger per configured provider. ``'csv'`` logs to ``run_name`` under
    ``log_root`` (matching :func:`training.job.run_training_job`'s default when
    only CSV is requested). For ``'wandb'`` the API key, if set, is also
    exported to the process environment since wandb reads it from there.

    Returns:
        A single Lightning logger, a list of them (when multiple providers are
        configured), or None when no providers are enabled — in which case
        callers keep the default CSVLogger.
    """
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

    if not loggers:
        return None
    if len(loggers) == 1:
        return loggers[0]
    return loggers
