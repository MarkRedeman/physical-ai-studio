# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""CLI-local Hugging Face inspection and resumable transfers."""

import click

from cli.hf.dataset import dataset
from cli.hf.model import model


@click.group()
def hf() -> None:
    """Inspect and transfer Studio assets on Hugging Face Hub.

    Commands operate on local Studio storage, not a remote Studio HTTP server.
    Use the same STORAGE_DIR and SETTINGS_FILE as the backend.
    """


hf.add_command(model)
hf.add_command(dataset)
