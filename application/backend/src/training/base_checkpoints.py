# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pretrained base checkpoints shared by the physicalai and LeRobot training engines.

VLA policies (pi05, smolvla) have no useful from-scratch initialization: they
must be fine-tuned from a pretrained checkpoint. Both engines read this map so a
job trains the same base weights regardless of which engine the user picks.
"""

PRETRAINED_BASE_CHECKPOINTS: dict[str, str] = {
    "pi05": "lerobot/pi05_base",
    "smolvla": "lerobot/smolvla_base",
}
"""Hub checkpoints used to initialize policies that only fine-tune from pretrained weights."""
