# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Portable RTC launcher, copied into runtime downloads (no Studio dependency)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from physicalai.config import Config
from physicalai.inference import InferenceModel
from physicalai.inference.callbacks import RTCLatencyTracker
from physicalai.runtime import PolicySource, RobotRuntime, RTCActionQueue, RTCExecution


def _construct(value: Any) -> Any:
    if isinstance(value, dict):
        if "class_path" in value:
            return Config.from_dict(value).instantiate()
        return {key: _construct(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_construct(item) for item in value]
    return value


def build_runtime(document: dict) -> RobotRuntime:
    """Share one latency tracker between the model and RTC execution.

    Plain JSON/YAML recipes cannot express shared object identity. Construct
    this pair explicitly, while keeping hardware and tuning in runtime.yaml.
    """
    args = dict(document["init_args"])
    source = args.pop("action_source")["init_args"]
    tracker = RTCLatencyTracker()
    model_args = dict(source["model"]["init_args"])
    callbacks = _construct(model_args.pop("callbacks", []))
    model = InferenceModel(**model_args, callbacks=[*callbacks, tracker])
    try:
        execution = RTCExecution(**source["execution"]["init_args"], latency_tracker=tracker)
        action_source = PolicySource(model, execution, RTCActionQueue(), task=source.get("task"))
        return RobotRuntime(action_source=action_source, **_construct(args))
    except Exception:
        model.close()
        raise


def main() -> None:
    """Run the selected export using the accompanying hardware recipe."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-s", type=float, default=60)
    arguments = parser.parse_args()
    os.chdir(Path(__file__).resolve().parent)
    with build_runtime(Config.load("runtime.yaml").to_dict()) as runtime:
        runtime.run(duration_s=arguments.duration_s)


if __name__ == "__main__":
    main()
