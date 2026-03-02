# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Utilities for building raw-video dataset configuration from the environment.

Provides :func:`build_raw_video_manifest_args` as the raw-video counterpart
to :func:`utils.dataset.build_lerobot_dataset_features`.  Instead of going
through the LeRobot processor pipeline, it queries robots and cameras directly
and returns a plain dict suitable for :meth:`RawVideoDatasetClient.create`.
"""

from __future__ import annotations

from robots.robot_client_factory import RobotClientFactory
from schemas.environment import EnvironmentWithRelations


async def build_raw_video_manifest_args(
    environment: EnvironmentWithRelations,
    robot_factory: RobotClientFactory,
) -> dict:
    """Build the *features* dict expected by :meth:`RawVideoDatasetClient.create`.

    The returned dict has the structure::

        {
            "state_names": ["shoulder_pan.pos", "shoulder_lift.pos", ...],
            "action_names": ["shoulder_pan.pos", "shoulder_lift.pos", ...],
            "cameras": [
                {"name": "top", "width": 640, "height": 480},
                {"name": "gripper", "width": 640, "height": 480},
            ],
        }

    Args:
        environment: The environment configuration with robots and cameras.
        robot_factory: Factory to build robot clients for querying features.

    Returns:
        A features dict ready to pass to ``RawVideoDatasetClient.create()``.

    Raises:
        ValueError: If the environment has multiple robots (not yet supported)
            or if a camera lacks dimension information.
    """
    if len(environment.robots) > 1:
        raise ValueError("Environments with multiple robots not implemented yet")

    # Query robot joint names — these serve as both state and action names
    # since the follower's joint positions are observed as state and commanded
    # as actions during teleoperation.
    joint_names: list[str] = []
    for robot_config in environment.robots:
        client = await robot_factory.build(robot_config.robot)
        joint_names.extend(client.features())

    # Build camera info from environment camera schemas.
    cameras: list[dict] = []
    for camera in environment.cameras:
        payload = camera.payload
        if payload is None:
            raise ValueError(f"Camera {camera.name!r} has no payload configuration")

        width = getattr(payload, "width", None)
        height = getattr(payload, "height", None)
        if width is None or height is None:
            raise ValueError(
                f"Camera {camera.name!r} is missing width/height in its payload. Cannot determine video dimensions."
            )

        cameras.append(
            {
                "name": camera.name.lower(),
                "width": width,
                "height": height,
            }
        )

    return {
        "state_names": joint_names,
        "action_names": joint_names,
        "cameras": cameras,
    }


__all__ = ["build_raw_video_manifest_args"]
