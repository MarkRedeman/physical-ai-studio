# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Schemas for backend-discovered recording profiles."""

from pydantic import BaseModel


class StreamingProfile(BaseModel):
    """One recording configuration the current backend has verified."""

    id: str
    name: str
    description: str


class StreamingProfiles(BaseModel):
    """The recording profiles available on the current backend."""

    profiles: list[StreamingProfile]
