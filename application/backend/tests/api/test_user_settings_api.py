# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the user-configurable settings API."""

from pathlib import Path

from fastapi.testclient import TestClient

from main import app
from settings import get_settings, write_user_settings


def test_get_settings_masks_secrets(monkeypatch, tmp_path: Path) -> None:
    settings_file = tmp_path / "settings.json"
    monkeypatch.setenv("SETTINGS_FILE", str(settings_file))
    write_user_settings(
        {
            "huggingface": {"hf_token": "super-secret"},
            "logger": {"providers": ["csv", "wandb"], "wandb_api_key": "api-secret", "wandb_project": "studio"},
        }
    )

    with TestClient(app) as client:
        response = client.get("/api/settings")

    assert response.status_code == 200
    body = response.json()
    assert body["streaming"]["vcodec"] == "auto"
    assert body["geti_action_dataset_path"].endswith("datasets")
    assert "super-secret" not in response.text
    assert "api-secret" not in response.text
    assert body["huggingface"]["hf_token"] is not None
    assert body["logger"]["wandb_project"] == "studio"


def test_patch_settings_persists_only_provided_fields(monkeypatch, tmp_path: Path) -> None:
    settings_file = tmp_path / "settings.json"
    monkeypatch.setenv("SETTINGS_FILE", str(settings_file))

    with TestClient(app) as client:
        response = client.patch(
            "/api/settings",
            json={
                "streaming": {"vcodec": "libx264", "crf": 23},
                "trainer": {"request_timeout_s": 45.0},
                "logger": {"providers": ["csv", "wandb"], "wandb_project": "prod"},
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["streaming"]["vcodec"] == "libx264"
    assert body["trainer"]["request_timeout_s"] == 45.0
    assert body["logger"]["providers"] == ["csv", "wandb"]

    # A second patch touching one group leaves the others untouched.
    with TestClient(app) as client:
        response = client.patch("/api/settings", json={"streaming": {"encoder_threads": 4}})

    assert response.status_code == 200
    body = response.json()
    assert body["streaming"]["encoder_threads"] == 4
    assert body["streaming"]["vcodec"] == "libx264"
    assert body["trainer"]["request_timeout_s"] == 45.0
    assert body["logger"]["providers"] == ["csv", "wandb"]

    # A fresh settings read sees the persisted values.
    fresh = get_settings()
    assert fresh.streaming.vcodec == "libx264"
    assert fresh.streaming.encoder_threads == 4
    assert fresh.trainer.request_timeout_s == 45.0
    assert fresh.logger.providers == ["csv", "wandb"]


def test_patch_settings_preserves_omitted_secrets(monkeypatch, tmp_path: Path) -> None:
    settings_file = tmp_path / "settings.json"
    monkeypatch.setenv("SETTINGS_FILE", str(settings_file))
    write_user_settings(
        {
            "huggingface": {"hf_token": "super-secret"},
            "logger": {"providers": ["wandb"], "wandb_api_key": "api-secret"},
        }
    )

    # Only the streaming group is provided: secrets must survive untouched.
    with TestClient(app) as client:
        response = client.patch("/api/settings", json={"streaming": {"vcodec": "libsvtav1"}})

    assert response.status_code == 200
    body = response.json()
    assert body["streaming"]["vcodec"] == "libsvtav1"
    assert "super-secret" not in response.text
    assert "api-secret" not in response.text
    assert body["huggingface"]["hf_token"] is not None
    assert body["logger"]["wandb_api_key"] is not None

    fresh = get_settings()
    assert fresh.huggingface.hf_token is not None
    assert fresh.huggingface.hf_token.get_secret_value() == "super-secret"
    assert fresh.logger.wandb_api_key is not None
    assert fresh.logger.wandb_api_key.get_secret_value() == "api-secret"


def test_patch_settings_clears_token_with_explicit_null(monkeypatch, tmp_path: Path) -> None:
    settings_file = tmp_path / "settings.json"
    monkeypatch.setenv("SETTINGS_FILE", str(settings_file))
    write_user_settings({"huggingface": {"hf_token": "super-secret"}})

    with TestClient(app) as client:
        response = client.patch("/api/settings", json={"huggingface": {"hf_token": None}})

    assert response.status_code == 200
    assert get_settings().huggingface.hf_token is None


def test_patch_settings_ignores_unknown_groups(monkeypatch, tmp_path: Path) -> None:
    settings_file = tmp_path / "settings.json"
    monkeypatch.setenv("SETTINGS_FILE", str(settings_file))

    with TestClient(app) as client:
        response = client.patch(
            "/api/settings",
            json={"streaming": {"vcodec": "libx264"}, "host": "10.0.0.1"},
        )

    assert response.status_code == 200
    # Env-only fields are not persisted via the API.
    assert get_settings().host == "0.0.0.0"
    assert get_settings().streaming.vcodec == "libx264"
