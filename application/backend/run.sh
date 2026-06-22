#!/bin/bash
set -euo pipefail

# -----------------------------------------------------------------------------
# run.sh - Script to run the Physical AI Studio server
#
# Features:
# - Runs database migrations on every start (idempotent via Alembic)
#
# Usage:
#   ./run.sh                    # Run server
#
# Environment variables:
#   APP_MODULE    Python module to run (default: src/main.py)
#   UV_CMD        Command to launch Uvicorn (default: "uv run")
#
# Requirements:
# - 'uv' CLI tool (Uvicorn) installed and available in PATH
# - Python modules and dependencies installed correctly
# -----------------------------------------------------------------------------

APP_MODULE=${APP_MODULE:-src/main.py}
UV_CMD=${UV_CMD:-uv run --no-sync}

export PYTHONUNBUFFERED=1

$UV_CMD physicalai-studio serve
