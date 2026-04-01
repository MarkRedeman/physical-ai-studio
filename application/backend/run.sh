#!/bin/bash
set -euo pipefail

# -----------------------------------------------------------------------------
# run.sh - Script to run the Physical AI Studio server
#
# Features:
# - Runs database migrations on every start (idempotent via Alembic)
# - Optionally seeds the database before starting the server by setting:
#     SEED_DB=true
#
# Usage:
#   SEED_DB=true ./run.sh       # Seed database before launching server
#   ./run.sh                    # Run server without seeding
#
# Environment variables:
#   SEED_DB       If set to "true", runs database seeding before starting.
#   APP_MODULE    Python module to run (default: src/main.py)
#   UV_CMD        Command to launch Uvicorn (default: "uv run")
#
# Requirements:
# - 'uv' CLI tool (Uvicorn) installed and available in PATH
# - Python modules and dependencies installed correctly
# -----------------------------------------------------------------------------

SEED_DB=${SEED_DB:-false}
APP_MODULE=${APP_MODULE:-src/main.py}
UV_CMD=${UV_CMD:-uv run --no-sync}

# Timestamped echo — mirrors the loguru format used by the Python app.
log() { echo "$(date '+%Y-%m-%d %H:%M:%S.%3N') | INFO     | run.sh - $*"; }

export PYTHONUNBUFFERED=1
export PYTHONPATH=src

# Always run migrations — Alembic is idempotent and will skip
# already-applied migrations. This ensures the persistent volume
# has an up-to-date schema regardless of how it was created.
log "Running database migrations..."
$UV_CMD src/cli.py migrate

if [[ "$SEED_DB" == "true" ]]; then
    log "Seeding the database..."
    $UV_CMD application/cli.py init-db
    $UV_CMD application/cli.py seed --with-model=True
fi

log "Starting FastAPI server..."

# Export process start time so the application can measure total startup
# wall-time including Python interpreter and module import overhead.
export _RUN_SH_START_MS=$(date +%s%3N)
log "$UV_CMD $APP_MODULE"
exec $UV_CMD "$APP_MODULE"
