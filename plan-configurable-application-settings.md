# Plan: Configurable Application Settings

Status: agreed design for implementation
Scope: `application/backend` (settings refactor, JSON config persistence, settings API, training wiring); optional UI follow-up.

## Goal

Let operators/users customize a subset of the backend settings without code changes:

- **Streaming / video-encoding settings** (`streaming` group)
- **Trainer (remote-training client) timeouts** (`trainer` group)
- **Hugging Face token** (`huggingface` group) — follow-up surface for more HF settings
- **Training-run logger** (wandb) — `logger` group

Configuration surfaces, in precedence order (highest first):

1. `settings.json` — a JSON config file under the storage dir, editable by hand or via the settings API/UI.
2. Environment variables / `.env` — flat, environment-only fields (server, storage, db, import-safety, ...).
3. Field defaults.

The user-configurable groups are configured **only** through `settings.json` / the settings API. There is **no migration of legacy flat settings formats** — the old `STREAMING_*`/`TRAINER_*`-style env vars are not bridged into the groups.

## Current state

- `application/backend/src/settings.py` defines one flat `pydantic_settings.BaseSettings` (`Settings`) reading `.env`, with per-field aliases, `case_sensitive=False`, `extra="ignore"`, and a `@lru_cache get_settings()`.
- Streaming settings are consumed in `workers/robot_control_worker.py:103` (`settings.streaming_*`) to build a lerobot `StreamingEncodingSettings`.
- Trainer client timeouts are consumed in `services/training_backends/remote.py` (`settings.trainer_*`).
- `HF_TOKEN` is documented in `.env.example` but never actually read.
- Training-run logging is hardcoded `CSVLogger` in `training/job.py:run_training_job`; the library `Trainer` accepts any Lightning logger (e.g. `WandbLogger`).

## Key findings (verified against pydantic-settings 2.14.2)

- `JsonConfigSettingsSource` maps **nested JSON keys** into nested Pydantic models and **merges with field defaults** (unset sub-fields keep defaults). A missing file is a no-op (falls back to defaults).
- Source precedence is controllable via `settings_customise_sources`.
- **Env vars cannot populate nested models when child field names contain underscores**: `STREAMING_PIX_FMT` / `TRAINER_REQUEST_TIMEOUT_S` do not reach `streaming.pix_fmt` / `trainer.request_timeout_s`, even with `env_nested_delimiter`. Since the groups are configured from `settings.json` only (no legacy env migration), this is not a constraint we work around.

See also https://pydantic.dev/docs/validation/latest/concepts/pydantic_settings/#other-settings-source

## Design

### 1. Nested grouped settings (`settings.py`)

New nested `pydantic.BaseModel`s:

- `StreamingSettings`: `vcodec`, `pix_fmt`, `crf`, `preset`, `extra_options`, `encoder_threads`, `encoder_queue_maxsize`.
- `TrainerClientSettings`: `request_timeout_s`, `download_read_timeout_s`, `stream_reconnect_max_s`, `stream_reconnect_backoff_max_s` (named to avoid clashing with the separate trainer service's `TrainerSettings`).
- `HuggingFaceSettings`: `hf_token: SecretStr | None`.
- `LoggerSettings`: `providers: list[Literal["csv", "tensorboard", "wandb"]]`, `wandb_project`, `wandb_entity`, `wandb_api_key: SecretStr | None`. Multiple providers (e.g. CSV **and** wandb) run simultaneously — Lightning's `Trainer` accepts a list of loggers.

`Settings` gains fields `streaming`, `trainer`, `huggingface`, `logger`. Existing env-only flat fields (app, server, db, storage, import-safety, proxy) are unchanged.

### 2. JSON config file (the UI/operator surface)

- Path: `SETTINGS_FILE` env override, else `<storage_dir>/settings.json` (helper `get_settings_file_path()`).
- Loaded via `JsonConfigSettingsSource` wired in `settings_customise_sources`, positioned **above** the env source so the file wins.
- File shape mirrors the model:

```json
{
  "streaming": { "vcodec": "libx264", "crf": 23 },
  "trainer": { "request_timeout_s": 30.0 },
  "huggingface": { "hf_token": "hf_..." },
  "logger": { "provider": "wandb", "wandb_project": "studio" }
}
```

- Written atomically (temp file + rename) by `write_user_settings()` so a crash never leaves a truncated file.
- Secrets are stored plaintext on disk (same trust model as `.env`); `SecretStr` keeps them out of reprs/logs, and the API returns them masked.

### 3. No legacy env migration

The groups are not read from environment variables. Setting `HF_TOKEN` in the environment still works for the training libraries themselves (huggingface_hub reads it directly), but the configurable groups — including the tucked-away `huggingface`/`logger` values — come from `settings.json`.

### 4. Freshness

Drop `@lru_cache` from `get_settings()` — the env + file parse is cheap, and consumers call it at point of use, so API/UI changes propagate to worker processes without restart. Modules that capture settings at import (`db/engine.py`, `core/logging/log_config.py`) only use env-level storage/db/debug settings and are unaffected.

### 5. Settings API (`api/settings.py`)

- `GET /api/settings` — returns effective settings (grouped; streaming, trainer, huggingface, logger) with secrets masked (`"********"` when set), plus the existing `geti_action_dataset_path`.
- `PUT /api/settings` — validates the payload against the configurable subset, writes `settings.json` atomically. Returns the updated effective settings.

Only the grouped, user-configurable settings are writable. Env-only fields (host, port, storage, db, ...) are not exposed for editing.

### 6. Runtime wiring

- `workers/robot_control_worker.py` — build `StreamingEncodingSettings` from `settings.streaming.*`.
- `services/training_backends/remote.py` — read `settings.trainer.*`.
- `training/job.py` — `RunOptions` (resume checkpoint, `LoggerSettings`, HF token) is now a field `run_options` on `TrainingJobSpec`, consumed by `run_training_job`; default stays `CSVLogger`. Sets `os.environ["HF_TOKEN"]` when a token is supplied.
  - `services/training_backends/local.py` (local): populates `spec.run_options` from `settings.logger` and `settings.huggingface.hf_token`. `training/logging.py:build_training_logger` builds the Lightning logger(s) — one or a list, so CSV **and** wandb run together.
  - `trainer/runner.py` (remote): unchanged defaults (CSVLogger, no token) — the spec arrives with null `run_options`, so the remote trainer service reads its own env.

## File changes

- `application/backend/src/settings.py` — nested models, JSON source, file helpers, drop cache.
- `application/backend/src/api/settings.py` — GET/PUT settings API.
- `application/backend/src/workers/robot_control_worker.py` — grouped streaming access.
- `application/backend/src/services/training_backends/remote.py` — grouped trainer access.
- `application/backend/src/training/job.py` — `logger`/`hf_token` args.
- `application/backend/src/services/training_backends/local.py` + new `training/logging.py` — build the Lightning logger, pass token.
- `application/backend/.env.example` — document `SETTINGS_FILE`; groups are configured via `settings.json`.
- `application/backend/tests/test_settings.py` and new tests — JSON precedence, merge, atomic write, API redaction/persistence.
- Optional follow-up: React settings page under `application/ui` + regenerated API types.

## Decisions recorded

- **JSON file (UI) beats env** for the configurable groups (`settings_customise_sources` ordering); env holds the flat, environment-only fields.
- **No legacy env migration** — `STREAMING_*`/`TRAINER_*`-style env vars are not bridged into the groups; configure them via `settings.json`/API. (An unset `...` in `.env` is simply ignored.)
- **HF token is not transmitted to remote trainers** in this iteration; the trainer service reads its own `HF_TOKEN` env. (Secure-transmission via HTTPS-gated job payload is a follow-up.)
- **No SQLite persistence** — `JsonConfigSettingsSource` + atomic JSON writes replace the earlier DB-table plan (no migration, no repository/service/session plumbing).

## Testing

- `prek run --all-files application/backend/` (ruff, mypy, formatting).
- Backend pytest suite for the backend.
- New unit tests:
  - JSON file sets group values; missing file is a no-op;
  - group values are not read from env;
  - partial JSON merges with defaults;
  - `write_user_settings` round-trips and is atomic;
  - GET masks secrets; PUT persists and is reflected in a fresh `get_settings()`.
