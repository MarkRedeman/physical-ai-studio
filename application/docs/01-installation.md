# Installation

This guide helps you install Physical AI Studio and confirm that the UI is ready.

## Choose how to run
[//]: # (Screenshot suggestion: side-by-side view of the docs snippets showing Docker and native options, with Docker visually highlighted as recommended.)

Most users should start with Docker:

- Docker starts both backend and UI together.
- You open one URL and can begin setting up projects right away.

If you already run backend and UI separately, you can keep using the native setup from `application/README.md`.

## Install with Docker (recommended)
[//]: # (Screenshot suggestion: terminal in application/docker showing cp .env.example .env and docker compose up completing successfully.)

From `application/docker/`:

```bash
cp .env.example .env
docker compose up
```

When startup is done, open `http://localhost:7860`.

## First UI check
[//]: # (Screenshot suggestion: first loaded app screen showing Projects and, after opening a project, the Robots/Datasets/Models navigation tabs.)

After opening the app, you should see:

- A Projects view.
- The main project navigation after you create/select a project:
  - Robots
  - Datasets
  - Models

If the page opens and you can create a project, your installation is ready.

## If something does not open
[//]: # (Screenshot suggestion: Docker Desktop or terminal status indicating containers running, plus browser URL bar at localhost:7860.)

- Confirm Docker is running.
- Confirm no other app is already using the same port.
- Restart the stack from `application/docker/`.

## Next
[//]: # (Screenshot suggestion: optional docs navigation screenshot showing links to Getting Started and Update Existing Installation.)

- Continue with `application/docs/03-getting-started.md`.
- If you are upgrading an existing setup, use `application/docs/02-update-existing-installation.md`.
