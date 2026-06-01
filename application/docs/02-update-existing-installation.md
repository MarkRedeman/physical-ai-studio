# Update Existing Installation

Use this guide after pulling new changes, so the UI and backend stay in sync.

First make sure you are on the latest `main` branch,

```bash
git fetch origin
git rebase origin/main
```

## Update Docker setup
[//]: # (Screenshot suggestion: terminal showing docker compose build and docker compose up with successful output.)

From `application/docker/`:

```bash
docker compose build
docker compose up -d --force-recreate
```

Then open `http://localhost:7860` and confirm your project loads.

## Update native setup

### Upate the backend
To update the backend, first go to `./application/backend` and update its dependencies,

```bash
uv sync --extra xpu # or --extra cpu, --extra cuda
```

Then restart the backnd by running `./run.sh` as mentioned in `./01-installation.md`.

### Upate the UI

Go to `./application/ui` and update its dependencies,

```bash
npm install
```

and start the ui by running `npm run start`.

## Next

- If this is your first run on a new machine, go to `application/docs/01-installation.md`.
- For the full workflow, continue with `application/docs/03-getting-started.md`.
