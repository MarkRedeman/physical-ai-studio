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
[//]: # (Screenshot suggestion: two terminal windows, one for backend and one for UI, both restarted after pulling latest changes.)

If you run backend and UI separately, then startup the backend and Ui separetly after pulling the new changes with git.

Follow the same steps in `./01-installation.md` for your hardware profile.

## Next

- If this is your first run on a new machine, go to `application/docs/01-installation.md`.
- For the full workflow, continue with `application/docs/03-getting-started.md`.
