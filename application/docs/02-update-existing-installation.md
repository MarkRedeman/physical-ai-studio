# Update Existing Installation

Use this guide after pulling new changes, so the UI and backend stay in sync.

## What users usually notice before an update
[//]: # (Screenshot suggestion: UI example where a page is stale or missing a recently added button/feature.)

- A new feature is missing from the UI.
- A button or flow behaves differently than expected.
- A page does not load after repository changes.

Refreshing your local installation usually resolves this.

## Update Docker setup
[//]: # (Screenshot suggestion: terminal showing docker compose build and docker compose up with successful output.)

From `application/docker/`:

```bash
docker compose build
docker compose up
```

Then open `http://localhost:7860` and confirm your project loads.

## Update native setup
[//]: # (Screenshot suggestion: two terminal windows, one for backend and one for UI, both restarted after pulling latest changes.)

If you run backend and UI separately, refresh both:

1. Backend dependencies and server.
2. UI dependencies and dev server.

Follow the exact commands in `application/README.md` for your hardware profile.

## UI verification checklist
[//]: # (Screenshot suggestion: project open with Robots, Datasets, Models tabs visible and one dialog open, such as New Dataset or Train model.)

After updating, verify in the UI:

- You can open a project.
- Robots, Datasets, and Models pages load.
- You can open at least one dialog (for example, New Dataset or Train model).

## Next
[//]: # (Screenshot suggestion: optional docs navigation screenshot linking back to Installation and forward to Getting Started.)

- If this is your first run on a new machine, go to `application/docs/01-installation.md`.
- For the full workflow, continue with `application/docs/03-getting-started.md`.
