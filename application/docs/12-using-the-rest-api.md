# Using the REST API

This chapter explains the API from a user point of view: where to find it and how it maps to UI actions.

## Open API documentation in the UI
[//]: # (Screenshot suggestion: in-app /openapi page with endpoint navigation panel visible.)

Physical AI Studio includes an API reference page in the app:

- Open `/openapi`.

Users can browse endpoints and example schemas there.

## How API maps to common UI actions
[//]: # (Screenshot suggestion: split view concept showing a UI action (for example Train model or Import dataset) next to matching API endpoint name.)

Most UI workflows call REST endpoints behind the scenes.

Examples users care about:

- Creating and listing projects.
- Managing robots, cameras, and environments.
- Creating datasets, listing episodes, importing/exporting data.
- Starting training jobs and checking job status.
- Downloading trained models.

## API behavior users should expect
[//]: # (Screenshot suggestion: job status UI progressing through pending/running/completed for an async action.)

- Some workflows are asynchronous and appear as jobs in the UI.
- Import and training update status over time, not instantly.
- Certain real-time experiences (camera/robot control) use WebSocket endpoints.

## Practical usage pattern
[//]: # (Screenshot suggestion: simple workflow visual: open /openapi, trigger action in UI, observe job completion.)

If you are a power user integrating with external tools:

1. Inspect endpoint in `/openapi`.
2. Mirror the same flow used by the UI.
3. Track asynchronous jobs until completion.

## Next
[//]: # (Screenshot suggestion: optional docs links screenshot to Development and dataset import/export chapters.)

- Development workflow: `application/docs/09-development.md`.
- Dataset import/export: `application/docs/061-exporting-datasets.md` and `application/docs/062-importing-datasets.md`.
