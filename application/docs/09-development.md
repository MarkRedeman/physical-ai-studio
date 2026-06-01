# Development

This chapter is for users who run the app locally while developing or testing changes.

## Run the UI in development
[//]: # (Screenshot suggestion: terminal running npm run start and browser open at localhost:3000.)

From `application/ui/`:

```bash
npm install
npm run start
```

Open `http://localhost:3000`.

## Run backend in development
[//]: # (Screenshot suggestion: backend terminal window running service alongside UI terminal window.)

From `application/backend/`, run the backend as documented in `application/README.md`.

When both UI and backend are running, you can use the full app flow locally.

## What users can test in the UI
[//]: # (Screenshot suggestion: quick montage of Robots, Datasets, and Models pages used during local testing.)

- Project creation and navigation.
- Robot/camera/environment setup flows.
- Dataset recording and management.
- Model training and inference actions.

## Useful developer-facing UI page
[//]: # (Screenshot suggestion: /openapi page displayed in the app.)

You can open the built-in API docs page in the UI at:

- `/openapi`

This is useful when validating requests/responses while building features.

## Next
[//]: # (Screenshot suggestion: optional docs links screenshot to Hardware Integration and REST API chapters.)

- Hardware onboarding: `application/docs/10-integrating-new-hardware.md`.
- REST API usage: `application/docs/12-using-the-rest-api.md`.
