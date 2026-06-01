# Training Policies

This guide describes how users train models from the Models page.

## Open training dialog
[//]: # (Screenshot suggestion: Models page with Train model button and the training dialog opened.)

In a project, go to Models and click Train model.

Users will see a dialog with:

- Dataset picker.
- Policy cards.
- Advanced settings.

## Choose a policy
[//]: # (Screenshot suggestion: policy card selection area showing ACT, SmolVLA, and Pi0.5 options.)

Current policy options shown in the UI include:

- ACT
- SmolVLA
- Pi0.5

Each card includes a short description to help users choose.

## Set training options
[//]: # (Screenshot suggestion: advanced settings section expanded with Max Steps, Batch Size, Data Workers, Precision, and Compile model.)

Users can configure:

- Max Steps
- Batch Size
- Data Workers
- Precision
- Compile model

Advanced settings are available in an expandable section.

## Start and monitor
[//]: # (Screenshot suggestion: Models page job table showing a running training job and status updates.)

After users click Train:

- A training job appears in the Models page.
- Users can monitor status and progress.
- Users can interrupt jobs if needed.

When complete, the trained model appears in the model list.

## Retraining flow
[//]: # (Screenshot suggestion: model actions menu with Retrain option and resulting retrain dialog state.)

From model actions, users can choose Retrain.

The retrain dialog pre-fills from the selected model context and starts a new training job.

## Next
[//]: # (Screenshot suggestion: optional docs links screenshot to Exporting Model Policies and Deploying Model Policies chapters.)

- Export model artifacts: `application/docs/071-exporting-model-policies.md`.
- Run/deploy in UI: `application/docs/08-deploying-model-policies.md`.
