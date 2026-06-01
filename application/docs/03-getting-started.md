# Getting Started

This is the fastest path from a fresh install to your first trained model in the UI.

## Goal
[//]: # (Screenshot suggestion: overview collage of the main UI areas used in this flow: Projects, Robots, Datasets, Models.)

By the end of this guide, you will:

- Create a project.
- Configure robots and cameras.
- Create an environment.
- Record a dataset.
- Train a model.
- Run inference with that model.

## 1) Create or open a project
[//]: # (Screenshot suggestion: Projects page with at least one project card/list item and the selected project state.)

Open the app and go to Projects.

- Create a new project if none exists.
- Open the project you want to work in.

When a project is selected, you will use the tabs:

- Robots
- Datasets
- Models

## 2) Set up robots and cameras
[//]: # (Screenshot suggestion: Robots page and Cameras page showing Add actions and configured entries.)

In Robots:

- Add a follower robot.
- Add a leader (teleoperator) robot.

In Cameras:

- Add one or more cameras.
- Verify each camera preview before saving.

## 3) Create an environment
[//]: # (Screenshot suggestion: New Environment form with follower/leader selection and camera selection visible.)

In Environments:

- Choose the follower and leader robot pair.
- Add cameras used for recording.
- Save the environment.

## 4) Create a dataset and record episodes
[//]: # (Screenshot suggestion: New Dataset dialog and recording screen showing Start episode / Accept / Discard controls.)

In Datasets:

- Click New Dataset.
- Select the environment.
- Name the dataset and set an optional task.

Open the dataset and start recording:

- Click Start episode.
- Perform the task.
- Click Accept to keep an episode, or Discard to drop it.

## 5) Train a model
[//]: # (Screenshot suggestion: Train model dialog with dataset picker, policy cards, and Train button.)

In Models:

- Click Train model.
- Pick a dataset.
- Select a policy.
- Click Train.

Watch the training job status in the Models page.

## 6) Run inference
[//]: # (Screenshot suggestion: Run model dialog with backend selection, then inference view after clicking Start.)

On a trained model:

- Click Run model.
- Select a backend.
- Click Start.

The inference view opens where you can start/stop model-driven execution.

## Next
[//]: # (Screenshot suggestion: optional docs links screenshot for Environment Setup, Recording Datasets, and Training Policies chapters.)

- For setup details, continue with `application/docs/04-environment-setup.md`.
- For recording best practices, see `application/docs/05-recording-datasets.md`.
- For training details, see `application/docs/07-training-policies.md`.
