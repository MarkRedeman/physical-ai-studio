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

| **Project list**               | **After creating a new project**                                    |
|--------------------------------|---------------------------------------------------------------------|
| ![Project list][projects-list] | ![Setup a robot after creating a new project][projects-new-project] |

[projects-list]: ./assets/03-project-list.png
[projects-new-project]: ./assets/03-project-list.png

Open the app and go to Projects.

- Create a new project if none exists.
- Open the project you want to work in.

Next you will want to set up the project with robots and cameras.

## 2) Set up robots and cameras

| **SO101 Leader robot**                                     | **Trossen WidowX AI follower robot**                                 | **Robots list**                           |
|------------------------------------------------------------|----------------------------------------------------------------------|-------------------------------------------|
| ![Setup a new SO101 Leader robot][robots-new-so101-leader] | ![Setup a new WidowXAI follower robot][robots-new-widowxai-follower] | ![List of configured robots][robots-list] |

[robots-new-so101-leader]: ./assets/03-robots-new-so101-leader.png
[robots-new-widowxai-follower]: ./assets/03-robots-new-widowxai-follower.png
[robots-list]: ./assets/03-robots-list.png

In Robots:

- Add a follower robot.
- Add a leader (teleoperator) robot.

| **New USB Camera**                                   | **Cameras list**                            |
|------------------------------------------------------|---------------------------------------------|
| ![Setup a new overview camera][cameras-new-overview] | ![List of configured cameras][cameras-list] |

[cameras-new-overview]: ./assets/03-cameras-new-overview.png
[cameras-list]: ./assets/03-cameras-list.png

In Cameras:

- Add one or more cameras.
- Verify each camera preview before saving.

## 3) Create an environment

| **New environment**                          | **Environment list**                       |
|----------------------------------------------|--------------------------------------------|
| ![Setup a new environment][environments-new] | ![List of environments][environments-list] |

[environments-new]: ./assets/03-environments-new.png
[environments-list]: ./assets/03-environments-list.png

In Environments:

- Choose the follower and leader robot pair.
- Add cameras used for recording.
- Save the environment.

## 4) Create a dataset and record episodes

| **Create a new**                      | **Dataset overview**                         | **Dataset recording**                      |
|---------------------------------------|----------------------------------------------|--------------------------------------------|
| ![Create a new dataset][datasets-new] | ![Dataset overview][datasets-overview] | ![Record episodes][datasets-recording] |

[dataset-new]: ./assets/03-datasets-new.png
[dataset-overview]: ./assets/03-datasets-overview.png
[dataset-recording]: ./assets/03-datasets-recording.png

In Datasets:

- Click New Dataset.
- Select the environment.
- Name the dataset and set an optional task.

Open the dataset and start recording:

- Click Start episode.
- Perform the task.
- Click Accept to keep an episode, or Discard to drop it.

## 5) Train a model

| **Train a model**              | **Model list**                                    |
|--------------------------------|---------------------------------------------------|
| ![Train a model][models-train] | ![Model list with training progress][models-list] |

[models-train]: ./assets/03-models-train.png
[models-list]: ./assets/03-models-list.png

In Models:

- Click Train model.
- Pick a dataset.
- Select a policy.
- Click Train.

Watch the training job status in the Models page.

## 6) Run inference

| **Run a model**               | **Model inference screen**                  |
|-------------------------------|---------------------------------------------|
| ![Run a model][inference-run] | ![Model inference screen][inference-screen] |

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
