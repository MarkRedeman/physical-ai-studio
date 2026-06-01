# Recording Datasets

| **Create a new**                      | **Dataset overview**                         | **Dataset recording**                      |
|---------------------------------------|----------------------------------------------|--------------------------------------------|
| ![Create a new dataset][datasets-new] | ![Dataset overview][datasets-overview] | ![Record episodes][datasets-recording] |

[dataset-new]: ./assets/03-datasets-new.png
[dataset-overview]: ./assets/03-datasets-overview.png
[dataset-recording]: ./assets/03-datasets-recording.png

This guide covers what users do in the UI to collect demonstration episodes.

## 1) Create a dataset
[//]: # (Screenshot suggestion: New Dataset dialog showing Environment picker, Dataset name, Task, and Save button.)

Start out by creating a dataset, provide it a name and a default task. The environment that you select will be used for recording episodes for the dataset.

### Import a dataset

| **Upload dataset**                        | **Import dataset**                 |
|-------------------------------------------|------------------------------------|
| ![Upload dataset][datasets-import-upload] | ![Import dataset][datasets-import] |

[dataset-import-upload]: ./assets/05-import-dataset-upload.png
[dataset-import]: ./assets/05-import-dataset.png

Alternatively you can import either a [lerobot v3](https://huggingface.co/docs/lerobot/lerobot-dataset-v3) dataset or an exported dataset from Physical AI Studio.
Similarly to adding a dataset set we need a name. The default task and environment for the dataset can be selected after uploading your dataset.

## 2) Open recording mode

| **No episodes yet**                      | **Dataset recording preview**                  | **Dataset recording**                  |
|------------------------------------------|----------------------------------------|----------------------------------------|
| ![No episodes yet][datasets-no-episodes] | ![Record episodes preview][datasets-recording-screen] | ![Recording an episode][datasets-recording] |

[dataset-no-episodes]: ./assets/05-no-episodes.png
[dataset-recording-screen]: ./assets/05-datasets-recording-screen.png
[dataset-recording]: ./assets/05-datasets-recording.png

Before you start recording episodes please make sure that both follower and leader arm are free to move.
Once you start recording the follower arm will follow the same movements as the leader.

From your dataset page, start recording by pressing the "Add episode".
Once your environment has finished loading you will see your camera feeds as well as a visualization of your follower robot.

In the top right you can also see the total episodes recorded in your dataset.
We recommend recording at least 50 episodes before you start training a model.
    
Typically recording episodes is done in a loop:

1. Reset your physical scene.
2. Enter or confirm the task.
3. Click Start episode.
4. Perform the demonstration.
5. Click Accept or Discard.

Keyboard shortcuts in recording view:

- Right arrow: Start episode or Accept.
- Left arrow: Discard.

## 3) Exporting datasets

At any time you may export a model to a [lerobot v3](https://huggingface.co/docs/lerobot/lerobot-dataset-v3) format.
You can import this back into Physical AI Studio on another system, use [physicalai-train](https://github.com/open-edge-platform/physical-ai-studio/tree/main/library) or [lerobot](https://huggingface.co/docs/lerobot/index) to train models using your dataset outside of the Studio.

## Next

- Manage and curate datasets in `application/docs/06-dataset-management.md`.
- Start training in `application/docs/07-training-policies.md`.
