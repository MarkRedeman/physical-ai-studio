# Training Policies

| **Train model policy**                           | **Open model training logs**             | **Model formats**                                  |
|--------------------------------------------------|------------------------------------------|----------------------------------------------------|
| ![Train model policy][models-train-model-policy] | ![Open model training logs][models-logs] | ![Download optimized model formats][model-formats] |

[models-train-model-policy]: ./assets/07-model-train-policy.png
[model-logs]: ./assets/07-model-logs.png
[model-formats]: ./assets/07-model-formats.png

This guide describes how users train models from the Models page.

## Train a new model policy

Once you've collected enough episodes for your dataset you can begin to train a new model policy.
First you will have to choose the model policy, we currently support:

- ACT
- SmolVLA
- Pi0.5

Depending on the amount of VRAM your GPU supports
Under the advanced settings you may find options for changing the _batch size_, _training steps_, _amount of data workers_, _precision_ and an option to _compile model_ before training.
You may need to tune these settings to get an optimal result.

## Monitor training progress

| **Training job in progress**                              | **Open model training logs**             |
|-----------------------------------------------------------|------------------------------------------|
| ![Training job in progress][models-train-job-in-progress] | ![Open model training logs][models-logs] |

[model-train-job-in-progress]: ./assets/07-model-train-job-in-progress.png
[model-logs]: ./assets/07-model-logs.png

Once you start a training you can see its progress in the models screen. Click the job to see a live view of its loss curve.
You may also view the training logs.

When a training job takes too long and you can interrupt training. This will store a checkpoint of the current model and export the model to deployable formats. 

## Model formats

| **Model formats**                                  |
|----------------------------------------------------|
| ![Download optimized model formats][model-formats] |

[model-formats]: ./assets/07-model-formats.png

When training finishes we export the model to all its supported backends: [PyTorch](https://github.com/pytorch/pytorch), [OpenVINO](https://github.com/openvinotoolkit/openvino), [ONNX](https://github.com/onnx/onnx) and [ExecuTorch](https://github.com/pytorch/executorch).
Download the model and then use [OpenVINO PhysicalAI](https://github.com/openvinotoolkit/physicalai) to deploy it on your hardware.

## Retraining flow

| **Retrain model from menu**                          | **Retrain model configuration**                              |
|------------------------------------------------------|--------------------------------------------------------------|
| ![Retrain model from menu][models-retrain-from-menu] | ![Retrain model configuration][models-retrain-configuration] |
|                                                      |                                                              |

From model actions, users can choose retrain.
This will use the same training configuration as its original model.

## Next

- Run/deploy in UI: `application/docs/08-deploying-model-policies.md`.
