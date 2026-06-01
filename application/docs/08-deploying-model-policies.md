# Deploying Model Policies

| **Run model policy**               | **Model inference screen**                  |
|------------------------------------|---------------------------------------------|
| ![Run model policy][inference-run] | ![Model inference screen][inference-screen] |

[inference-run]: ./assets/08-inference-run.png
[inference-screen]: ./assets/08-inference-screen.png

Models can be deployed using [OpenVINO PhysicalAI](https://github.com/openvinotoolkit/physicalai) or can run inside of Physical AI Studio.
When on the models screen press the "Run model" button of your newly trained policy, then select the inference backend and the inference device. We currently support PyTorch and OpenVINO as runtimes in Studio using either GPU or CPU.

When you start the model we will load the environment used to record the dataset the model was trained on.
Once we finished loading the environment and model you will see a similar screen as used when recording the dataset. Pick the task the model should perform and press "Play".

## Next

- For training, see `application/docs/07-training-policies.md`.
