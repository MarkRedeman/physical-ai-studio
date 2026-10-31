# Single-robot training recipes

These candidates target one recorded robot and one or two manipulation tasks.
They replace simulation workloads as Studio default references, not as benchmarks.
They have **not** been validated with PhysicalAI physical rollouts. Upstream runs
provide starting points, not guarantees for another arm, calibration, or camera setup.

| Policy | Recipe | Dataset | Batch | Training limit |
| --- | --- | --- | ---: | ---: |
| ACT | [SO101](act/so101/default.yaml) | `Daankrol/pick-and-place-multi-obj` | 8 | 10 epochs |
| SmolVLA | [SO100 pick-and-place](smolvla/so100/default.yaml) | `lerobot/svla_so100_pickplace` | 64 | 20,000 steps |
| MolmoAct2 | [Single robot, general checkpoint](molmoact2/single_robot.yaml) | `Daankrol/pick-and-place-multi-obj` | 16 | 8 epochs |

ACT trains the task policy from scratch with a pretrained image backbone.
SmolVLA trains its action expert/projections with the VLM frozen. MolmoAct2 uses
VLM LoRA and a fully trainable action expert; rank 64/alpha 16 match its native
policy defaults. Pi0.5, XR0, and RLDX-1 Studio defaults remain provisional.

## Reference evidence

- [Official SmolVLA guide](https://huggingface.co/docs/lerobot/main/smolvla):
  50 SO100 pick-and-place episodes, batch 64, 20k steps, overview/wrist mapping.
- [Real SO101 SmolVLA/ACT runs](https://ggando.com/blog/smolvla-so101):
  75 demonstrations, dual cameras, physical rollout results. Its ACT batch-64
  run resizes to 224x224; our batch-8 baseline keeps native ACT preprocessing.
  Its SmolVLA launcher enables augmentation; this candidate leaves it off.
- [Real WidowX ACT experiment](https://github.com/Nagarakshith1/lerobot_widowX_250):
  the [saved config](https://huggingface.co/msnaga/widowx_250_pick_and_place_block_policy/blob/main/train_config.json)
  has batch 8, 4 workers, LR 1e-5, no AMP/augmentation, and a 100k-step limit.
  Its README reports 60k steps; the exact completed budget is not established.
- [MolmoAct2 custom-data guidance](https://github.com/allenai/molmoact2/tree/main/experiments#fine-tuning-a-new-dataset):
  prefer LoRA for small datasets and describe robot/action semantics accurately.
  [Our specialized SO101 recipe](molmoact2/so101.yaml) additionally uses the
  SO100/101 checkpoint, joint-frame adaptation, Gaussian action initialization,
  and different optimizer/scheduler settings. Studio's general baseline does not
  reproduce those settings. Do not apply SO101 frame corrections to another robot.

## Choosing epochs or steps

Studio offers one active training limit: epochs (dataset passes) or steps
(optimizer updates, not samples). The recipe chooses the initial mode; users can
switch modes and edit the budget. Unit switching preserves separate drafts and
does not perform an automatic conversion. Auto batch size does not change a step
limit, but changes how many samples those updates consume.

Studio requests with neither limit keep the five-epoch fallback. If an older
payload contains both, epochs take precedence for compatibility. New step-only
requests are honored as steps, not silently replaced with five epochs. Job and
model summaries retain the selected unit. Remote trainers need an updated image
to accept step-budgeted jobs.

Library YAMLs can use either `trainer.max_epochs` or `trainer.max_steps` already.
Unlike Studio's compatibility rule, Lightning stops at the first of two supplied
limits. When switching a YAML's limit via CLI overrides, disable the other one:

```bash
# Switch the step-based SmolVLA recipe to epochs.
physicalai fit --config configs/physicalai/smolvla/so100/default.yaml \
  --trainer.max_steps=-1 --trainer.max_epochs=75

# Switch the epoch-based ACT recipe to optimizer steps.
physicalai fit --config configs/physicalai/act/so101/default.yaml \
  --trainer.max_epochs=-1 --trainer.max_steps=100000
```

### SnapFlow

Teacher and distillation budgets use the same unit. Distillation is additional,
not part of the teacher limit: 20k teacher steps plus 5k distillation steps means
25k total updates. In Studio recipes, use `SnapflowDefaults(distill_steps=5000)`
with `max_steps`, or `SnapflowDefaults(distill_epochs=3)` with `max_epochs`.
Step-budgeted job payloads use `snapflow_distill_steps`; epoch jobs retain
`snapflow_distill_epochs`. The library's `SnapFlowPhaseCallback` supports the
corresponding `start_step` or `start_epoch` boundary.

Step-budgeted Studio runs validate every 1,000 training batches across epochs,
or sooner for short runs/teachers. Studio currently uses one update per batch.
If no validation checkpoint is saved, the existing final-weight fallback is used.

### Estimating an alternative budget

The official SmolVLA dataset has 19,631 frames. Daankrol's dataset has 94,908
frames, 200 episodes, two cameras, and one task despite its "multi-obj" name.
For one device, without accumulation, a rough conversion is:

```text
steps_per_epoch = floor(training_frames / batch_size)
epochs = ceil(target_steps / steps_per_epoch)
```

A 10% episode holdout gives SmolVLA roughly 276 steps/epoch on the official
dataset, versus 1,334 on Daankrol's dataset. A 20k-step target therefore takes
about 73 versus 15 epochs. The SmolVLA recipe now uses steps directly. Use the actual
training split's frame count: random held-out episodes can have different lengths.
Changing batch size, Auto batch size, or dataset size changes the update budget.

## Your robot and task

From `library/`, replace the dataset **and its pinned revision** and adjust the
budget. Set `DATASET_REVISION` to your dataset's commit SHA, for example:

```bash
physicalai fit --config configs/physicalai/smolvla/so100/default.yaml \
  --data.init_args.repo_id=your-org/your-task \
  --data.init_args.revision="$DATASET_REVISION" --trainer.max_steps=20000
```

Map your scene camera to slot 0 and wrist camera to slot 1; the YAML names are
specific to the example dataset. Fine-tuning another robot needs its own recorded
actions and normalization statistics, not another arm's calibration. Studio's
mapping step handles camera names interactively.

Before a full run, check config resolution and run a one-batch smoke test. Validate
with held-out physical rollouts, not just imitation loss. Record the dataset
revision, camera mapping, hardware/software versions, effective batch, update
count, duration, and complete-task successes out of a stated number of trials.
For two tasks, balance demonstrations and report results separately per task.
The published two-task SmolVLA checkpoint is useful supporting evidence, but its
dataset is not publicly accessible and its card does not report rollout success.
