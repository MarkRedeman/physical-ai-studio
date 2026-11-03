# StarArm follower placeholder — not measured calibration

`stararm.json` is a software-test placeholder for the pinned upstream
`stararm102_fl` driver with its default joint IDs and configuration. It assumes
the physical servo origins are already correct. It neither sets nor verifies
them. Do not use it as evidence that an arm has been calibrated.

The seven entries use zero homing offsets and drive mode zero, matching the
current follower's calibration format. Ranges are derived from upstream's
configured joint ranges converted back to raw servo degrees by dividing by
`joint_directions` and sorting the endpoints. They are **not measurements**
or validated safe mechanical limits. In particular, the gripper's configured
`[-270, 0]` divided by its direction multiplier `-6` becomes raw `[0, 45]`.

This driver reads and writes with `normalize=False`; it uses configuration
`joint_directions` and `joint_ranges`, not these JSON ranges, to transform and
clamp commands. Loading this file satisfies its joint-name-based calibration
check without actually validating the hardware. This file is not a motion
safety mechanism.

To opt into using the placeholder in Studio, select
`LeRobot_stararm102_fl_Follower` and set:

```json
{
  "id": "stararm",
  "calibration_dir": "/home/mark/projects/intel/physical-ai-studio/stararm-calibration-placeholder"
}
```

The file is intentionally not installed under LeRobot's default calibration
cache. No hardware connection was made to create or validate it. Confirm the
expected physical reference pose and raw readings before enabling any action
source. Even connecting the upstream driver unlocks joints and resets
multi-turn counts: secure/support the arm first. Use a standalone read-only
check, not a teleoperation or policy session, for initial inspection.

Do not use this for the legacy `lerobot_robot_stararm102` follower, whose
joint names and normalization differ, or assume this is a measured leader
calibration. For actual calibration, follow `../debugging-with-stararms.md`.
