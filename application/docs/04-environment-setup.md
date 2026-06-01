# Environment Setup

In Physical AI Studio, an environment combines:

- Robots (follower + teleoperator)
- Cameras

Datasets and model runs depend on this setup, so create it carefully in the UI.

## 1) Add robots
[//]: # (Screenshot suggestion: Robots section landing view with list of configured robots and Add new robot action.)

Open the Robots section and click to add a robot.

What you see in the UI depends on robot type.

### SO101 flow
[//]: # (Screenshot suggestion: SO101 setup wizard stepper showing Diagnostics, Setup Motors, Calibration, Verification, Save Robot.)

For SO101, the UI can guide you through a setup wizard.

Typical wizard steps you will see:

- Diagnostics
- Setup Motors
- Calibration
- Verification
- Save Robot

### Other supported robot flow
[//]: # (Screenshot suggestion: non-SO101 robot creation form with connection details fields and Save action.)

For other robot types, you fill in connection details directly (for example IP address) and save.

## 2) Add cameras
[//]: # (Screenshot suggestion: camera creation form showing driver/device selector, resolution/FPS fields, and live preview.)

Open Cameras and add each camera you need.

The UI supports multiple camera drivers and only shows what is available.

During setup, confirm:

- You selected the correct device.
- Resolution and FPS look right.
- Preview feed is correct before saving.

## 3) Create environment
[//]: # (Screenshot suggestion: environment form populated with follower, leader, and camera list before Save.)

Open Environments and create a new one.

In the form, you will:

- Select the follower robot.
- Select the teleoperator (leader) robot.
- Add one or more cameras.
- Save the environment.

After saving, open the environment and verify all robots/cameras appear as expected.

## Common UI checks
[//]: # (Screenshot suggestion: environment details view confirming attached robots/cameras and healthy statuses.)

Before moving on, confirm:

- Robot status looks correct in lists/details.
- Camera previews are stable.
- Environment includes the exact devices you expect.

## Next
[//]: # (Screenshot suggestion: optional docs navigation screenshot to Recording Datasets and Hardware Integration chapters.)

- Continue with recording in `application/docs/05-recording-datasets.md`.
- For hardware-specific troubleshooting, see `application/docs/10-integrating-new-hardware.md`.
