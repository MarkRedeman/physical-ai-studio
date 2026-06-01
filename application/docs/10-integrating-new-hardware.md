# Integrating New Hardware

This guide describes the user-visible flow for adding new robot or camera hardware into a project.

## Hardware users can add in the UI
[//]: # (Screenshot suggestion: hardware-related navigation areas showing Robots and Cameras sections.)

Users can add:

- Robots (for follower/leader roles)
- Cameras (USB, IP, and other supported drivers shown in UI)

The exact options shown depend on detected hardware and current support.

## Add a robot
[//]: # (Screenshot suggestion: add robot form with robot type selector and connection detail fields.)

In Robots:

1. Click to add a new robot.
2. Choose robot type.
3. Enter/select connection details.
4. Save (or run setup wizard if offered).

For SO101 robots, users may see a guided setup wizard before saving.

## Add a camera
[//]: # (Screenshot suggestion: add camera form with driver/device selection and preview pane.)

In Cameras:

1. Click to add a camera.
2. Select driver/device.
3. Configure resolution/FPS.
4. Check preview.
5. Save.

## Validate hardware in UI
[//]: # (Screenshot suggestion: robot detail and camera detail pages confirming status/preview after setup.)

After adding hardware, users should verify:

- Robot appears in list and can be opened.
- Robot status is reported as expected.
- Camera appears with correct preview.

Then attach hardware to an environment and test recording/inference flow.

## If hardware is not visible
[//]: # (Screenshot suggestion: UI state where a device is missing plus any visible status indicator to help troubleshooting.)

From a user perspective, common checks are:

- Reconnect device and refresh the page.
- Confirm host permissions/device access.
- Restart the application stack.

## Next
[//]: # (Screenshot suggestion: optional docs links screenshot to Environment Setup and Getting Started chapters.)

- Environment setup: `application/docs/04-environment-setup.md`.
- Full quick start: `application/docs/03-getting-started.md`.
