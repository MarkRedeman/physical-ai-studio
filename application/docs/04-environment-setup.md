# Environment Setup

In Physical AI Studio, an environment combines:

- Robots (follower + teleoperator)
- Cameras

Datasets and model runs depend on this setup, so create it carefully in the UI.

## 1) Add robots

| **SO101 Leader robot**                                     | **Trossen WidowX AI follower robot**                                 | **Robots list**                           |
|------------------------------------------------------------|----------------------------------------------------------------------|-------------------------------------------|
| ![Setup a new SO101 Leader robot][robots-new-so101-leader] | ![Setup a new WidowXAI follower robot][robots-new-widowxai-follower] | ![List of configured robots][robots-list] |

[robots-new-so101-leader]: ./assets/03-robots-new-so101-leader.png
[robots-new-widowxai-follower]: ./assets/03-robots-new-widowxai-follower.png
[robots-list]: ./assets/03-robots-list.png

[//]: # (Screenshot suggestion: Robots section landing view with list of configured robots and Add new robot action.)

Open the Robots section and click to add a robot.

What you see in the UI depends on robot type.

### SO101 setup

| **SO101 Motor Setup**                                                 | **SO101 Calibration**                            | **SO101 Verification**                                   |
|-----------------------------------------------------------------------|--------------------------------------------------|----------------------------------------------------------|
| ![Setup a motors of an off the shelf so101 robot][so101-setup-motors] | ![Calibrate your SO101 robot][so101-calibration] | ![Verify calibration was successful][so101-verification] |

[so101-setup-motors]: ./assets/04-so101-motor-setup.png
[so101-calibration]: ./assets/04-so101-calibration.png
[so101-verification]: ./assets/04-so101-verification.png

For SO101, the UI can guide you through a setup wizard.
If you've not yet configured your SO101's control board then we will help you setup the motors, and subsequently calibrate the arm. 
Once done you can verify that calibration is successful by moving the arm and check that its movement matches the screen.

You may also choose to "Skip verification" if you have previously calibrated the arm. In this case we read the calibration values from the control board.

### Trossen WidowX AI setup

| **Trossen WidowX AI follower robot**                                 |
|----------------------------------------------------------------------|
| ![Setup a new WidowXAI follower robot][robots-new-widowxai-follower] |

To configure your Trossen's WidowX AI arms provide their IP Address, verify 

### Bimanual Trossen WidowX AI setup

| **Bimanual Trossen WidowX AI setup**                                 |
|----------------------------------------------------------------------|
| ![Setup a new bimanual WidowXAI follower robot][robots-bimanual-widowx-ai] |

[robots-bimanual-widowx-ai]: ./assets/04-robots-new-bimanual-widowx-ai.png

### Extending Physical AI Studio with new robots

> TODO: TBD

## 2) Add cameras

| **New USB Camera**                                   | **Cameras list**                            |
|------------------------------------------------------|---------------------------------------------|
| ![Setup a new overview camera][cameras-new-overview] | ![List of configured cameras][cameras-list] |

[cameras-new-overview]: ./assets/03-cameras-new-overview.png
[cameras-list]: ./assets/03-cameras-list.png

Open Cameras and add each camera you need.

The UI supports multiple camera drivers and only shows what is available.

During setup, confirm:

- You selected the correct device.
- Resolution and FPS look right.
- Preview feed is correct before saving.

In our own setups we often use a resolution of 640 x 480 at 30 FPS.
Internally the models trained by Physical AI Studio resize images to a low resolution, hence choosing a high resolution for your cameras may not be effective and end up consuming more disk space.

## 3) Create environment

| **New environment**                          | **Environment list**                       |
|----------------------------------------------|--------------------------------------------|
| ![Setup a new environment][environments-new] | ![List of environments][environments-list] |

[environments-new]: ./assets/03-environments-new.png
[environments-list]: ./assets/03-environments-list.png

Open Environments and create a new one.

In the form, you will:

- Select the follower robot.
- Select the teleoperator (leader) robot.
- Add one or more cameras.
- Save the environment.

After saving, open the environment and verify all robots and cameras appear as expected.

## Next
[//]: # (Screenshot suggestion: optional docs navigation screenshot to Recording Datasets and Hardware Integration chapters.)

- Continue with recording in `application/docs/05-recording-datasets.md`.
- For hardware-specific troubleshooting, see `application/docs/10-integrating-new-hardware.md`.
