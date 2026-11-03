# Debugging Studio with StarArm arms

## Scope and status

Compare the current [upstream StarArm single-arm integration](https://github.com/servodevelop/Star-Arm-102/tree/24e463b7b34cd97f694a59011acea65c5213bdc3/lerobot/single-arm)
with our native `physicalai-stararm-plugin` through Studio's LeRobot bridge.
Initial installation and software-only checks were performed on November 3, 2026.
No arms were connected, calibrated, or moved by the agent. The initial checks
did not restart the backend. In a follow-up, the user restarted it with the
trust setting enabled, and all three upstream entries were verified in the
live catalog at `http://localhost:3000/api/robots/catalog`.

Installed upstream packages:

| Package                           | Version/source                                                                                              | Purpose                                                  |
| --------------------------------- | ----------------------------------------------------------------------------------------------------------- | -------------------------------------------------------- |
| `lerobot_teleoperator_stararm102` | Git commit `24e463b7b34cd97f694a59011acea65c5213bdc3`, subdirectory `lerobot/single-arm/lerobot-stararm102` | Both `stararm102_hd` leader and `stararm102_fl` follower |
| `lerobot-motor-starai`            | `0.0.6`                                                                                                     | LeRobot motor bus                                        |
| `fashionstar-uart-sdk`            | `1.3.12`                                                                                                    | Servo UART SDK                                           |
| `lerobot_robot_stararm102`        | `0.0.1`, same Git commit, subdirectory `lerobot/single-arm/lerobot-robot-stararm102`                        | Legacy `Motor_0` follower, added on request              |

The upstream package reports version `0.0.0` through its build-version fallback;
the Git commit, not that version string, identifies the tested source.

The legacy follower `lerobot_robot_stararm102` can coexist with the current
package and is now installed for comparison. Its `Motor_0`-style coordinates
target the older released ACT example; do not pair it directly with the
current named-joint leader or share their calibration files, datasets, or model
inputs. Use a separate environment for the full legacy leader/follower pair:
the legacy leader in `lerobot/single-arm/lerobot-teleoperator-stararm102` shares
the current package's installation/import name and cannot coexist with it.

## Initial installation steps

1. Inspected Studio's backend dependencies and the local bridge at
   `/home/mark/projects/intel/physicalai/packages/physicalai-lerobot-plugin`.
2. Checked the running backend command and its Python environment. Studio was
   already launched with editable Runtime, Rebot, native StarArm, and LeRobot
   bridge packages. These live in a uv overlay above
   `application/backend/.venv`; inspecting the base environment alone does not
   show all packages used by the running application.
3. Cloned upstream for source inspection and recorded its commit:

   ```bash
   git clone --depth 1 https://github.com/servodevelop/Star-Arm-102.git /tmp/opencode/Star-Arm-102
   git -C /tmp/opencode/Star-Arm-102 rev-parse HEAD
   ```

4. Read the upstream package metadata, config registration, driver code, and
   calibration implementation. The current package exports both devices from
   its package root and uses LeRobot's config registration mechanism, so it
   fits the bridge's automatic discovery path.
5. Ran a dependency-resolving installation dry run. It proposed downgrading
   setuptools, contrary to Studio's existing security override. Installed only
   the three new packages, leaving existing dependencies intact:

   ```bash
   # From the Studio repository root:
   uv pip install --python application/backend/.venv/bin/python --dry-run \
     /tmp/opencode/Star-Arm-102/lerobot/single-arm/lerobot-stararm102 \
     'lerobot-motor-starai==0.0.6' \
     'fashionstar-uart-sdk==1.3.12'

   uv pip install --python application/backend/.venv/bin/python --no-deps \
     'lerobot_teleoperator_stararm102 @ git+https://github.com/servodevelop/Star-Arm-102.git@24e463b7b34cd97f694a59011acea65c5213bdc3#subdirectory=lerobot/single-arm/lerobot-stararm102' \
     'lerobot-motor-starai==0.0.6' \
     'fashionstar-uart-sdk==1.3.12'
   ```

   This was a targeted initial installation, not a recommendation to bypass
   dependency resolution for future updates.

6. In a fresh process using the running backend's overlay, enabled
   `TRUST_UNVERIFIED_PLUGIN=1` and called the bridge's
   `register_physicalai_studio_plugin` with an in-memory registry. Both StarArm
   entries registered automatically, and their payload validation and JSON
   schema generation passed.
7. Constructed both devices through LeRobot's factories using a nonexistent
   serial path and temporary calibration directories. Neither constructor
   connected to hardware. The bridge's `connect_device` rejected missing
   calibration with `RobotCalibrationError` before opening the serial port.
8. Ran `python -m lerobot.scripts.lerobot_calibrate --help` with the backend
   interpreter. It listed both StarArm types and the calibration/button fields.
9. Ran `uv pip check --python application/backend/.venv/bin/python`. It reported
   existing setuptools constraint conflicts: the base environment had
   setuptools `84.0.0`, while LeRobot requires `<81` and torch requires `<82`.
   Studio explicitly overrides setuptools to `>=83.0.0` for a security fix;
   the installation did not change it. The launch overlay separately reported
   setuptools `80.10.2`, so do not assume the base and overlay resolve identically.

## Follow-up: separate robot package and missing live entries

Installed the legacy follower explicitly requested after the initial setup:

```bash
# From the Studio repository root:
uv pip install --python application/backend/.venv/bin/python --no-deps \
  'lerobot_robot_stararm102 @ git+https://github.com/servodevelop/Star-Arm-102.git@24e463b7b34cd97f694a59011acea65c5213bdc3#subdirectory=lerobot/single-arm/lerobot-robot-stararm102'
```

Added `lerobot-robot-stararm102` to the backend dependencies, with the same
pinned Git commit and its own subdirectory source, then ran `uv lock` from
`application/backend`. This added only that package.

The live catalog initially listed built-in LeRobot devices and native StarArm
devices, but no upstream StarArm extensions. Inspection of the running
backend's `/proc/<pid>/environ` confirmed that `TRUST_UNVERIFIED_PLUGIN` was
unset. The user restarted with it enabled. A subsequent GET of
`http://localhost:3000/api/robots/catalog` confirmed all three upstream entries
listed below, alongside the three native StarArm entries. Installing an
additional robot package alone does not bypass the trust gate.

## Keeping the installation across syncs

All four upstream packages are now declared as regular dependencies in
`application/backend/pyproject.toml`. Its `[tool.uv.sources]` pins both StarArm
packages to the tested Git commit and their respective subdirectories. The backend lockfile
records the resolved packages. The temporary inspection clone is not needed
for subsequent installations.

From `application/backend`, use the normal hardware-specific sync, for example:

```bash
uv sync --locked --extra xpu
```

Use `cpu` or `cuda` instead on other machines. These dependencies are not an
optional StarArm extra, so ordinary backend syncs retain them. A frozen run
does not refresh an old environment; sync the updated lockfile first.

The local Runtime and native/bridge plugins remain launch-time editable
overlays, not new dependencies or source overrides in this change. Preserve
those flags when comparing implementations. The local bridge requires Runtime
`>=0.4.0`, while the backend's ordinary Runtime dependency remains `~=0.3.0`.
Installing just the upstream packages does not install or enable the bridge.

Validation after declaring the dependencies:

- `uv lock` added only the three upstream packages; existing locked package
  versions did not change.
- `uv lock --check` passed.
- `uv sync --locked --extra xpu --dry-run` retained all three installed upstream
  packages. It also proposed removing unrelated non-default test/viewer
  packages; the real sync was not run.
- `prek run --files debugging-with-stararms.md application/backend/pyproject.toml application/backend/uv.lock`
  passed, including documentation formatting and the backend lockfile check.

## Starting Studio and checking discovery

Restart the backend with `TRUST_UNVERIFIED_PLUGIN=1`. This explicitly trusts
installed third-party LeRobot extensions and executes their package code; it
is not a package-specific trust setting.

For this machine, the existing launch command with that setting added is:

```bash
cd /home/mark/projects/intel/physical-ai-studio/application/backend
TRUST_UNVERIFIED_PLUGIN=1 uv run --frozen --extra xpu \
  --with-editable /home/mark/projects/intel/physicalai \
  --with-editable /home/mark/projects/intel/physicalai/packages/physicalai-rebot-b601-plugin \
  --with-editable /home/mark/projects/intel/physicalai/packages/physicalai-stararm-plugin \
  --with-editable /home/mark/projects/intel/physicalai/packages/physicalai-lerobot-plugin \
  physicalai-studio serve
```

Stop the existing backend normally before starting another instance. Expect:

- `LeRobot_stararm102_hd_Leader`
- `LeRobot_stararm102_fl_Follower`
- `LeRobot_lerobot_robot_stararm102_Follower` (legacy follower)
- The native StarArm LD/HD/FL entries from `physicalai-stararm-plugin`.

Without the trust setting, installed third-party LeRobot extensions are not
imported by the bridge. Restart after changing it: discovery is cached in the
running process. If entries are missing, check the backend interpreter,
editable flags, trust setting, and plugin import errors before checking USB.

## Calibration required by the current upstream integration

Create **one LeRobot calibration JSON per physical arm**: one for the leader
and one for the follower. Each contains calibration entries for all seven
joints: `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_yaw`,
`wrist_roll`, and `gripper` (servo IDs 0–6 by default).

Studio's adapter connects non-interactively with `calibrate=False`; it does not
run terminal prompts or silently repair calibration. Missing or invalid files
must be resolved before connecting through Studio.

**Calibration writes servo origins and unlocks joints.** Secure both bases,
support the arms, and confirm the physical revision's correct upstream zero
pose before starting. Close Studio robot sessions and other serial tools.
Use the real leader/follower port paths, not the example numbering below.
Do not run calibration just to test package discovery.

From `application/backend`:

```bash
CAL_DIR="$HOME/.local/share/physicalai/stararm102/lerobot"

.venv/bin/lerobot-calibrate \
  --teleop.type=stararm102_hd \
  --teleop.port=/dev/ttyUSB0 \
  --teleop.id=stararm102_leader \
  --teleop.calibration_dir="$CAL_DIR" \
  --teleop.button.enabled=false

.venv/bin/lerobot-calibrate \
  --robot.type=stararm102_fl \
  --robot.port=/dev/ttyUSB1 \
  --robot.id=stararm102_follower \
  --robot.calibration_dir="$CAL_DIR"
```

Follow the prompts to set the confirmed zero pose with the gripper closed,
then record each joint's safe motion range without forcing mechanical stops.
Successful calibration saves:

```text
$HOME/.local/share/physicalai/stararm102/lerobot/stararm102_leader.json
$HOME/.local/share/physicalai/stararm102/lerobot/stararm102_follower.json
```

In Studio's advanced configuration, set:

| Device            | `id`                  | `calibration_dir`                                 |
| ----------------- | --------------------- | ------------------------------------------------- |
| Upstream leader   | `stararm102_leader`   | The expanded absolute path to the directory above |
| Upstream follower | `stararm102_follower` | The expanded absolute path to the directory above |

Use the directory, not the JSON filename, for `calibration_dir`. Do not enter
literal `$HOME` or assume Studio expands `~`. IDs select filenames; they are
not servo bus IDs. Use separate IDs/files for additional physical arms.

If `calibration_dir` is unset, LeRobot uses its environment-dependent defaults
under `HF_LEROBOT_CALIBRATION`, with `robots/stararm102_fl` for the follower and
`teleoperators/stararm102_hd` for the leader. Explicit directories avoid
confusion between terminal and application environments.

Our native plugin does **not** consume these LeRobot JSON files. Its leader
calibration step in Studio calls `set_zero_position()`, storing origins in the
servos. Switching implementations or recalibrating changes shared hardware
state: saved JSON files alone do not restore a previous physical zero.

### Current follower placeholder for debugging

For a software-only workaround to the current follower's missing-file check,
see [the explicitly unmeasured placeholder](stararm-calibration-placeholder/README.md).
It loads successfully through the installed current follower's constructor
without opening a serial port. It assumes existing servo zero origins and
uses config-derived raw ranges, not measured travel. Its acceptance by
`is_calibrated` does not validate the hardware. It is not for the legacy
follower described below.

### Legacy follower calibration

The additional `lerobot_robot_stararm102` follower needs its **own** calibration
using `Motor_0` through `Motor_5` and `gripper`. The named-joint JSON above is
not compatible. To calibrate it, use `--robot.type=lerobot_robot_stararm102`,
a distinct ID such as `stararm102_legacy_follower`, and a separate directory
such as `$HOME/.local/share/physicalai/stararm102/legacy-lerobot` with the
terminal calibration command. Apply the same physical setup precautions;
this hardware workflow has not been tested here. If no directory is set, its
device name resolves to `robots/stararm102_follower`, not the config type name.

Legacy bus mode defaults to counts and normalized joints, unlike the current
degree-based implementation. Merely renaming joint keys is not sufficient to
make their actions or calibration interchangeable. The legacy follower's
`max_relative_target` config exists, but its implementation comments out the
corresponding safety clamp; do not rely on that setting to limit motion.

## Comparison checklist and remaining risks

- Start with matched upstream leader/follower and native leader/follower
  pairs. Do not mix implementations until coordinate conventions agree.
- Upstream's `stararm102_hd` type also covers an LD leader. Keep
  `button.enabled=false` for LD. Enable it on HD only after checking the ID-7
  board and lock behavior. The button is not a hardware emergency stop.
- Both implementations use the same seven named channels, but direction,
  scaling, and limits differ. Upstream's gripper direction multiplier is `-6`
  with range `[-270, 0]`; the native plugin's default gripper range is `[0, 270]`.
  Other joint limits also differ. Calibration does not automatically reconcile
  these defaults. Do not exchange datasets/checkpoints without validating
  their coordinate conventions.
- After checking zero poses, test one joint at a time through a small safe
  range before coordinated teleoperation. Keep servo power accessible for
  stopping and never open one port in both implementations simultaneously.
- Upstream connection unlocks joints and resets multi-turn counts. Reconnect
  only in the expected reference pose with the arms supported.
- Upstream caches observations/actions on some read failures. Review this
  behavior when comparing disconnect handling; a cached value is not fresh
  feedback and a logged warning is not an emergency stop.
- Upstream's documented workflow targets LeRobot `0.4.1`; Studio has `0.6.0`
  and Python `3.13`. Discovery, schemas, constructors, missing-calibration
  guards, and calibration CLI help passed here. Actual calibration, motion,
  recording, and policy execution still require bench validation. Upstream
  also documents that end-to-end hardware validation remains outstanding.
- The generic bridge currently supplies no StarArm-specific URDF asset for
  these generated catalog entries; the native plugin bundles its own URDF.

References:

- [Pinned upstream single-arm guide](https://github.com/servodevelop/Star-Arm-102/blob/24e463b7b34cd97f694a59011acea65c5213bdc3/lerobot/single-arm/README.md)
- [LeRobot bridge documentation](https://github.com/openvinotoolkit/physicalai/tree/main/packages/physicalai-lerobot-plugin)
- Local native plugin: `/home/mark/projects/intel/physicalai/packages/physicalai-stararm-plugin/README.md`
