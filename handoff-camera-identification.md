# Linux UVC Camera Identification Handoff

## Problem

Some identical UVC cameras expose the same USB serial number. For example,
three Innomaker U20CAM-1080p-S1 cameras all report `SN0001`. Their
`/dev/v4l/by-id/...` paths are therefore ambiguous and must not be used to
select an individual camera.

## Application Contract

Use the camera identifier returned by `physicalai.capture.discover_all()`.

- When `DeviceInfo.id_stable` is `True`, persist `DeviceInfo.device_id`.
- A unique USB serial produces a `/dev/v4l/by-id/...` selector.
- A duplicated USB serial produces a `/dev/v4l/by-path/...` selector.
- Treat `DeviceInfo.physical_port` as an informational port identity, not a
  device serial number.

For duplicate-serial cameras, the identity follows the USB port. Reconnecting
the same camera to the same port restores its selector; swapping cameras or
moving a cable changes which physical camera that selector addresses.

## Example

```python
from physicalai.capture import create_camera, discover_all

for camera in discover_all()["uvc"]:
    print(camera.name, camera.device_id, camera.physical_port)

camera = create_camera(
    "uvc",
    device="/dev/v4l/by-path/pci-0000:00:14.0-usb-0:2.1.2:1.0-video-index0",
    width=640,
    height=480,
    fps=30,
)
```

Persist the selector unchanged in application configuration:

```yaml
cameras:
  left:
    class_path: physicalai.capture.UVCCamera
    init_args:
      device: /dev/v4l/by-path/pci-0000:00:14.0-usb-0:2.1.2:1.0-video-index0
      width: 640
      height: 480
      fps: 30
```

Do not resolve the selector to `/dev/videoN` before saving it. Video indices
are assigned dynamically and can change after re-enumeration.

## Validation

Run the setup validator before deployment:

```bash
python examples/runtime/validate_setup.py \
  --camera left:uvc:/dev/v4l/by-path/pci-0000:00:14.0-usb-0:2.1.2:1.0-video-index0 \
  --camera right:uvc:/dev/v4l/by-path/pci-0000:00:14.0-usb-0:2.1.3:1.0-video-index0
```

It validates each device individually, then all devices concurrently. A
concurrent-only failure can indicate USB hub bandwidth or power contention.

## Operational Rules

- Label the physical USB ports and keep camera cables in their assigned ports.
- Do not use `/dev/videoN` for persistent configuration.
- Do not use `/dev/v4l/by-id/...` for cameras reporting duplicate serials.
- Re-run discovery after changing hubs, host controllers, or physical wiring.
- For `SharedCamera`, pass the same by-path selector in the camera recipe; the
  transport derives a stable service identity from the port selector.
