import re
from collections.abc import Mapping
from typing import Any, TypedDict

from physicalai.inference.manifest import ComponentSpec, Manifest

from schemas.environment import EnvironmentWithRelations


class ObservationSampleContext(TypedDict):
    name: str
    value: str


def control_loop_command(environment: EnvironmentWithRelations | None, task: str | None) -> str:
    """Build the README example command for running robot inference."""
    robot = _control_loop_robot(environment)
    task_arg = _shell_quote(task or "pick up the object")
    lines = [
        "python examples/runtime/sync_inference.py \\",
        f"  --robot {robot} \\",
        "  --port /dev/ttyACM0 \\",
        "  --calibration ./calibration.json \\",
        "  --model path/to/model \\",
    ]
    camera_args = _control_loop_cameras(environment)
    lines.extend([f"  --camera {camera} \\" for camera in camera_args])
    lines.extend(
        [
            f"  --task {task_arg} \\",
            "  --device CPU",
        ]
    )
    return "\n".join(lines)


def observation_context(manifests: Mapping[str, Manifest]) -> list[ObservationSampleContext]:
    """Build README example observations from the preferred inference manifest."""
    manifest = _select_inference_manifest(manifests)
    if manifest is None:
        return []

    return [sample for feature in manifest.model.input_features if (sample := _sample(feature))]


def _control_loop_robot(environment: EnvironmentWithRelations | None) -> str:
    if environment is None or not environment.robots:
        return "so101"

    robot_type = str(environment.robots[0].robot.type).lower()
    if "so101" in robot_type:
        return "so101"
    if "widowxai" in robot_type:
        return "trossen_widowxai"

    return _slug(robot_type)


def _control_loop_cameras(environment: EnvironmentWithRelations | None) -> list[str]:
    if environment is None or not environment.cameras:
        return ["overhead:uvc:/dev/video0"]

    return [
        f"{_slug(camera.name)}:"
        f"{_runtime_camera_driver(camera.driver)}:"
        f"{_runtime_camera_source(camera.driver, index)}"
        for index, camera in enumerate(environment.cameras)
    ]


def _runtime_camera_driver(driver: str) -> str:
    return {"usb_camera": "uvc", "ipcam": "ipcam"}.get(driver, driver)


def _runtime_camera_source(driver: str, index: int) -> str:
    if driver == "usb_camera":
        return f"/dev/video{index}"
    if driver == "ipcam":
        return "rtsp://<camera-host>/stream"
    return f"<camera-{index}>"


def _shell_quote(value: str) -> str:
    return '"{}"'.format(value.replace('"', '\\"'))


def _select_inference_manifest(manifests: Mapping[str, Manifest]) -> Manifest | None:
    for backend in ("openvino", "torch", "onnx", "executorch"):
        manifest = manifests.get(backend)
        if manifest is not None and manifest.model.input_features:
            return manifest

    return next((manifest for manifest in manifests.values() if manifest.model.input_features), None)


def _sample(feature: ComponentSpec) -> ObservationSampleContext | None:
    name = feature.init_args.get("name")
    if not isinstance(name, str) or not name:
        return None

    dtype = feature.init_args.get("dtype")
    shape = feature.init_args.get("shape")
    if dtype == "string":
        value = '["sample task description"]'
    elif dtype == "int64":
        value = f"np.zeros({_format_numpy_shape(shape)}, dtype=np.int64)"
    elif shape == []:
        value = "np.array(0.0, dtype=np.float32)"
    else:
        value = f"np.random.rand{_format_rand_shape(shape)}.astype(np.float32)"

    return {"name": name, "value": value}


def _format_numpy_shape(shape: Any) -> str:
    if not isinstance(shape, list):
        return "(1,)"
    if len(shape) == 0:
        return "()"
    return repr((1, *shape))


def _format_rand_shape(shape: Any) -> str:
    if not isinstance(shape, list):
        return "(1)"
    if len(shape) == 0:
        return "()"
    return "(" + ", ".join(str(dim) for dim in (1, *shape)) + ")"


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "camera"
