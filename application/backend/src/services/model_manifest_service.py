import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any
from uuid import UUID

import yaml
from huggingface_hub import ModelCard, ModelCardData
from physicalai.inference.manifest import ComponentSpec, Manifest

from schemas.calibration import Calibration
from schemas.dataset import Dataset
from schemas.environment import EnvironmentWithRelations, RobotWithTeleoperator
from schemas.model import Model
from schemas.project_camera import Camera
from schemas.robot import Robot

_MANIFEST_FILENAME = "manifest.json"
_README_FILENAME = "README.md"
_ENVIRONMENT_FILENAME = "environment.json"
_CALIBRATION_FILENAME = "calibration.json"
_TORCH_BACKEND = "torch"
_TORCH_MANIFEST_PATH = Path("exports") / _TORCH_BACKEND / _MANIFEST_FILENAME
_POLICY_PACKAGE_FORMAT = "policy_package"
_ENVIRONMENT_FORMAT = "physical_ai_studio_environment"
_MODEL_CARD_TEMPLATES_DIR = Path(__file__).parents[1] / "models"
_DEFAULT_MODEL_CARD_TEMPLATE = "model_card_act.md"
_POLICY_TITLES = {
    "act": "Action Chunking Transformer (ACT)",
    "smolvla": "SmolVLA",
    "pi05": "Pi0.5",
}
_BACKEND_USE = {
    "torch": "Canonical checkpoint and Python inference",
    "openvino": "CPU, Intel GPU, and NPU inference",
    "onnx": "Runtime portability",
    "executorch": "Edge and mobile runtime experiments",
}


class ModelManifestService:
    """Generate package-level model metadata files."""

    @staticmethod
    def write_root_manifest(model_dir: Path) -> Path | None:
        """Create a root manifest from the torch export manifest.

        The torch export is the canonical checkpoint representation. Its manifest
        can also serve as the package manifest when artifact paths are rewritten
        from torch-export-relative paths to model-root-relative paths.
        """
        torch_manifest_path = model_dir / _TORCH_MANIFEST_PATH
        if not torch_manifest_path.is_file():
            return None

        manifest = ModelManifestService._load_manifest(torch_manifest_path)
        if manifest is None:
            return None

        root_manifest = ModelManifestService._with_root_relative_artifacts(manifest)
        root_manifest_path = model_dir / _MANIFEST_FILENAME
        root_manifest.save(root_manifest_path)

        return root_manifest_path

    @staticmethod
    def write_environment_description(
        model_dir: Path,
        environment: EnvironmentWithRelations,
        calibrations: dict[UUID, Calibration] | None = None,
    ) -> Path:
        """Create a sanitized training environment description."""
        environment_path = model_dir / _ENVIRONMENT_FILENAME
        environment_path.write_text(
            json.dumps(
                ModelManifestService._environment_context(environment, calibrations or {}),
                indent=2,
            ),
            encoding="utf-8",
        )
        return environment_path

    @staticmethod
    def write_runtime_calibration(
        model_dir: Path,
        environment: EnvironmentWithRelations,
        calibrations: dict[UUID, Calibration] | None = None,
    ) -> Path | None:
        """Create a LeRobot-compatible calibration file for the primary trained robot."""
        calibration = ModelManifestService._select_runtime_calibration(environment, calibrations or {})
        if calibration is None:
            return None

        calibration_path = model_dir / _CALIBRATION_FILENAME
        calibration_path.write_text(
            json.dumps(ModelManifestService._runtime_calibration_context(calibration), indent=2),
            encoding="utf-8",
        )
        return calibration_path

    @staticmethod
    def write_model_card(
        model_dir: Path,
        model: Model | None = None,
        dataset: Dataset | None = None,
        environment: EnvironmentWithRelations | None = None,
        calibrations: dict[UUID, Calibration] | None = None,
    ) -> Path | None:
        """Create a Hugging Face style README from the root model manifest."""
        manifest = ModelManifestService._load_manifest(model_dir / _MANIFEST_FILENAME)
        if manifest is None:
            return None

        policy_name = manifest.policy.name
        export_manifests = ModelManifestService._load_export_manifests(model_dir)
        backends = sorted(export_manifests)
        display_name = model.name if model and model.name else policy_name or "PhysicalAI Model"
        environment_context = (
            ModelManifestService._environment_context(environment, calibrations or {}) if environment else None
        )
        dataset_task = dataset.default_task if dataset else None
        card_data = ModelCardData(
            license="apache-2.0",
            library_name="physicalai",
            pipeline_tag="robotics",
            tags=sorted(
                {
                    "vision-language-action",
                    "robotics",
                    "physicalai",
                    "physical-ai-studio",
                    policy_name,
                    *backends,
                }
                - {""}
            ),
            model_name=display_name,
        )
        card = ModelCard.from_template(
            card_data,
            template_str=ModelManifestService._read_template(policy_name),
            model_name=display_name,
            model_title=ModelManifestService._policy_title(policy_name, display_name),
            policy_name=policy_name or "unknown",
            dataset_name=dataset.name if dataset else None,
            exports=ModelManifestService._export_context(export_manifests),
            io_groups=ModelManifestService._io_group_context(export_manifests),
            observation_samples=ModelManifestService._observation_context(export_manifests),
            environment=environment_context,
            environment_yaml=ModelManifestService._environment_yaml(environment_context),
            control_loop_command=ModelManifestService._control_loop_command(environment_context, dataset_task),
        )
        card.validate()

        readme_path = model_dir / _README_FILENAME
        card.save(readme_path)
        readme_path.write_text(
            ModelManifestService._compact_markdown(readme_path.read_text(encoding="utf-8")),
            encoding="utf-8",
        )
        return readme_path

    @staticmethod
    def _load_manifest(manifest_path: Path) -> Manifest | None:
        try:
            manifest = Manifest.load(manifest_path)
        except (FileNotFoundError, ValueError):
            return None

        return manifest if manifest.format == _POLICY_PACKAGE_FORMAT else None

    @staticmethod
    def _load_export_manifests(model_dir: Path) -> dict[str, Manifest]:
        root_manifest = ModelManifestService._load_manifest(model_dir)
        manifests = dict.fromkeys(root_manifest.model.artifacts, root_manifest) if root_manifest else {}

        exports_dir = model_dir / "exports"
        if not exports_dir.is_dir():
            return manifests

        for manifest_path in sorted(exports_dir.glob(f"*/{_MANIFEST_FILENAME}")):
            manifest = ModelManifestService._load_manifest(manifest_path)
            if manifest is not None:
                manifests[manifest_path.parent.name] = ModelManifestService._with_export_relative_artifacts(
                    manifest_path,
                    manifest,
                    model_dir,
                )

        return manifests

    @staticmethod
    def _with_root_relative_artifacts(manifest: Manifest) -> Manifest:
        artifacts = {
            backend: str(Path("exports") / _TORCH_BACKEND / artifact)
            for backend, artifact in manifest.model.artifacts.items()
        }
        return manifest.model_copy(update={"model": manifest.model.model_copy(update={"artifacts": artifacts})})

    @staticmethod
    def _with_export_relative_artifacts(manifest_path: Path, manifest: Manifest, model_dir: Path) -> Manifest:
        export_dir = manifest_path.parent
        artifacts = {
            backend: str((export_dir / artifact).relative_to(model_dir))
            for backend, artifact in manifest.model.artifacts.items()
        }
        return manifest.model_copy(update={"model": manifest.model.model_copy(update={"artifacts": artifacts})})

    @staticmethod
    def _read_template(policy_name: str) -> str:
        template_path = _MODEL_CARD_TEMPLATES_DIR / f"model_card_{policy_name}.md"
        if not template_path.is_file():
            template_path = _MODEL_CARD_TEMPLATES_DIR / _DEFAULT_MODEL_CARD_TEMPLATE
        return template_path.read_text(encoding="utf-8")

    @staticmethod
    def _compact_markdown(markdown: str) -> str:
        markdown = re.sub(r"\n[ \t]*\n(?=})", "\n", markdown)
        return re.sub(r"\n{3,}", "\n\n", markdown).rstrip() + "\n"

    @staticmethod
    def _policy_title(policy_name: str, fallback: str) -> str:
        return _POLICY_TITLES.get(policy_name, fallback)

    @staticmethod
    def _export_context(manifests: dict[str, Manifest]) -> list[dict[str, str]]:
        return [
            {
                "backend": backend,
                "artifact": artifact,
                "use": _BACKEND_USE.get(backend, "Backend-specific inference"),
            }
            for manifest in manifests.values()
            for backend, artifact in sorted(manifest.model.artifacts.items())
        ]

    @staticmethod
    def _io_group_context(manifests: dict[str, Manifest]) -> list[dict[str, Any]]:
        grouped_specs: dict[str, dict[str, Any]] = {}
        grouped_backends: dict[str, list[str]] = defaultdict(list)
        for backend, manifest in sorted(manifests.items()):
            inputs = ModelManifestService._feature_context(manifest.model.input_features)
            outputs = ModelManifestService._feature_context(manifest.model.output_features)
            if not inputs and not outputs:
                continue

            signature = json.dumps({"inputs": inputs, "outputs": outputs}, sort_keys=True)
            grouped_specs.setdefault(signature, {"inputs": inputs, "outputs": outputs})
            grouped_backends[signature].append(backend)

        return [
            {
                "backends": backends,
                "backend_label": ", ".join(f"`{backend}`" for backend in backends),
                "inputs": grouped_specs[signature]["inputs"],
                "outputs": grouped_specs[signature]["outputs"],
            }
            for signature, backends in grouped_backends.items()
        ]

    @staticmethod
    def _feature_context(features: list[ComponentSpec]) -> list[dict[str, str]]:
        return [
            {
                "name": str(feature.init_args["name"]),
                "ftype": str(feature.init_args.get("ftype", "-")),
                "shape": ModelManifestService._format_shape(feature.init_args.get("shape")),
                "dtype": str(feature.init_args.get("dtype", "-")),
            }
            for feature in features
            if isinstance(feature.init_args.get("name"), str) and feature.init_args["name"]
        ]

    @staticmethod
    def _environment_context(
        environment: EnvironmentWithRelations,
        calibrations: dict[UUID, Calibration],
    ) -> dict[str, Any]:
        return {
            "format": _ENVIRONMENT_FORMAT,
            "version": "1.0",
            "name": environment.name,
            "robots": [
                ModelManifestService._environment_robot_context(robot_config, calibrations)
                for robot_config in environment.robots
            ],
            "cameras": [ModelManifestService._environment_camera_context(camera) for camera in environment.cameras],
        }

    @staticmethod
    def _environment_robot_context(
        robot_config: RobotWithTeleoperator,
        calibrations: dict[UUID, Calibration],
    ) -> dict[str, Any]:
        robot = robot_config.robot
        context: dict[str, Any] = {
            "name": robot.name,
            "type": str(robot.type),
            "calibration": ModelManifestService._calibration_context(calibrations.get(robot.id)),
        }

        if robot_config.tele_operator.type == "robot" and robot_config.tele_operator.robot is not None:
            context["teleoperator"] = ModelManifestService._teleoperator_context(robot_config.tele_operator.robot)
        else:
            context["teleoperator"] = {"type": "none"}

        return context

    @staticmethod
    def _environment_yaml(environment: dict[str, Any] | None) -> str | None:
        if environment is None:
            return None

        yaml_environment = {
            "name": environment["name"],
            "robots": [
                {key: value for key, value in robot.items() if key != "teleoperator" and value is not None}
                for robot in environment["robots"]
            ],
            "cameras": environment["cameras"],
        }
        return yaml.safe_dump(yaml_environment, sort_keys=False).strip()

    @staticmethod
    def _control_loop_command(environment: dict[str, Any] | None, task: str | None) -> str:
        robot = ModelManifestService._control_loop_robot(environment)
        task_arg = ModelManifestService._shell_quote(task or "pick up the object")
        lines = [
            "python examples/runtime/sync_inference.py \\",
            f"  --robot {robot} \\",
            "  --port /dev/ttyACM0 \\",
            "  --calibration ./calibration.json \\",
            "  --model path/to/model \\",
        ]
        camera_args = ModelManifestService._control_loop_cameras(environment)
        lines.extend([f"  --camera {camera} \\" for camera in camera_args])
        lines.extend(
            [
                f"  --task {task_arg} \\",
                "  --device CPU",
            ]
        )
        return "\n".join(lines)

    @staticmethod
    def _control_loop_robot(environment: dict[str, Any] | None) -> str:
        if environment is None or not environment["robots"]:
            return "so101"

        robot_type = str(environment["robots"][0]["type"]).lower()
        if "so101" in robot_type:
            return "so101"
        if "widowxai" in robot_type:
            return "trossen_widowxai"

        return ModelManifestService._slug(robot_type)

    @staticmethod
    def _control_loop_cameras(environment: dict[str, Any] | None) -> list[str]:
        if environment is None or not environment["cameras"]:
            return ["overhead:uvc:/dev/video0"]

        return [
            "{}:{}:{}".format(
                ModelManifestService._slug(str(camera["name"])),
                ModelManifestService._runtime_camera_driver(str(camera["driver"])),
                ModelManifestService._runtime_camera_source(str(camera["driver"]), index),
            )
            for index, camera in enumerate(environment["cameras"])
        ]

    @staticmethod
    def _runtime_camera_driver(driver: str) -> str:
        return {"usb_camera": "uvc", "ipcam": "ipcam"}.get(driver, driver)

    @staticmethod
    def _runtime_camera_source(driver: str, index: int) -> str:
        if driver == "usb_camera":
            return f"/dev/video{index}"
        if driver == "ipcam":
            return "rtsp://<camera-host>/stream"
        return f"<camera-{index}>"

    @staticmethod
    def _slug(value: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
        return slug or "camera"

    @staticmethod
    def _shell_quote(value: str) -> str:
        return '"{}"'.format(value.replace('"', '\\"'))

    @staticmethod
    def _teleoperator_context(robot: Robot) -> dict[str, str]:
        return {
            "type": "robot",
            "name": robot.name,
            "robot_type": str(robot.type),
        }

    @staticmethod
    def _calibration_context(calibration: Calibration | None) -> dict[str, dict[str, int]] | None:
        if calibration is None:
            return None

        return {
            joint_name: {
                "id": value.id,
                "drive_mode": value.drive_mode,
                "homing_offset": value.homing_offset,
                "range_min": value.range_min,
                "range_max": value.range_max,
            }
            for joint_name, value in sorted(calibration.values.items())
        }

    @staticmethod
    def _select_runtime_calibration(
        environment: EnvironmentWithRelations,
        calibrations: dict[UUID, Calibration],
    ) -> Calibration | None:
        for robot_config in environment.robots:
            calibration = calibrations.get(robot_config.robot.id)
            if calibration is not None:
                return calibration
        return None

    @staticmethod
    def _runtime_calibration_context(calibration: Calibration) -> dict[str, dict[str, int]]:
        return {
            joint_name: {
                "id": value.id,
                "drive_mode": value.drive_mode,
                "homing_offset": value.homing_offset,
                "range_min": value.range_min,
                "range_max": value.range_max,
            }
            for joint_name, value in sorted(calibration.values.items())
        }

    @staticmethod
    def _environment_camera_context(camera: Camera) -> dict[str, Any]:
        payload = camera.payload
        context: dict[str, Any] = {
            "name": camera.name,
            "driver": camera.driver,
            "hardware_name": camera.hardware_name,
            "width": getattr(payload, "width", None),
            "height": getattr(payload, "height", None),
            "fps": getattr(payload, "fps", None),
        }

        if hasattr(payload, "output_type"):
            context["output_type"] = payload.output_type
        if hasattr(payload, "depth_range_min"):
            context["depth_range_min"] = payload.depth_range_min
        if hasattr(payload, "depth_range_max"):
            context["depth_range_max"] = payload.depth_range_max

        return context

    @staticmethod
    def _observation_context(manifests: dict[str, Manifest]) -> list[dict[str, str]]:
        manifest = ModelManifestService._select_inference_manifest(manifests)
        if manifest is None:
            return []

        return [
            sample
            for feature in manifest.model.input_features
            if (sample := ModelManifestService._sample(feature))
        ]

    @staticmethod
    def _select_inference_manifest(manifests: dict[str, Manifest]) -> Manifest | None:
        for backend in ("openvino", "torch", "onnx", "executorch"):
            manifest = manifests.get(backend)
            if manifest is not None and manifest.model.input_features:
                return manifest

        return next((manifest for manifest in manifests.values() if manifest.model.input_features), None)

    @staticmethod
    def _sample(feature: ComponentSpec) -> dict[str, str] | None:
        name = feature.init_args.get("name")
        if not isinstance(name, str) or not name:
            return None

        dtype = feature.init_args.get("dtype")
        shape = feature.init_args.get("shape")
        if dtype == "string":
            value = '["sample task description"]'
        elif dtype == "int64":
            value = f"np.zeros({ModelManifestService._format_numpy_shape(shape)}, dtype=np.int64)"
        elif shape == []:
            value = "np.array(0.0, dtype=np.float32)"
        else:
            value = f"np.random.rand{ModelManifestService._format_rand_shape(shape)}.astype(np.float32)"

        return {"name": name, "value": value}

    @staticmethod
    def _format_numpy_shape(shape: Any) -> str:
        if not isinstance(shape, list):
            return "(1,)"
        if len(shape) == 0:
            return "()"
        return repr((1, *shape))

    @staticmethod
    def _format_rand_shape(shape: Any) -> str:
        if not isinstance(shape, list):
            return "(1)"
        if len(shape) == 0:
            return "()"
        return "(" + ", ".join(str(dim) for dim in (1, *shape)) + ")"

    @staticmethod
    def _format_shape(shape: Any) -> str:
        if not isinstance(shape, list):
            return "-"
        if len(shape) == 0:
            return "scalar"
        return "[{}]".format(", ".join(str(item) for item in shape))
