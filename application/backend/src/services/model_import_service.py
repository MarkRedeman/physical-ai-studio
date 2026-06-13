import asyncio
import csv
import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID, uuid4

import lightning
import torch
import yaml

from exceptions import InvalidArchiveError
from schemas import Model
from schemas.dataset import Dataset
from services.dataset_service import DatasetService
from services.model_service import ModelService
from settings import get_settings

# We assume the directory/zip is taken directly from Physical AI Studio, either
# by exporting the model from the UI, or by taking it from our storage dir
_REQUIRED_FILES = (
    "version_0/hparams.yaml",
    "version_0/metrics.csv",
    "exports/torch/manifest.json",
)
_TORCH_MANIFEST_PATH = "exports/torch/manifest.json"
_SUPPORTED_POLICIES = frozenset({"act", "smolvla", "pi05"})
_LEROBOT_CONFIG_PATH = "config.json"
_LEROBOT_TRAIN_CONFIG_PATH = "train_config.json"


class ModelReader(Protocol):
    """Abstract reader for model files (ZIP archive or directory)."""

    def file_exists(self, path: str) -> bool:
        """Check if a file exists at the given relative path."""
        ...

    def read_json(self, path: str) -> dict[str, Any] | None:
        """Read and parse a JSON file. Returns None if not found or invalid."""
        ...


class DirectoryModelReader:
    """ModelReader implementation backed by a filesystem directory."""

    def __init__(self, root: Path) -> None:
        self._root = root

    def file_exists(self, path: str) -> bool:
        return (self._root / path).is_file()

    def read_json(self, path: str) -> dict[str, Any] | None:
        file_path = self._root / path
        if not file_path.is_file():
            return None
        try:
            with file_path.open(encoding="utf-8") as fobj:
                data = json.load(fobj)
        except (OSError, ValueError):
            return None
        if isinstance(data, dict):
            return data
        return None


class ModelImportService:
    async def import_model_directory(
        self,
        *,
        source_dir: Path,
        project_id: UUID,
        dataset_id: UUID,
        model_name: str,
        move: bool = False,
        base_model_id: UUID | None = None,
        version: int = 1,
    ) -> Model:
        """Import a model from a directory (copy or move)."""
        if not source_dir.exists() or not source_dir.is_dir():
            raise InvalidArchiveError(f"Model directory does not exist: {source_dir}")

        settings = get_settings()
        dataset = await DatasetService.get_dataset_by_id(dataset_id)
        if dataset.project_id != project_id:
            raise InvalidArchiveError("Dataset does not belong to the specified project")

        model_dir = settings.models_dir / str(uuid4())

        reader = DirectoryModelReader(source_dir)
        policy = self._inspect_model(reader)

        try:
            if move:
                await asyncio.to_thread(shutil.move, str(source_dir), str(model_dir))
            else:
                await asyncio.to_thread(shutil.copytree, source_dir, model_dir)

            return await self._finalize_import(
                model_dir=model_dir,
                dataset=dataset,
                model_name=model_name,
                policy=policy,
                base_model_id=base_model_id,
                version=version,
            )
        except Exception:
            shutil.rmtree(model_dir, ignore_errors=True)
            raise

    async def _finalize_import(
        self,
        *,
        model_dir: Path,
        dataset: Dataset,
        model_name: str,
        policy: str,
        base_model_id: UUID | None,
        version: int,
    ) -> Model:
        """Create job, and model record after files are in place."""
        project_id = dataset.project_id
        dataset_id = dataset.id

        model = Model(
            id=UUID(model_dir.name),
            project_id=project_id,
            dataset_id=dataset_id,
            path=str(model_dir),
            name=model_name,
            # Imported models don't have a snapshot: the provided dataset may differ
            # from what was actually used for training (possibly on another machine).
            snapshot_id=None,
            policy=policy,
            properties={},
            train_job_id=None,
            parent_model_id=base_model_id,
            version=version,
            created_at=None,
        )
        return await ModelService.create_model(model)

    async def import_from_lerobot_directory(
        self,
        *,
        source_dir: Path,
        project_id: UUID,
        dataset_id: UUID,
        model_name: str,
        move: bool = False,
        base_model_id: UUID | None = None,
        version: int = 1,
        allow_unknown_policy: bool = False,
    ) -> Model:
        """Import a LeRobot policy directory as a PAS model (best effort)."""
        if not source_dir.exists() or not source_dir.is_dir():
            raise InvalidArchiveError(f"LeRobot model directory does not exist: {source_dir}")

        settings = get_settings()
        dataset = await DatasetService.get_dataset_by_id(dataset_id)
        if dataset.project_id != project_id:
            raise InvalidArchiveError("Dataset does not belong to the specified project")

        reader = DirectoryModelReader(source_dir)
        policy_name, config_dict = self._inspect_lerobot_model(reader, allow_unknown_policy=allow_unknown_policy)

        model_dir = settings.models_dir / str(uuid4())
        lerobot_target_dir = model_dir / "exports" / "lerobot"

        try:
            if move:
                model_dir.mkdir(parents=True, exist_ok=True)
                lerobot_target_dir.parent.mkdir(parents=True, exist_ok=True)
                await asyncio.to_thread(shutil.move, str(source_dir), str(lerobot_target_dir))
            else:
                await asyncio.to_thread(shutil.copytree, source_dir, lerobot_target_dir)

            self._write_import_metadata_files(model_dir, source_dir, policy_name, config_dict, reader)

            policy = self._load_policy_from_lerobot_dir(
                lerobot_target_dir,
                policy_name=policy_name,
                original_source_dir=source_dir,
            )
            self._write_model_checkpoint(policy, model_dir / "model.ckpt")
            self._export_all_supported_backends(policy, model_dir)

            return await self._finalize_import(
                model_dir=model_dir,
                dataset=dataset,
                model_name=model_name,
                policy=policy_name,
                base_model_id=base_model_id,
                version=version,
            )
        except Exception:
            shutil.rmtree(model_dir, ignore_errors=True)
            raise

    def _inspect_model(self, reader: ModelReader) -> str:
        """Validate model structure and infer policy."""
        for required in _REQUIRED_FILES:
            if not reader.file_exists(required):
                raise InvalidArchiveError(f"Model is missing required file '{required}'")

        torch_manifest = self._read_manifest(reader, _TORCH_MANIFEST_PATH)
        self._validate_torch_artifact(torch_manifest, reader)
        return self._infer_policy(torch_manifest, _TORCH_MANIFEST_PATH)

    def _read_manifest(self, reader: ModelReader, path: str) -> dict[str, Any]:
        """Read and validate a manifest JSON file."""
        data = reader.read_json(path)
        if data is None:
            raise InvalidArchiveError(f"Model is missing required file '{path}'")
        if data.get("format") != "policy_package":
            raise InvalidArchiveError(f"Manifest '{path}' must declare format='policy_package'")
        return data

    def _validate_torch_artifact(self, torch_manifest: dict[str, Any], reader: ModelReader) -> None:
        """Validate that the torch artifact referenced in the manifest exists."""
        torch_artifact = self._extract_torch_artifact_path(torch_manifest, _TORCH_MANIFEST_PATH)
        artifact_path = f"exports/torch/{torch_artifact}"
        if not reader.file_exists(artifact_path):
            raise InvalidArchiveError(
                f"Manifest '{_TORCH_MANIFEST_PATH}' references missing torch artifact '{torch_artifact}'"
            )

    @staticmethod
    def _extract_torch_artifact_path(torch_manifest: dict[str, Any], label: str) -> str:
        """Extract and validate the torch artifact path from the manifest."""
        model_section = torch_manifest.get("model")
        if not isinstance(model_section, dict):
            raise InvalidArchiveError(f"Manifest '{label}' is missing object field 'model'")

        artifacts = model_section.get("artifacts")
        if not isinstance(artifacts, dict):
            raise InvalidArchiveError(f"Manifest '{label}' is missing object field 'model.artifacts'")

        torch_artifact = artifacts.get("torch")
        if not isinstance(torch_artifact, str) or not torch_artifact.strip():
            raise InvalidArchiveError(f"Manifest '{label}' is missing non-empty 'model.artifacts.torch' entry")

        artifact_path = Path(torch_artifact)
        if artifact_path.is_absolute() or ".." in artifact_path.parts:
            raise InvalidArchiveError(f"Manifest '{label}' contains unsafe torch artifact path '{torch_artifact}'")

        return torch_artifact

    def _infer_policy(self, manifest: dict[str, Any], manifest_path: str) -> str:
        """Extract and validate the policy name from the manifest."""
        policy_section = manifest.get("policy")
        if not isinstance(policy_section, dict):
            raise InvalidArchiveError(f"Manifest '{manifest_path}' is missing 'policy' section")

        policy_name = policy_section.get("name")
        if not isinstance(policy_name, str) or not policy_name:
            raise InvalidArchiveError(f"Manifest '{manifest_path}' is missing 'policy.name'")

        if policy_name not in _SUPPORTED_POLICIES:
            raise InvalidArchiveError(
                f"Manifest '{manifest_path}' declares unsupported policy '{policy_name}'. "
                f"Supported policies are: {', '.join(sorted(_SUPPORTED_POLICIES))}"
            )

        return policy_name

    def _inspect_lerobot_model(self, reader: ModelReader, *, allow_unknown_policy: bool) -> tuple[str, dict[str, Any]]:
        """Validate a LeRobot model directory and infer policy name."""
        config = reader.read_json(_LEROBOT_CONFIG_PATH)
        if config is None:
            raise InvalidArchiveError(f"LeRobot model is missing required file '{_LEROBOT_CONFIG_PATH}'")

        policy_name = config.get("type")
        if not isinstance(policy_name, str) or not policy_name.strip():
            raise InvalidArchiveError("LeRobot config.json is missing a non-empty 'type' field")

        policy_name = policy_name.lower()
        if policy_name not in _SUPPORTED_POLICIES and not allow_unknown_policy:
            raise InvalidArchiveError(
                f"LeRobot policy '{policy_name}' is not supported by default. "
                "Re-run with --allow-unknown-policy for best-effort import."
            )

        if not (reader.file_exists("model.safetensors") or reader.file_exists("model.pt")):
            raise InvalidArchiveError("LeRobot model must contain either 'model.safetensors' or 'model.pt'")

        return policy_name, config

    def _write_import_metadata_files(
        self,
        model_dir: Path,
        source_dir: Path,
        policy_name: str,
        config_dict: dict[str, Any],
        reader: ModelReader,
    ) -> None:
        """Create PAS-compatible metadata files under version_0/."""
        version_dir = model_dir / "version_0"
        version_dir.mkdir(parents=True, exist_ok=True)

        train_config = reader.read_json(_LEROBOT_TRAIN_CONFIG_PATH)
        hparams = {
            "policy": policy_name,
            "source": "lerobot",
            "imported_from": str(source_dir),
            "lerobot_config": config_dict,
        }
        if train_config is not None:
            hparams["lerobot_train_config"] = train_config

        hparams_path = version_dir / "hparams.yaml"
        with hparams_path.open("w", encoding="utf-8") as fobj:
            yaml.safe_dump(hparams, fobj, sort_keys=False)

        metrics_path = version_dir / "metrics.csv"
        with metrics_path.open("w", encoding="utf-8", newline="") as fobj:
            writer = csv.writer(fobj)
            writer.writerow(["step", "imported"])
            writer.writerow([0, 1])

    def _load_policy_from_lerobot_dir(self, model_dir: Path, *, policy_name: str, original_source_dir: Path) -> object:
        """Load a policy from a LeRobot directory, preferring native wrappers when possible."""
        from physicalai.policies import ACT, Pi05, SmolVLA
        from physicalai.policies.lerobot import LeRobotPolicy

        native_loaders: dict[str, type] = {
            "act": ACT,
            "pi05": Pi05,
            "smolvla": SmolVLA,
        }

        native_cls = native_loaders.get(policy_name)
        if native_cls is not None:
            model_root = model_dir.parent.parent
            source_model_root = original_source_dir.parent.parent
            checkpoint_candidates = [
                model_root / "model.ckpt",
                model_root / "last.ckpt",
                source_model_root / "model.ckpt",
                source_model_root / "last.ckpt",
                model_dir / "model.ckpt",
                model_dir / "last.ckpt",
            ]

            for checkpoint_path in checkpoint_candidates:
                if checkpoint_path.exists():
                    return native_cls.load_from_checkpoint(str(checkpoint_path))

            if policy_name == "smolvla":
                return SmolVLA(pretrained_name_or_path=model_dir)
            if policy_name == "pi05":
                return Pi05(pretrained_name_or_path=model_dir)
            if policy_name == "act":
                act_checkpoint_candidates = [
                    model_dir / "model.pt",
                    model_dir / "model.ckpt",
                ]
                for checkpoint_path in act_checkpoint_candidates:
                    if checkpoint_path.exists():
                        try:
                            return ACT.load_from_checkpoint(str(checkpoint_path), weights_only=False)
                        except Exception:  # noqa: BLE001
                            continue
                if (model_dir / "model.safetensors").exists():
                    return self._load_native_act_from_lerobot_safetensors(model_dir)

        return LeRobotPolicy.from_pretrained(model_dir)

    def _load_native_act_from_lerobot_safetensors(self, model_dir: Path) -> object:
        """Build native ACT policy from a LeRobot safetensors directory."""
        from loguru import logger
        from safetensors.torch import load_file

        from physicalai.policies import ACT

        config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        preprocessor = json.loads((model_dir / "policy_preprocessor.json").read_text(encoding="utf-8"))

        normalizer_step = None
        for step in preprocessor.get("steps", []):
            step_name = str(step.get("registry_name", step.get("type", ""))).lower()
            if "normalizer" in step_name:
                normalizer_step = step
                break

        if normalizer_step is None:
            raise InvalidArchiveError("LeRobot ACT preprocessor is missing a normalizer step")

        features = normalizer_step.get("config", {}).get("features")
        if not isinstance(features, dict):
            raise InvalidArchiveError("LeRobot ACT preprocessor normalizer step is missing feature definitions")

        stats_file = normalizer_step.get("state_file")
        tensor_stats = {}
        if isinstance(stats_file, str) and stats_file:
            stats_path = model_dir / stats_file
            if stats_path.exists():
                tensor_stats = load_file(str(stats_path))

        dataset_stats: dict[str, dict[str, Any]] = {}

        def _identity_stats(shape: tuple[int, ...]) -> tuple[Any, Any]:
            if len(shape) >= 3:
                channels = shape[0]
                return [[[0.0]] for _ in range(channels)], [[[1.0]] for _ in range(channels)]
            dim = shape[0] if shape else 1
            return [0.0] * dim, [1.0] * dim

        for feature_name, feature_spec in features.items():
            if not isinstance(feature_spec, dict):
                continue
            feature_type = str(feature_spec.get("type", ""))
            if feature_type not in {"STATE", "VISUAL", "ACTION"}:
                continue

            raw_shape = feature_spec.get("shape")
            if not isinstance(raw_shape, list) or not all(isinstance(v, int) for v in raw_shape):
                continue
            shape = tuple(raw_shape)

            if feature_name.startswith("observation.images."):
                camera_name = feature_name.split(".", 2)[2]
                target_key = f"observation.{camera_name}"
                short_name = camera_name
            elif feature_name == "observation.state":
                target_key = feature_name
                short_name = "state"
            elif feature_name == "action":
                target_key = feature_name
                short_name = "action"
            else:
                target_key = feature_name
                short_name = feature_name.split(".")[-1]

            mean_key = f"{feature_name}.mean"
            std_key = f"{feature_name}.std"
            if mean_key in tensor_stats and std_key in tensor_stats:
                mean_value = tensor_stats[mean_key].cpu().tolist()
                std_value = tensor_stats[std_key].cpu().tolist()
            else:
                mean_value, std_value = _identity_stats(shape)

            dataset_stats[target_key] = {
                "name": short_name,
                "type": feature_type,
                "shape": shape,
                "mean": mean_value,
                "std": std_value,
            }

        visual_shapes = [
            tuple(spec.get("shape", []))
            for spec in features.values()
            if isinstance(spec, dict) and spec.get("type") == "VISUAL"
        ]
        image_size = (384, 384)
        if visual_shapes and len(visual_shapes[0]) >= 3:
            image_size = (int(visual_shapes[0][1]), int(visual_shapes[0][2]))

        act_config_keys = {
            "n_obs_steps",
            "chunk_size",
            "n_action_steps",
            "vision_backbone",
            "pretrained_backbone_weights",
            "replace_final_stride_with_dilation",
            "pre_norm",
            "dim_model",
            "n_heads",
            "dim_feedforward",
            "feedforward_activation",
            "n_encoder_layers",
            "n_decoder_layers",
            "use_vae",
            "latent_dim",
            "n_vae_encoder_layers",
            "temporal_ensemble_coeff",
            "dropout",
            "kl_weight",
            "optimizer_lr",
            "optimizer_weight_decay",
            "optimizer_grad_clip_norm",
        }
        act_kwargs = {k: v for k, v in config.items() if k in act_config_keys}
        act_kwargs["image_size"] = image_size

        policy = ACT(dataset_stats=dataset_stats, **act_kwargs)

        if policy.model is None:
            raise RuntimeError("Failed to initialize ACT model from LeRobot dataset stats")

        raw_weights = load_file(str(model_dir / "model.safetensors"))
        mapped_weights = {key.removeprefix("model."): value for key, value in raw_weights.items()}
        missing, unexpected = policy.model._model.load_state_dict(mapped_weights, strict=False)  # noqa: SLF001
        if missing:
            logger.warning("ACT safetensors load: missing {} keys", len(missing))
        if unexpected:
            logger.warning("ACT safetensors load: unexpected {} keys", len(unexpected))

        return policy

    def _write_model_checkpoint(self, policy: object, checkpoint_path: Path) -> None:
        """Write a Lightning-compatible checkpoint to model.ckpt."""
        if not hasattr(policy, "state_dict"):
            raise RuntimeError("Loaded policy does not expose state_dict()")

        checkpoint: dict[str, Any] = {
            "state_dict": policy.state_dict(),
            "epoch": 0,
            "global_step": 0,
            "pytorch-lightning_version": lightning.__version__,
            "loops": {},
            "hparams_name": "kwargs",
        }

        hparams = getattr(policy, "hparams", None)
        checkpoint["hyper_parameters"] = dict(hparams) if isinstance(hparams, Mapping) else {}

        on_save_checkpoint = getattr(policy, "on_save_checkpoint", None)
        if callable(on_save_checkpoint):
            on_save_checkpoint(checkpoint)

        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
        torch.save(checkpoint, str(checkpoint_path))  # nosec B614

    def _export_all_supported_backends(self, policy: object, model_dir: Path) -> None:
        """Export policy to all reported backends (best effort)."""
        from loguru import logger

        from physicalai.export import ExportablePolicyMixin

        if not isinstance(policy, ExportablePolicyMixin):
            return

        for backend in policy.get_supported_export_backends():
            backend_name = backend.value if hasattr(backend, "value") else str(backend)
            export_dir = model_dir / "exports" / backend_name

            try:
                if export_dir.exists():
                    logger.info("Skipping backend export '{}': already exists", backend_name)
                    continue
                policy.export(export_dir, backend=backend)
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed exporting model to backend '{}': {}", backend_name, exc)
