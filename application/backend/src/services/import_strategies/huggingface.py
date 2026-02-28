"""Import strategy for HuggingFace/LeRobot model archives.

HuggingFace model archives are zip files containing:
- config.json (with "type", "input_features", "output_features")
- model.safetensors
- train_config.json (optional)

The model is loaded via LeRobotPolicy.from_pretrained() and then manually
converted to the Physical AI Studio inference format (exports/torch/{policy}.pt
+ metadata.yaml).
"""

import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from uuid import UUID

from loguru import logger

from exceptions import ImportConversionError, ImportDependencyError, ImportValidationError
from schemas import Model
from services.import_strategies.base import ImportStrategy
from services.model_service import ModelService
from settings import Settings

# Supported HuggingFace/LeRobot policy types for import
SUPPORTED_HF_POLICY_TYPES: set[str] = {"act", "smolvla"}


class HuggingFaceImportStrategy(ImportStrategy):
    """Handles import and conversion of HuggingFace/LeRobot model archives."""

    def __init__(
        self,
        model_service: ModelService,
        settings: Settings,
    ) -> None:
        self._model_service = model_service
        self._settings = settings

    async def import_model(
        self,
        zf: zipfile.ZipFile,
        all_names: list[str],
        name: str,
        project_id: str,
    ) -> Model:
        """Import a model from a HuggingFace/LeRobot model archive."""
        hf_root = self._find_hf_root(zf)
        if hf_root is None:
            raise ImportValidationError("Could not locate config.json in the archive.")

        prefix = f"{hf_root}/" if hf_root else ""

        # Parse and validate config.json
        config = self._read_json_entry(zf, f"{prefix}config.json")
        policy_type = self._validate_hf_config(config)

        # Optional train_config.json
        train_config = self._try_read_json_entry(zf, f"{prefix}train_config.json", all_names)

        # Build Model
        model = Model(
            name=name,
            path="",
            policy=policy_type,
            properties={
                "source": "huggingface",
                "config": config,
                **({"train_config": train_config} if train_config else {}),
            },
            project_id=UUID(project_id),
            dataset_id=None,
            snapshot_id=None,
        )
        model_dir = self._settings.models_dir / str(model.id)
        model_dir.mkdir(parents=True, exist_ok=True)
        model.path = str(model_dir)

        try:
            self._extract_and_convert(zf, all_names, prefix, model_dir, policy_type)
        except (ImportValidationError, ImportConversionError, ImportDependencyError):
            shutil.rmtree(model_dir, ignore_errors=True)
            raise
        except Exception as e:
            shutil.rmtree(model_dir, ignore_errors=True)
            logger.exception("Unexpected error during HuggingFace model import")
            raise ImportConversionError(f"Unexpected error during model import: {e}") from e

        # Verify export artifacts were created
        exports_dir = model_dir / "exports"
        if not exports_dir.is_dir() or not any(exports_dir.rglob("*")):
            shutil.rmtree(model_dir, ignore_errors=True)
            raise ImportValidationError("Model conversion succeeded but produced no export artifacts.")

        return await self._model_service.create_model(model)

    # ------------------------------------------------------------------
    # Extraction and conversion
    # ------------------------------------------------------------------

    def _extract_and_convert(
        self,
        zf: zipfile.ZipFile,
        all_names: list[str],
        prefix: str,
        model_dir: Path,
        policy_type: str,
    ) -> None:
        """Extract HF files to a temp directory, validate, and convert."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            hf_files_extracted = 0
            for entry in all_names:
                if entry.endswith("/"):
                    continue
                if prefix:
                    if not entry.startswith(prefix):
                        continue
                    relative = entry[len(prefix) :]
                else:
                    relative = entry
                if not relative:
                    continue

                target = tmp_path / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(zf.read(entry))
                hf_files_extracted += 1

            if hf_files_extracted == 0:
                raise ImportValidationError("No files could be extracted from the archive.")

            if not (tmp_path / "config.json").is_file():
                raise ImportValidationError("config.json was not found after extraction.")
            if not (tmp_path / "model.safetensors").is_file():
                raise ImportValidationError("model.safetensors was not found after extraction.")

            exports_dir = model_dir / "exports"
            self._convert_hf_model(tmp_path, exports_dir, policy_type)

    def _convert_hf_model(self, source_dir: Path, export_dir: Path, policy_type: str) -> None:
        """Load a HuggingFace model and export it to physicalai format.

        Uses LeRobotPolicy.from_pretrained() to load the model, then manually
        replicates the Export.to_torch() logic to produce the checkpoint and
        metadata files. This is necessary because:
        - Base physicalai policy classes have Export but no from_pretrained()
        - LeRobot wrapper classes have from_pretrained() but no Export mixin
        So we bridge the gap by loading via LeRobot and exporting manually.

        Args:
            source_dir: Directory containing config.json and model.safetensors.
            export_dir: Target directory for exports (e.g., model_dir/exports).
            policy_type: The policy type string (e.g., "act", "smolvla").
        """
        try:
            from physicalai.policies.lerobot import LeRobotPolicy
        except ImportError as e:
            raise ImportDependencyError(
                "The physicalai-train library is not available. Cannot convert HuggingFace models without it."
            ) from e

        # Load the HuggingFace model via LeRobotPolicy.from_pretrained()
        try:
            policy = LeRobotPolicy.from_pretrained(str(source_dir))
        except ImportError as e:
            raise ImportDependencyError(f"Failed to load HuggingFace model: missing dependency. {e}") from e
        except Exception as e:
            raise ImportConversionError(f"Failed to load HuggingFace model: {e}") from e

        # Manually replicate Export.to_torch() + _create_metadata()
        # The loaded policy has state_dict() and hparams but no export() method.
        try:
            import lightning
            import torch
            import yaml

            from physicalai.train import __version__

            for backend in self._settings.supported_backends:
                if backend != "torch":
                    logger.warning("Skipping unsupported backend '%s' for HuggingFace import.", backend)
                    continue

                backend_dir = export_dir / backend
                backend_dir.mkdir(parents=True, exist_ok=True)

                # Use policy.policy_name for the filename (e.g., "act", "smolvla")
                policy_name = getattr(policy, "policy_name", policy_type)
                model_path = backend_dir / f"{policy_name}.pt"

                # Build checkpoint matching Export.to_torch() structure
                checkpoint = {
                    "state_dict": policy.state_dict(),
                    "epoch": 0,
                    "global_step": 0,
                    "pytorch-lightning_version": lightning.__version__,
                    "loops": {},
                    "hparams_name": "kwargs",
                    "hyper_parameters": dict(policy.hparams),
                }

                # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
                torch.save(checkpoint, str(model_path))  # nosec B614

                # Write metadata.yaml matching Export._create_metadata() structure
                metadata = {
                    "physicalai_train_version": __version__,
                    "policy_class": f"{policy.__class__.__module__}.{policy.__class__.__name__}",
                    "backend": backend,
                }
                metadata_path = backend_dir / "metadata.yaml"
                with metadata_path.open("w") as f:
                    yaml.dump(metadata, f, default_flow_style=False)

                logger.info(
                    "Converted HuggingFace %s model to %s format: %s",
                    policy_type,
                    backend,
                    model_path,
                )
        except (ImportDependencyError, ImportConversionError):
            raise
        except Exception as e:
            raise ImportConversionError(f"Failed to convert model to inference format: {e}") from e

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_hf_root(zf: zipfile.ZipFile) -> str | None:
        """Find the directory prefix containing config.json in a zip archive.

        Returns "" for root, "subdir" for one level of nesting, or None if not found.
        """
        for name in zf.namelist():
            if name == "config.json" or name.endswith("/config.json"):
                parts = Path(name).parts
                if len(parts) == 1:
                    return ""
                elif len(parts) == 2:
                    return parts[0]
        return None

    @staticmethod
    def _validate_hf_config(config: dict) -> str:
        """Validate HuggingFace config.json and return the policy type."""
        policy_type = config.get("type")
        if not policy_type:
            raise ImportValidationError("config.json is missing the 'type' field. Cannot determine the policy type.")

        policy_type = policy_type.lower()
        if policy_type not in SUPPORTED_HF_POLICY_TYPES:
            supported = ", ".join(sorted(SUPPORTED_HF_POLICY_TYPES))
            raise ImportValidationError(f"Unsupported policy type '{policy_type}'. Supported types are: {supported}.")

        if "input_features" not in config or "output_features" not in config:
            raise ImportValidationError(
                "config.json is missing required fields: input_features, output_features. "
                "These are needed for model conversion."
            )

        return policy_type

    @staticmethod
    def _read_json_entry(zf: zipfile.ZipFile, entry_name: str) -> dict:
        """Read and parse a JSON file from the zip archive."""
        try:
            raw = zf.read(entry_name)
            return json.loads(raw)
        except (KeyError, json.JSONDecodeError) as e:
            raise ImportValidationError(f"Invalid {entry_name}: unable to parse as JSON. {e}") from e

    @staticmethod
    def _try_read_json_entry(zf: zipfile.ZipFile, entry_name: str, all_names: list[str]) -> dict | None:
        """Try to read a JSON entry; return None if absent or unparseable."""
        if entry_name not in all_names:
            return None
        try:
            return json.loads(zf.read(entry_name))
        except (json.JSONDecodeError, KeyError):
            logger.warning("'%s' found but could not be parsed; skipping.", entry_name)
            return None
