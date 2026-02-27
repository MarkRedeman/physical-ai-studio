"""Service for importing models from zip archives.

Supports two archive formats:
- Physical AI Studio exports (zip with exports/*/metadata.yaml)
- HuggingFace/LeRobot models (zip with config.json + model.safetensors)
"""

import io
import json
import logging
import shutil
import tempfile
import zipfile
from pathlib import Path
from uuid import UUID

import yaml

from exceptions import ImportConversionError, ImportDependencyError, ImportValidationError
from schemas import Model
from services.dataset_service import DatasetService
from services.model_service import ModelService
from services.snapshot_service import SnapshotService
from settings import Settings, get_settings

logger = logging.getLogger(__name__)

# Known policy class paths mapped to short policy names
POLICY_CLASS_MAP: dict[str, str] = {
    "physicalai.policies.act.policy.ACT": "act",
    "physicalai.policies.pi0.policy.Pi0": "pi0",
    "physicalai.policies.smolvla.policy.SmolVLA": "smolvla",
    "physicalai.policies.lerobot.universal.LeRobotPolicy": "lerobot",
}

# Supported HuggingFace/LeRobot policy types for import
SUPPORTED_HF_POLICY_TYPES: set[str] = {"act", "smolvla"}


class ModelImportService:
    """Handles model import from zip archives in multiple formats."""

    def __init__(
        self,
        model_service: ModelService,
        dataset_service: DatasetService,
        snapshot_service: SnapshotService,
        settings: Settings | None = None,
    ) -> None:
        self._model_service = model_service
        self._dataset_service = dataset_service
        self._snapshot_service = snapshot_service
        self._settings = settings or get_settings()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def import_model(self, file_content: bytes, filename: str, name: str, project_id: str) -> Model:
        """Import a model from raw zip bytes.

        Detects the archive format and delegates to the appropriate handler.

        Args:
            file_content: Raw bytes of the uploaded zip file.
            filename: Original filename (used for .zip validation).
            name: Display name for the imported model.
            project_id: Project to associate the model with.

        Returns:
            The created Model record.

        Raises:
            ImportValidationError: On bad archive, unrecognized format, or missing data.
            ImportConversionError: On HuggingFace conversion failure.
            ImportDependencyError: If physicalai-train is unavailable.
        """
        self._validate_filename(filename)
        zf = self._open_zip(file_content)

        with zf:
            all_names = zf.namelist()
            if not all_names:
                raise ImportValidationError("The zip archive is empty.")

            archive_format = self._detect_archive_format(all_names)

            if archive_format == "unknown":
                self._raise_unknown_format_error(all_names)

            if archive_format == "export":
                return await self._import_from_export(zf, all_names, name, project_id)
            else:
                return await self._import_from_huggingface(zf, all_names, name, project_id)

    # ------------------------------------------------------------------
    # Format detection
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_filename(filename: str | None) -> None:
        if not filename or not filename.endswith(".zip"):
            raise ImportValidationError("Uploaded file must be a .zip archive.")

    @staticmethod
    def _open_zip(content: bytes) -> zipfile.ZipFile:
        try:
            return zipfile.ZipFile(io.BytesIO(content))
        except zipfile.BadZipFile:
            raise ImportValidationError("Uploaded file is not a valid zip archive.")

    @staticmethod
    def _detect_archive_format(all_names: list[str]) -> str:
        """Detect whether a zip contains a Studio export or a HuggingFace model.

        Returns "export", "huggingface", or "unknown".
        """
        basenames: set[str] = set()
        has_exports_metadata = False

        for name in all_names:
            parts = Path(name).parts
            if name.endswith("metadata.yaml") and "exports" in parts:
                has_exports_metadata = True

            if len(parts) == 1:
                basenames.add(parts[0])
            elif len(parts) == 2:
                basenames.add(parts[1])

        if has_exports_metadata:
            return "export"

        if "config.json" in basenames and "model.safetensors" in basenames:
            return "huggingface"

        return "unknown"

    @staticmethod
    def _raise_unknown_format_error(all_names: list[str]) -> None:
        """Raise an ImportValidationError with a helpful message about the unrecognized format."""
        basenames: set[str] = set()
        for n in all_names:
            parts = Path(n).parts
            basenames.add(parts[-1] if parts else n)

        if "config.json" in basenames and "model.safetensors" not in basenames:
            raise ImportValidationError(
                "Found config.json but model.safetensors is missing. "
                "A HuggingFace model archive must contain both config.json "
                "and model.safetensors."
            )

        raise ImportValidationError(
            "Unrecognized model archive format. Expected either a "
            "Physical AI Studio export (containing exports/*/metadata.yaml) "
            "or a HuggingFace model (containing config.json and model.safetensors)."
        )

    # ------------------------------------------------------------------
    # Studio export import
    # ------------------------------------------------------------------

    async def _import_from_export(
        self,
        zf: zipfile.ZipFile,
        all_names: list[str],
        name: str,
        project_id: str,
    ) -> Model:
        """Import a model from a Physical AI Studio export archive."""
        zip_root = Path(all_names[0]).parts[0]

        metadata_entries = [n for n in all_names if n.endswith("metadata.yaml") and "/exports/" in n]
        if not metadata_entries:
            raise ImportValidationError("Invalid model archive: no exports/*/metadata.yaml found.")

        metadata_path = metadata_entries[0]
        metadata = yaml.safe_load(zf.read(metadata_path))
        policy = self._extract_policy_from_metadata(metadata)

        # Read manifest.json if present
        manifest: dict = {}
        manifest_path = f"{zip_root}/manifest.json"
        if manifest_path in all_names:
            manifest = json.loads(zf.read(manifest_path))

        # Create model and its directory
        model = Model(
            name=name,
            path="",
            policy=policy,
            properties=manifest.get("properties", {}),
            project_id=UUID(project_id),
            dataset_id=None,
            snapshot_id=None,
        )
        model_dir = self._settings.models_dir / str(model.id)
        model_dir.mkdir(parents=True, exist_ok=True)
        model.path = str(model_dir)

        # Try to reconnect dataset/snapshot from manifest
        model.dataset_id = await self._try_resolve_dataset(manifest.get("original_dataset_id"))
        model.snapshot_id = await self._try_resolve_snapshot(manifest.get("original_snapshot_id"))

        # Extract export artifacts
        for entry in all_names:
            if entry.endswith("/"):
                continue
            entry_path = Path(entry)
            try:
                relative = entry_path.relative_to(zip_root)
            except ValueError:
                continue
            if not str(relative).startswith("exports"):
                continue
            target = model_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zf.read(entry))

        # Verify extraction succeeded
        exports_dir = model_dir / "exports"
        if not exports_dir.is_dir() or not any(exports_dir.rglob("*")):
            shutil.rmtree(model_dir, ignore_errors=True)
            raise ImportValidationError("Failed to extract model exports from the archive.")

        return await self._model_service.create_model(model)

    # ------------------------------------------------------------------
    # HuggingFace import
    # ------------------------------------------------------------------

    async def _import_from_huggingface(
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
            self._extract_and_convert_hf(zf, all_names, prefix, model_dir, policy_type)
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

    def _extract_and_convert_hf(
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

    # ------------------------------------------------------------------
    # HF conversion
    # ------------------------------------------------------------------

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
    def _extract_policy_from_metadata(metadata: dict) -> str:
        """Extract short policy name from metadata.yaml contents."""
        policy_class = metadata.get("policy_class", "")
        if policy_class in POLICY_CLASS_MAP:
            return POLICY_CLASS_MAP[policy_class]
        parts = policy_class.rsplit(".", 1)
        return parts[-1].lower() if parts else "unknown"

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

    async def _try_resolve_dataset(self, dataset_id_str: str | None) -> UUID | None:
        """Try to resolve a dataset ID; return None if it no longer exists."""
        if not dataset_id_str:
            return None
        try:
            dataset = await self._dataset_service.get_dataset_by_id(UUID(dataset_id_str))
            return dataset.id
        except Exception:
            return None

    async def _try_resolve_snapshot(self, snapshot_id_str: str | None) -> UUID | None:
        """Try to resolve a snapshot ID; return None if it no longer exists."""
        if not snapshot_id_str:
            return None
        try:
            snapshot = await self._snapshot_service.get_snapshot_by_id(UUID(snapshot_id_str))
            return snapshot.id
        except Exception:
            return None
