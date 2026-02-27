"""Import strategy for Physical AI Studio export archives.

Studio exports are zip files containing:
- exports/{backend}/{policy}.pt + metadata.yaml
- manifest.json (optional, for reconnecting dataset/snapshot/model references)
- metrics.csv (optional, training metrics from CSVLogger)
"""

import json
import logging
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from uuid import UUID

import yaml

from exceptions import ImportValidationError
from schemas import Model
from services.dataset_service import DatasetService
from services.import_strategies.base import ImportStrategy
from services.model_service import ModelService
from services.snapshot_service import SnapshotService
from settings import Settings

logger = logging.getLogger(__name__)

# Known policy class paths mapped to short policy names
POLICY_CLASS_MAP: dict[str, str] = {
    "physicalai.policies.act.policy.ACT": "act",
    "physicalai.policies.pi0.policy.Pi0": "pi0",
    "physicalai.policies.smolvla.policy.SmolVLA": "smolvla",
    "physicalai.policies.lerobot.universal.LeRobotPolicy": "lerobot",
}


class PhysicalAIStudioExportImportStrategy(ImportStrategy):
    """Handles import of Physical AI Studio export archives."""

    def __init__(
        self,
        model_service: ModelService,
        dataset_service: DatasetService,
        snapshot_service: SnapshotService,
        settings: Settings,
    ) -> None:
        self._model_service = model_service
        self._dataset_service = dataset_service
        self._snapshot_service = snapshot_service
        self._settings = settings

    async def import_model(
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
            version=manifest.get("version", 1),
            parent_model_id=None,
            created_at=self._parse_created_at(manifest.get("created_at")),
        )
        model_dir = self._settings.models_dir / str(model.id)
        model_dir.mkdir(parents=True, exist_ok=True)
        model.path = str(model_dir)

        # Try to reconnect dataset/snapshot/parent from manifest
        model.dataset_id = await self._try_resolve_dataset(manifest.get("original_dataset_id"))
        model.snapshot_id = await self._try_resolve_snapshot(manifest.get("original_snapshot_id"))
        model.parent_model_id = await self._try_resolve_model(manifest.get("parent_model_id"))

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

        # Extract training metrics if present
        metrics_arcname = f"{zip_root}/metrics.csv"
        if metrics_arcname in all_names:
            metrics_dir = model_dir / "version_0"
            metrics_dir.mkdir(parents=True, exist_ok=True)
            (metrics_dir / "metrics.csv").write_bytes(zf.read(metrics_arcname))

        # Verify extraction succeeded
        exports_dir = model_dir / "exports"
        if not exports_dir.is_dir() or not any(exports_dir.rglob("*")):
            shutil.rmtree(model_dir, ignore_errors=True)
            raise ImportValidationError("Failed to extract model exports from the archive.")

        return await self._model_service.create_model(model)

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

    async def _try_resolve_model(self, model_id_str: str | None) -> UUID | None:
        """Try to resolve a parent model ID; return None if it no longer exists."""
        if not model_id_str:
            return None
        try:
            model = await self._model_service.get_model_by_id(UUID(model_id_str))
            return model.id
        except Exception:
            return None

    @staticmethod
    def _parse_created_at(value: str | None) -> datetime | None:
        """Parse an ISO-format datetime string from the manifest."""
        if not value:
            return None
        try:
            return datetime.fromisoformat(value)
        except (ValueError, TypeError):
            return None
