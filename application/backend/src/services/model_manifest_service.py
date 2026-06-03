import json
from copy import deepcopy
from pathlib import Path
from typing import Any

_MANIFEST_FILENAME = "manifest.json"
_TORCH_BACKEND = "torch"
_TORCH_MANIFEST_PATH = Path("exports") / _TORCH_BACKEND / _MANIFEST_FILENAME
_POLICY_PACKAGE_FORMAT = "policy_package"


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

        manifest = ModelManifestService._read_manifest(torch_manifest_path)
        if manifest is None or manifest.get("format") != _POLICY_PACKAGE_FORMAT:
            return None

        root_manifest = ModelManifestService._with_root_relative_artifacts(manifest)
        root_manifest_path = model_dir / _MANIFEST_FILENAME
        with root_manifest_path.open("w", encoding="utf-8") as f:
            json.dump(root_manifest, f, indent=2)
            f.write("\n")

        return root_manifest_path

    @staticmethod
    def _read_manifest(manifest_path: Path) -> dict[str, Any] | None:
        try:
            with manifest_path.open(encoding="utf-8") as f:
                manifest = json.load(f)
        except (OSError, ValueError):
            return None

        return manifest if isinstance(manifest, dict) else None

    @staticmethod
    def _with_root_relative_artifacts(manifest: dict[str, Any]) -> dict[str, Any]:
        root_manifest = deepcopy(manifest)

        model_section = root_manifest.get("model")
        if not isinstance(model_section, dict):
            return root_manifest

        artifacts = model_section.get("artifacts")
        if not isinstance(artifacts, dict):
            return root_manifest

        model_section["artifacts"] = {
            key: str(Path("exports") / _TORCH_BACKEND / artifact)
            for key, artifact in artifacts.items()
            if isinstance(key, str) and isinstance(artifact, str)
        }

        return root_manifest
