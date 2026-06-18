import json
import re
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from huggingface_hub import ModelCard
from physicalai.inference.manifest import Manifest

from models.model_card import ModelCardContext, calibration_values, environment_context, select_runtime_calibration
from schemas.calibration import Calibration
from schemas.dataset import Dataset
from schemas.environment import EnvironmentWithRelations
from schemas.model import Model


@dataclass(frozen=True, slots=True)
class ModelManifestConfig:
    manifest_filename: str = "manifest.json"
    readme_filename: str = "README.md"
    environment_filename: str = "environment.json"
    calibration_filename: str = "calibration.json"
    torch_backend: str = "torch"
    policy_package_format: str = "policy_package"
    environment_format: str = "physical_ai_studio_environment"
    model_card_template: str = "model_card.md"
    model_card_templates_dir: Path = Path(__file__).parents[1] / "models" / "model_card"

    @property
    def torch_manifest_path(self) -> Path:
        return Path("exports") / self.torch_backend / self.manifest_filename

    @property
    def model_card_template_path(self) -> Path:
        return self.model_card_templates_dir / self.model_card_template

class ModelManifestService:
    """Generate package-level model metadata files."""

    _CONFIG = ModelManifestConfig()

    @classmethod
    def _config(cls) -> ModelManifestConfig:
        return cls._CONFIG

    @staticmethod
    def write_root_manifest(model_dir: Path) -> Path | None:
        """Create a root manifest from the torch export manifest.

        The torch export is the canonical checkpoint representation. Its manifest
        can also serve as the package manifest when artifact paths are rewritten
        from torch-export-relative paths to model-root-relative paths.
        """
        config = ModelManifestService._config()
        torch_manifest_path = model_dir / config.torch_manifest_path
        if not torch_manifest_path.is_file():
            return None

        manifest = ModelManifestService._load_manifest(torch_manifest_path)
        if manifest is None:
            return None

        root_manifest = ModelManifestService._with_root_relative_artifacts(manifest)
        root_manifest_path = model_dir / config.manifest_filename
        root_manifest.save(root_manifest_path)

        return root_manifest_path

    @staticmethod
    def write_environment_description(
        model_dir: Path,
        environment: EnvironmentWithRelations,
        calibrations: dict[UUID, Calibration] | None = None,
    ) -> Path:
        """Create a sanitized training environment description."""
        environment_path = model_dir / ModelManifestService._config().environment_filename
        environment_path.write_text(
            json.dumps(
                environment_context(
                    environment,
                    calibrations or {},
                    environment_format=ModelManifestService._config().environment_format,
                ),
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
        calibration = select_runtime_calibration(environment, calibrations or {})
        if calibration is None:
            return None

        calibration_path = model_dir / ModelManifestService._config().calibration_filename
        calibration_path.write_text(
            json.dumps(calibration_values(calibration), indent=2),
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
        config = ModelManifestService._config()
        manifest = ModelManifestService._load_manifest(model_dir / config.manifest_filename)
        if manifest is None:
            return None

        policy_name = manifest.policy.name
        export_manifests = ModelManifestService._load_export_manifests(model_dir)
        card_data = ModelCardContext.card_data(
            policy_name=policy_name,
            model=model,
            backends=sorted(export_manifests),
        )
        context = ModelCardContext.from_manifest(
            manifest=manifest,
            export_manifests=export_manifests,
            model=model,
            dataset=dataset,
            environment=environment,
            calibrations=calibrations or {},
        )
        card = ModelCard.from_template(
            card_data,
            template_str=ModelManifestService._read_template(),
            **context.template_kwargs(),
        )
        card.validate()

        readme_path = model_dir / config.readme_filename
        card.save(readme_path)
        readme_path.write_text(
            ModelManifestService._compact_markdown(readme_path.read_text(encoding="utf-8")),
            encoding="utf-8",
        )
        return readme_path

    @staticmethod
    def _load_manifest(manifest_path: Path) -> Manifest | None:
        policy_package_format = ModelManifestService._config().policy_package_format
        try:
            manifest = Manifest.load(manifest_path)
        except (FileNotFoundError, ValueError):
            return None

        return manifest if manifest.format == policy_package_format else None

    @staticmethod
    def _load_export_manifests(model_dir: Path) -> dict[str, Manifest]:
        config = ModelManifestService._config()
        root_manifest = ModelManifestService._load_manifest(model_dir / config.manifest_filename)
        manifests = dict.fromkeys(root_manifest.model.artifacts, root_manifest) if root_manifest else {}

        exports_dir = model_dir / "exports"
        if not exports_dir.is_dir():
            return manifests

        for manifest_path in sorted(exports_dir.glob(f"*/{config.manifest_filename}")):
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
        torch_backend = ModelManifestService._config().torch_backend
        artifacts = {
            backend: str(Path("exports") / torch_backend / artifact)
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
    def _read_template() -> str:
        return ModelManifestService._config().model_card_template_path.read_text(encoding="utf-8")

    @staticmethod
    def _compact_markdown(markdown: str) -> str:
        markdown = re.sub(r"\n[ \t]*\n(?=})", "\n", markdown)
        return re.sub(r"\n{3,}", "\n\n", markdown).rstrip() + "\n"
