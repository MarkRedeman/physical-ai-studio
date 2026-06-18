from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any
from uuid import UUID

import yaml
from huggingface_hub import ModelCardData
from physicalai.inference.manifest import Manifest

from models.model_card.environment import environment_context
from models.model_card.manifest_context import ExportContext, IOGroupContext, export_context, io_group_context
from models.model_card.policy_text import PolicyCardTextFactory
from models.model_card.runtime import ObservationSampleContext, control_loop_command, observation_context
from schemas.calibration import Calibration
from schemas.dataset import Dataset
from schemas.environment import EnvironmentWithRelations
from schemas.model import Model


@dataclass(frozen=True, slots=True)
class ModelCardContext:
    model_name: str
    model_title: str
    policy_name: str
    policy_overview: str
    intended_use_text: str
    dataset_name: str | None
    exports: list[ExportContext]
    io_groups: list[IOGroupContext]
    observation_samples: list[ObservationSampleContext]
    environment_name: str | None
    environment_yaml: str | None
    control_loop_command: str
    reproduce_text: str

    @staticmethod
    def display_name(policy_name: str, model: Model | None) -> str:
        default_title = PolicyCardTextFactory.default().default_title
        return model.name if model and model.name else policy_name or default_title

    @staticmethod
    def card_data(policy_name: str, model: Model | None, backends: list[str]) -> ModelCardData:
        display_name = ModelCardContext.display_name(policy_name, model)
        return ModelCardData(
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

    @staticmethod
    def from_manifest(
        manifest: Manifest,
        export_manifests: Mapping[str, Manifest],
        model: Model | None,
        dataset: Dataset | None,
        environment: EnvironmentWithRelations | None,
        calibrations: Mapping[UUID, Calibration],
    ) -> "ModelCardContext":
        policy_name = manifest.policy.name
        factory = PolicyCardTextFactory.default()
        policy_text = factory.build(policy_name)
        display_name = ModelCardContext.display_name(policy_name, model)
        return ModelCardContext(
            model_name=display_name,
            model_title=policy_text.title if factory.is_known_policy(policy_name) else display_name,
            policy_name=policy_name or "unknown",
            policy_overview=factory.overview(policy_name, policy_text),
            intended_use_text=policy_text.intended_use,
            dataset_name=dataset.name if dataset else None,
            exports=export_context(export_manifests),
            io_groups=io_group_context(export_manifests),
            observation_samples=observation_context(export_manifests),
            environment_name=environment.name if environment else None,
            environment_yaml=_environment_yaml(environment, calibrations),
            control_loop_command=control_loop_command(
                environment,
                dataset.default_task if dataset else None,
            ),
            reproduce_text=policy_text.reproduction,
        )

    def template_kwargs(self) -> dict[str, Any]:
        return asdict(self)


def _environment_yaml(
    environment: EnvironmentWithRelations | None,
    calibrations: Mapping[UUID, Calibration],
) -> str | None:
    if environment is None:
        return None

    payload = environment_context(environment, calibrations)
    yaml_environment = {
        "name": payload["name"],
        "robots": [
            {key: value for key, value in robot.items() if key != "teleoperator" and value is not None}
            for robot in payload["robots"]
        ],
        "cameras": payload["cameras"],
    }
    return yaml.safe_dump(yaml_environment, sort_keys=False).strip()
