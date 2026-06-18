from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True, slots=True)
class PolicyCardText:
    title: str
    overview: str
    intended_use: str
    reproduction: str


@dataclass(frozen=True, slots=True)
class PolicyCardTextFactory:
    default_text: PolicyCardText
    policy_texts: Mapping[str, PolicyCardText]

    _DEFAULT_FACTORY: ClassVar["PolicyCardTextFactory | None"] = None

    @classmethod
    def default(cls) -> "PolicyCardTextFactory":
        if cls._DEFAULT_FACTORY is None:
            default_text = PolicyCardText(
                title="PhysicalAI Model",
                overview=(
                    "This is a vision-language-action policy for robot control from multimodal observations and "
                    "language instructions."
                ),
                intended_use=(
                    "Use this model for language-conditioned robot inference in setups matching the training dataset, "
                    "robot embodiment, camera viewpoints, and task phrasing. Validate behavior in simulation or a safe "
                    "test cell before running on hardware."
                ),
                reproduction=(
                    "To reproduce behavior on your own hardware, match the exported I/O specification, robot type, "
                    "camera viewpoints, control frequency, language prompts, and calibration values from "
                    "`environment.json` as closely as possible."
                ),
            )

            cls._DEFAULT_FACTORY = cls(
                default_text=default_text,
                policy_texts={
                    "act": PolicyCardText(
                        title="Action Chunking Transformer (ACT)",
                        overview=(
                            "[Action Chunking with Transformers (ACT)](https://huggingface.co/papers/2304.13705) "
                            "is an imitation-learning policy that predicts short action chunks from robot state and "
                            "visual observations. The robot can execute those chunks as a sequence of real-world "
                            "movements."
                        ),
                        intended_use=(
                            "Use this model for robot imitation-learning inference in setups matching the training "
                            "dataset, robot embodiment, camera viewpoints, and task instructions. Validate behavior in "
                            "simulation or a safe test cell before running on hardware."
                        ),
                        reproduction=(
                            "To reproduce behavior on your own hardware, match the exported I/O specification, robot "
                            "type, camera viewpoints, control frequency, and calibration values from "
                            "`environment.json` as closely as possible."
                        ),
                    ),
                    "smolvla": PolicyCardText(
                        title="SmolVLA",
                        overview=(
                            "[SmolVLA](https://huggingface.co/papers/2506.01844) is a compact vision-language-action "
                            "policy for robot control from visual observations, robot state, and language "
                            "instructions. It targets practical fine-tuning and edge deployment use cases."
                        ),
                        intended_use=default_text.intended_use,
                        reproduction=default_text.reproduction,
                    ),
                    "pi05": PolicyCardText(
                        title="Pi0.5",
                        overview=(
                            "Pi0.5 is a vision-language-action policy designed for open-world robot control from "
                            "multimodal observations and language instructions. It can consume images, robot "
                            "state, and task text and produce action chunks for robot execution."
                        ),
                        intended_use=default_text.intended_use,
                        reproduction=default_text.reproduction,
                    ),
                },
            )
        return cls._DEFAULT_FACTORY

    @property
    def default_title(self) -> str:
        return self.default_text.title

    def build(self, policy_name: str) -> PolicyCardText:
        return self.policy_texts.get(policy_name, self.default_text)

    def is_known_policy(self, policy_name: str) -> bool:
        return policy_name in self.policy_texts

    def overview(self, policy_name: str, policy_text: PolicyCardText) -> str:
        if self.is_known_policy(policy_name):
            return policy_text.overview
        if policy_name:
            return (
                f"{policy_name} is a vision-language-action policy for robot control from multimodal observations and "
                "language instructions."
            )
        return policy_text.overview
