# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Export support for LeRobot-backed policies.

This module is intentionally self-contained and easy to remove: it adds
OpenVINO/ONNX export capability to :class:`~physicalai.policies.lerobot.policy.LeRobotPolicy`
without touching the policy class itself. Consumers that need to re-export a
trained LeRobot policy load it through
:class:`ExportableLeRobotPolicy.load_from_checkpoint` instead of the bare
:class:`~physicalai.policies.lerobot.policy.LeRobotPolicy`, and call
:func:`export_lerobot_policy`.

Why a dedicated module instead of extending :class:`LeRobotPolicy` in place:

- The bare wrapper only advertises the ``torch`` backend and has no I/O schema
  (``inputs_schema``/``outputs_schema`` are ``None``), so the shared export
  machinery refuses ONNX/OpenVINO.
- LeRobot's inner policy ``forward`` computes a *training* loss and cannot be
  traced for inference; the exportable path traces ``predict_action_chunk`` on
  the inner policy instead, which is the pure observation-to-action mapping.
- Keeping the code here means removing it later is a one-file deletion.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import openvino
import torch
from physicalai.inference.data import InferenceFeature, InferenceFeatureDtype, InferenceFeatureType
from physicalai.inference.manifest import ComponentSpec
from physicalai.inference.runners.single_pass import SinglePass
from torch import nn

from physicalai.export.backends import ExportBackend
from physicalai.export.mixin_policy import _match_default_device  # noqa: PLC2701
from physicalai.policies.lerobot.policy import LeRobotPolicy

if TYPE_CHECKING:
    from lerobot.configs.policies import PreTrainedConfig

#: LeRobot feature types that map cleanly onto inference features.
_LEROBOT_TO_INFERENCE_TYPE: dict[str, InferenceFeatureType] = {
    "STATE": InferenceFeatureType.STATE,
    "ENV": InferenceFeatureType.STATE,
    "VISUAL": InferenceFeatureType.VISUAL,
    "ACTION": InferenceFeatureType.ACTION,
    "LANGUAGE": InferenceFeatureType.LANGUAGE,
}

#: Backends the exportable wrapper advertises. Torch is always available; ONNX
#: is the intermediate format the OpenVINO converter consumes.
_SUPPORTED_EXPORT_BACKENDS: list[str | ExportBackend] = [
    ExportBackend.TORCH,
    ExportBackend.OPENVINO,
    ExportBackend.ONNX,
]


def _lerobot_to_inference_type(feature: object) -> InferenceFeatureType | None:
    """Map a LeRobot :class:`PolicyFeature` type to an inference feature type.

    Args:
        feature: A LeRobot :class:`PolicyFeature` (or any object exposing
            ``type`` as a lerobot :class:`FeatureType`).

    Returns:
        The matching inference feature type, or ``None`` for unmapped types.
    """
    feature_type = getattr(feature, "type", None)
    return _LEROBOT_TO_INFERENCE_TYPE.get(getattr(feature_type, "value", str(feature_type)))


def _feature_dtype(ftype: InferenceFeatureType) -> InferenceFeatureDtype:
    """Return the inference dtype for a feature type (strings for language)."""
    return InferenceFeatureDtype.STRING if ftype is InferenceFeatureType.LANGUAGE else InferenceFeatureDtype.FLOAT32


def _feature_shape(feature: object) -> tuple[int, ...]:
    """Return a LeRobot feature's shape as a plain tuple.

    Args:
        feature: A LeRobot :class:`PolicyFeature` exposing ``shape``.

    Returns:
        The feature shape, or ``()`` when it is not defined.
    """
    shape = getattr(feature, "shape", None)
    return tuple(shape) if shape is not None else ()


def build_inputs_schema(policy: LeRobotPolicy) -> list[InferenceFeature]:
    """Describe a LeRobot policy's model inputs for export tracing.

    Shapes mirror the model config's ``input_features`` (e.g. ``(3, H, W)`` for
    images, ``(state_dim,)`` for state), i.e. the raw features the policy's
    preprocessors feed into the model.

    Args:
        policy: The LeRobot policy to describe.

    Returns:
        The list of input features.
    """
    schema: list[InferenceFeature] = []
    for name, feature in (getattr(policy._config, "input_features", None) or {}).items():  # noqa: SLF001
        ftype = _lerobot_to_inference_type(feature)
        if ftype is None:
            continue
        schema.append(
            InferenceFeature(
                ftype=ftype,
                shape=_feature_shape(feature),
                name=str(name),
                dtype=_feature_dtype(ftype),
            ),
        )
    return schema


def build_outputs_schema(policy: LeRobotPolicy) -> list[InferenceFeature]:
    """Describe a LeRobot policy's model outputs for export metadata.

    Action outputs are reported as ``(n_action_steps, *action_feature.shape)``,
    matching what the model returns from ``predict_action_chunk``.

    Args:
        policy: The LeRobot policy to describe.

    Returns:
        The list of output features.
    """
    n_action_steps = getattr(policy._config, "n_action_steps", None)  # noqa: SLF001
    schema: list[InferenceFeature] = []
    for name, feature in (getattr(policy._config, "output_features", None) or {}).items():  # noqa: SLF001
        ftype = _lerobot_to_inference_type(feature)
        if ftype is None:
            continue
        shape = _feature_shape(feature)
        if ftype is InferenceFeatureType.ACTION and n_action_steps is not None:
            shape = (int(n_action_steps), *shape)
        schema.append(
            InferenceFeature(
                ftype=ftype,
                shape=shape,
                name=str(name),
                dtype=_feature_dtype(ftype),
            ),
        )
    return schema


class _LerobotInferenceWrapper(nn.Module):
    """Expose a LeRobot policy's inference path for ONNX tracing.

    ``predict_action_chunk`` on the inner policy is the pure observation-to-
    action mapping (no normalization), which is what the traced model should
    contain: pre/postprocessing stays outside, in Runtime's pipeline.
    """

    def __init__(self, policy: LeRobotPolicy) -> None:
        """Wrap a LeRobot policy for tracing."""
        super().__init__()
        self._policy = policy

    def forward(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        return self._policy.model.predict_action_chunk(observation)


class ExportableLeRobotPolicy(LeRobotPolicy):
    """A :class:`LeRobotPolicy` that advertises ONNX/OpenVINO export backends.

    Drop-in for :class:`~physicalai.policies.lerobot.policy.LeRobotPolicy` in
    export flows. It derives ``policy_name`` from the model config when
    reconstructed from a checkpoint, and supplies an I/O schema from the
    config's ``input_features``/``output_features``.
    """

    def __init__(
        self,
        policy_name: str | None = None,
        config: PreTrainedConfig | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Initialize the exportable wrapper, deriving policy_name from config.

        Args:
            policy_name: LeRobot registry name. When ``None`` and a ``config``
                is supplied, it is derived from ``config.type``.
            config: Pre-built LeRobot config object.
            **kwargs: Forwarded to :class:`LeRobotPolicy`.

        Raises:
            ValueError: If neither ``policy_name`` nor a config with a ``type``
                is provided.
        """
        if policy_name is None and config is not None:
            policy_name = getattr(config, "type", None)
        if policy_name is None:
            msg = "policy_name must be provided or derivable from config"
            raise ValueError(msg)
        super().__init__(policy_name=policy_name, config=config, **kwargs)

    @staticmethod
    def get_supported_export_backends() -> list[str | ExportBackend]:
        """Return the export backends this wrapper supports."""
        return _SUPPORTED_EXPORT_BACKENDS

    @property
    def inputs_schema(self) -> list[InferenceFeature] | None:
        """Describe the policy's expected model inputs for export tracing."""
        if not hasattr(self, "_config") or self._config is None:
            return None
        return build_inputs_schema(self)

    @property
    def outputs_schema(self) -> list[InferenceFeature] | None:
        """Describe the policy's model outputs for export metadata."""
        if not hasattr(self, "_config") or self._config is None:
            return None
        return build_outputs_schema(self)


# The shared export machinery derives artifact filenames and manifest policy
# names from ``__class__.__name__.lower()``. Keep them identical to the bare
# wrapper's (``lerobotpolicy.pt`` / ``lerobotpolicy.xml``) so re-exports match
# the artifacts the LeRobot training engine already produces. The manifest
# records that class name as the policy ``class_path``, so expose it as a
# module attribute too: inference load resolves ``class_path`` via
# ``physicalai.config.import_class``, which needs a real module attribute.
ExportableLeRobotPolicy.__name__ = "LerobotPolicy"
LerobotPolicy = ExportableLeRobotPolicy


def _build_trace_sample(policy: ExportableLeRobotPolicy) -> dict[str, torch.Tensor]:
    """Return a normalized observation sample to trace the model with.

    Built from the policy's I/O schema and pushed through its preprocessor,
    matching what Runtime's preprocessors would feed the model.

    Args:
        policy: The exportable policy.

    Returns:
        A dict of observation tensors suitable for tracing.

    Raises:
        RuntimeError: If the policy cannot produce an input sample.
    """
    sample = policy._get_default_export_input_sample()  # noqa: SLF001
    if sample is None:
        msg = "Cannot build an export input sample from the policy config"
        raise RuntimeError(msg)
    return sample


def _trace_onnx(wrapper: nn.Module, sample: dict[str, torch.Tensor], onnx_path: Path) -> None:
    """Trace the inference wrapper to ONNX, writing external data next to it."""
    with _match_default_device(wrapper):
        torch.onnx.export(
            wrapper,
            args=(),
            kwargs={"observation": sample},
            f=str(onnx_path),
            input_names=list(sample.keys()),
            output_names=["action"],
        )


def _export_graph_backend(policy: ExportableLeRobotPolicy, output_dir: Path, backend: ExportBackend) -> None:
    """Trace the inference graph and save it as ONNX or OpenVINO.

    Args:
        policy: The exportable policy.
        output_dir: Destination directory for the backend export.
        backend: ONNX or OpenVINO.

    Raises:
        ValueError: If ``backend`` is not an ONNX/OpenVINO graph export.
    """
    export_dir = Path(output_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    sample = _build_trace_sample(policy)
    wrapper = _LerobotInferenceWrapper(policy).eval()

    if backend == ExportBackend.ONNX:
        _trace_onnx(wrapper, sample, export_dir / "lerobotpolicy.onnx")
    elif backend == ExportBackend.OPENVINO:
        with tempfile.TemporaryDirectory() as tmp_dir:
            onnx_path = Path(tmp_dir) / "model.onnx"
            _trace_onnx(wrapper, sample, onnx_path)
            ov_model = openvino.convert_model(
                str(onnx_path),
                input=[openvino.Shape(tuple(t.shape)) for t in sample.values()],
            )
            openvino.save_model(
                ov_model,
                str(export_dir / "lerobotpolicy.xml"),
                compress_to_fp16=True,
            )
    else:
        msg = f"Unsupported graph export backend: {backend}"
        raise ValueError(msg)

    policy.create_manifest(
        export_dir,
        backend,
        runner=ComponentSpec.from_class(SinglePass),
        input_features=policy._to_component_specs(policy.inputs_schema or []),  # noqa: SLF001
        output_features=policy._to_component_specs(policy.outputs_schema or []),  # noqa: SLF001
    )


def export_lerobot_policy(
    policy: ExportableLeRobotPolicy,
    output_dir: str,
    *,
    backends: list[str | ExportBackend] | None = None,
) -> None:
    """Export a LeRobot policy to the requested backends.

    Args:
        policy: The exportable policy (load it via
            :meth:`ExportableLeRobotPolicy.load_from_checkpoint`).
        output_dir: Destination directory; one subdirectory per backend
            (``torch``, ``openvino``, ``onnx``) is created inside it.
        backends: Backends to export to. Defaults to torch and OpenVINO.

    Raises:
        TypeError: If ``policy`` is not an :class:`ExportableLeRobotPolicy`.
        ValueError: If an unsupported backend is requested.
    """
    if not isinstance(policy, ExportableLeRobotPolicy):
        msg = "export_lerobot_policy requires an ExportableLeRobotPolicy"
        raise TypeError(msg)

    selected = [ExportBackend(b) for b in (backends or [ExportBackend.TORCH, ExportBackend.OPENVINO])]
    policy.to("cpu")
    policy.eval()

    for backend in selected:
        backend_dir = Path(output_dir) / backend.value
        if backend == ExportBackend.TORCH:
            policy.export(backend_dir, backend=backend)
        elif backend in {ExportBackend.ONNX, ExportBackend.OPENVINO}:
            _export_graph_backend(policy, backend_dir, backend)
        else:
            msg = f"Unsupported export backend: {backend}"
            raise ValueError(msg)
