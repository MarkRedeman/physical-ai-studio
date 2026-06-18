import json
from collections import defaultdict
from collections.abc import Mapping
from typing import Any, TypedDict

from physicalai.inference.manifest import ComponentSpec, Manifest


class ExportContext(TypedDict):
    backend: str
    artifact: str
    use: str


class FeatureContext(TypedDict):
    name: str
    ftype: str
    shape: str
    dtype: str


class IOGroupContext(TypedDict):
    backends: list[str]
    backend_label: str
    inputs: list[FeatureContext]
    outputs: list[FeatureContext]


_BACKEND_USE: Mapping[str, str] = {
    "torch": "Canonical checkpoint and Python inference",
    "openvino": "CPU, Intel GPU, and NPU inference",
    "onnx": "Runtime portability",
    "executorch": "Edge and mobile runtime experiments",
}


def export_context(manifests: Mapping[str, Manifest]) -> list[ExportContext]:
    """Build README rows for exported artifacts."""
    return [
        {
            "backend": backend,
            "artifact": artifact,
            "use": _BACKEND_USE.get(backend, "Backend-specific inference"),
        }
        for manifest in manifests.values()
        for backend, artifact in sorted(manifest.model.artifacts.items())
    ]


def io_group_context(manifests: Mapping[str, Manifest]) -> list[IOGroupContext]:
    """Build README I/O specs grouped by identical feature signatures."""
    grouped_specs: dict[str, tuple[list[FeatureContext], list[FeatureContext]]] = {}
    grouped_backends: dict[str, list[str]] = defaultdict(list)
    for backend, manifest in sorted(manifests.items()):
        inputs = _feature_context(manifest.model.input_features)
        outputs = _feature_context(manifest.model.output_features)
        if not inputs and not outputs:
            continue

        signature = json.dumps({"inputs": inputs, "outputs": outputs}, sort_keys=True)
        grouped_specs.setdefault(signature, (inputs, outputs))
        grouped_backends[signature].append(backend)

    return [
        {
            "backends": backends,
            "backend_label": ", ".join(f"`{backend}`" for backend in backends),
            "inputs": grouped_specs[signature][0],
            "outputs": grouped_specs[signature][1],
        }
        for signature, backends in grouped_backends.items()
    ]


def format_shape(shape: Any) -> str:
    """Format manifest feature shapes for README tables."""
    if not isinstance(shape, list):
        return "-"
    if len(shape) == 0:
        return "scalar"
    return "[{}]".format(", ".join(str(item) for item in shape))


def _feature_context(features: list[ComponentSpec]) -> list[FeatureContext]:
    return [
        {
            "name": str(feature.init_args["name"]),
            "ftype": str(feature.init_args.get("ftype", "-")),
            "shape": format_shape(feature.init_args.get("shape")),
            "dtype": str(feature.init_args.get("dtype", "-")),
        }
        for feature in features
        if isinstance(feature.init_args.get("name"), str) and feature.init_args["name"]
    ]
