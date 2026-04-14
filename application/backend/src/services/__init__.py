from importlib import import_module
from typing import Any

__all__ = [
    "DatasetDownloadService",
    "DatasetService",
    "EpisodeThumbnailService",
    "ModelDownloadService",
    "ModelService",
    "ProjectCameraService",
    "ProjectService",
]

_EXPORTS = {
    "DatasetDownloadService": "services.dataset_download_service",
    "DatasetService": "services.dataset_service",
    "EpisodeThumbnailService": "services.episode_thumbnail_service",
    "ModelDownloadService": "services.model_download_service",
    "ModelService": "services.model_service",
    "ProjectCameraService": "services.project_camera_service",
    "ProjectService": "services.project_service",
}


def __getattr__(name: str) -> Any:
    module_path = _EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module 'services' has no attribute {name!r}")

    module = import_module(module_path)
    value = getattr(module, name)
    globals()[name] = value
    return value
