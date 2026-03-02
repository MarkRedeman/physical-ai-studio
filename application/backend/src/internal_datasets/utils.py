from pathlib import Path

from loguru import logger

from internal_datasets.dataset_client import DatasetClient
from internal_datasets.lerobot.lerobot_dataset import InternalLeRobotDataset
from internal_datasets.raw_video.raw_video_dataset_client import RawVideoDatasetClient
from schemas import Dataset


def get_internal_dataset(dataset: Dataset) -> DatasetClient:
    """Load dataset from dataset data class.

    Auto-detects the dataset format based on sentinel files:

    - ``manifest.json`` present -> raw-video format -> :class:`RawVideoDatasetClient`
    - ``meta/info.json`` present -> LeRobot format -> :class:`InternalLeRobotDataset`
    - Neither present (new dataset) -> defaults to :class:`RawVideoDatasetClient`
    """
    dataset_path = Path(dataset.path)

    if (dataset_path / "manifest.json").is_file():
        logger.debug("Detected raw-video dataset at {}", dataset_path)
        return RawVideoDatasetClient(dataset_path)

    if (dataset_path / "meta" / "info.json").is_file():
        logger.debug("Detected LeRobot dataset at {}", dataset_path)
        return InternalLeRobotDataset(dataset_path)

    # Path does not exist yet — default to the raw-video format for new datasets.
    logger.debug("No existing dataset at {}; defaulting to raw-video format", dataset_path)
    return RawVideoDatasetClient(dataset_path)
