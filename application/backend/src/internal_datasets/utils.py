from pathlib import Path

from internal_datasets.dataset_client import DatasetClient
from schemas import Dataset


def get_internal_dataset(dataset: Dataset) -> DatasetClient:
    """Load dataset from dataset data class."""
    from internal_datasets.lerobot.lerobot_dataset import InternalLeRobotDataset

    return InternalLeRobotDataset(Path(dataset.path))
