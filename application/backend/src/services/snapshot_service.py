import time
from pathlib import Path
from shutil import copytree
from uuid import UUID

from db.engine import get_async_db_session_ctx
from exceptions import ResourceAlreadyExistsError, ResourceNotFoundError, ResourceType
from repositories.snapshot_repo import SnapshotRepository
from schemas import Dataset, Snapshot


class SnapshotService:
    """Allow for snapshotting of dataset to specific folder."""

    @staticmethod
    async def get_snapshot_by_id(snapshot_id: UUID) -> Snapshot:
        async with get_async_db_session_ctx() as session:
            repo = SnapshotRepository(session)
            snapshot = await repo.get_by_id(snapshot_id)
            if snapshot is None:
                raise ResourceNotFoundError(ResourceType.SNAPSHOT, str(snapshot_id))
            return snapshot

    @staticmethod
    async def create_snapshot_for_dataset(dataset: Dataset, destination: Path) -> Snapshot:
        if destination.exists():
            raise ResourceAlreadyExistsError("Snapshot", f"Destination directory already exists:{destination}")

        snapshot = Snapshot(
            dataset_id=dataset.id,
            path=str(destination),
        )
        SnapshotService._copy_dataset(Path(dataset.path), destination)

        async with get_async_db_session_ctx() as session:
            repo = SnapshotRepository(session)
            await repo.save(snapshot)

        return snapshot

    @staticmethod
    def _copy_dataset(source: Path, destination: Path) -> None:
        copytree(source, destination)

    @staticmethod
    def generate_snapshot_folder_name() -> str:
        return time.strftime("snapshot_%Y-%m-%d_%H-%M-%S")
