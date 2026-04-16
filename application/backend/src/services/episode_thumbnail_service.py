import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import format_datetime
from pathlib import Path
from uuid import UUID, uuid4

from internal_datasets.dataset_client import DatasetClient
from settings import get_settings


@dataclass(frozen=True)
class EpisodeThumbnail:
    content: bytes
    etag: str
    last_modified: str


class EpisodeThumbnailService:
    def __init__(self, cache_dir: Path | None = None) -> None:
        settings = get_settings()
        self._cache_dir = cache_dir or settings.cache_dir / "episode_thumbnails"

    def get_thumbnail(
        self,
        dataset_id: UUID,
        dataset: DatasetClient,
        episode_index: int,
        camera: str | None = None,
        width: int = 320,
        height: int = 240,
    ) -> EpisodeThumbnail | None:
        video_key = self._resolve_video_key(dataset, camera)
        if video_key is None:
            return None

        video_path = self._resolve_video_path(dataset, episode_index, video_key)
        if video_path is None:
            return None

        try:
            video_stat = video_path.stat()
        except FileNotFoundError:
            return None
        etag_payload = (
            f"{dataset_id}:{episode_index}:{video_key}:{width}x{height}:{video_stat.st_mtime_ns}:{video_stat.st_size}"
        )
        etag = hashlib.sha256(etag_payload.encode()).hexdigest()
        last_modified = format_datetime(datetime.fromtimestamp(video_stat.st_mtime, tz=UTC), usegmt=True)

        cached_content = self._read_cached_thumbnail(etag)
        if cached_content is not None:
            return EpisodeThumbnail(content=cached_content, etag=f'"{etag}"', last_modified=last_modified)

        result = dataset.get_episode_thumbnail_png(episode_index, video_key, width, height)
        if result is None:
            return None

        thumbnail_bytes, _ = result
        self._write_cached_thumbnail(etag, thumbnail_bytes)
        return EpisodeThumbnail(content=thumbnail_bytes, etag=f'"{etag}"', last_modified=last_modified)

    def _resolve_video_key(self, dataset: DatasetClient, camera: str | None) -> str | None:
        video_keys = dataset.get_video_keys()
        if len(video_keys) == 0:
            return None

        if camera is None:
            return video_keys[0]

        if camera in video_keys:
            return camera

        prefixed_camera = f"observation.images.{camera}"
        if prefixed_camera in video_keys:
            return prefixed_camera

        return None

    def _resolve_video_path(self, dataset: DatasetClient, episode_index: int, video_key: str) -> Path | None:
        if video_key.startswith("observation.images."):
            camera = video_key.removeprefix("observation.images.")
        else:
            camera = video_key

        try:
            return dataset.get_video_path(episode_index, camera)
        except (FileNotFoundError, KeyError, ValueError):
            return None

    def _cache_path_for_etag(self, etag: str) -> Path:
        return self._cache_dir / etag[:2] / f"{etag}.png"

    def _read_cached_thumbnail(self, etag: str) -> bytes | None:
        cache_path = self._cache_path_for_etag(etag)
        try:
            return cache_path.read_bytes()
        except FileNotFoundError:
            return None

    def _write_cached_thumbnail(self, etag: str, thumbnail: bytes) -> None:
        cache_path = self._cache_path_for_etag(etag)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = cache_path.with_name(f"{cache_path.name}.{uuid4().hex}.tmp")
        temp_path.write_bytes(thumbnail)
        temp_path.replace(cache_path)
