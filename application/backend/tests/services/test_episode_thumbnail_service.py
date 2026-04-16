from pathlib import Path
from uuid import uuid4

from services.episode_thumbnail_service import EpisodeThumbnailService


class _StubDataset:
    def __init__(self, video_path: Path) -> None:
        self.video_path = video_path
        self.thumbnail_generation_calls = 0
        self.next_thumbnail = b"thumbnail-bytes"

    def get_video_keys(self) -> list[str]:
        return ["observation.images.front"]

    def get_video_path(self, episode: int, camera: str) -> Path:
        assert episode == 0
        assert camera == "front"
        return self.video_path

    def get_episode_thumbnail_png(
        self,
        episode_index: int,
        video_key: str,
        width: int = 320,
        height: int = 240,
    ) -> tuple[bytes, Path] | None:
        assert episode_index == 0
        assert video_key == "observation.images.front"
        assert width == 320
        assert height == 240
        self.thumbnail_generation_calls += 1
        return self.next_thumbnail, self.video_path


def test_get_thumbnail_writes_thumbnail_to_filesystem_cache(tmp_path) -> None:
    video_path = tmp_path / "episode.mp4"
    video_path.write_bytes(b"video-content")
    dataset = _StubDataset(video_path=video_path)
    service = EpisodeThumbnailService(cache_dir=tmp_path / "thumbnail-cache")

    thumbnail = service.get_thumbnail(dataset_id=uuid4(), dataset=dataset, episode_index=0)

    assert thumbnail is not None
    assert thumbnail.content == b"thumbnail-bytes"
    assert dataset.thumbnail_generation_calls == 1
    cached_pngs = list((tmp_path / "thumbnail-cache").rglob("*.png"))
    assert len(cached_pngs) == 1
    assert cached_pngs[0].read_bytes() == b"thumbnail-bytes"


def test_get_thumbnail_reads_thumbnail_from_filesystem_cache_on_second_call(tmp_path) -> None:
    video_path = tmp_path / "episode.mp4"
    video_path.write_bytes(b"video-content")
    dataset = _StubDataset(video_path=video_path)
    service = EpisodeThumbnailService(cache_dir=tmp_path / "thumbnail-cache")
    dataset_id = uuid4()

    first_thumbnail = service.get_thumbnail(dataset_id=dataset_id, dataset=dataset, episode_index=0)
    dataset.next_thumbnail = b"newly-generated-thumbnail"
    second_thumbnail = service.get_thumbnail(dataset_id=dataset_id, dataset=dataset, episode_index=0)

    assert first_thumbnail is not None
    assert second_thumbnail is not None
    assert first_thumbnail.content == b"thumbnail-bytes"
    assert second_thumbnail.content == b"thumbnail-bytes"
    assert dataset.thumbnail_generation_calls == 1


def test_get_thumbnail_invalidates_filesystem_cache_when_video_changes(tmp_path) -> None:
    video_path = tmp_path / "episode.mp4"
    video_path.write_bytes(b"video-content")
    dataset = _StubDataset(video_path=video_path)
    service = EpisodeThumbnailService(cache_dir=tmp_path / "thumbnail-cache")
    dataset_id = uuid4()

    first_thumbnail = service.get_thumbnail(dataset_id=dataset_id, dataset=dataset, episode_index=0)

    video_path.write_bytes(b"updated-video-content-with-different-size")
    dataset.next_thumbnail = b"thumbnail-after-video-change"
    second_thumbnail = service.get_thumbnail(dataset_id=dataset_id, dataset=dataset, episode_index=0)

    assert first_thumbnail is not None
    assert second_thumbnail is not None
    assert first_thumbnail.content == b"thumbnail-bytes"
    assert second_thumbnail.content == b"thumbnail-after-video-change"
    assert dataset.thumbnail_generation_calls == 2
