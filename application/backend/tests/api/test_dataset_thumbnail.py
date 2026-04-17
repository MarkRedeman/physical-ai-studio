from uuid import uuid4

from fastapi import status
from fastapi.testclient import TestClient

from api.dependencies import get_dataset_service, get_episode_thumbnail_service
from main import app
from schemas import Dataset
from services.episode_thumbnail_service import EpisodeThumbnail


class _StubDatasetService:
    def __init__(self, dataset: Dataset) -> None:
        self._dataset = dataset

    async def get_dataset_by_id(self, _dataset_id):
        return self._dataset


class _StubThumbnailService:
    def __init__(self, thumbnail: EpisodeThumbnail | None) -> None:
        self.thumbnail = thumbnail
        self.calls = 0

    def get_thumbnail(self, **_kwargs):
        self.calls += 1
        return self.thumbnail


def _make_lerobot_dataset() -> Dataset:
    return Dataset(
        id=uuid4(),
        name="Demo dataset",
        default_task="Pick and place",
        path="/datasets/test",
        project_id=uuid4(),
        environment_id=uuid4(),
    )


class _FakeInternalLeRobotDataset:
    pass


def test_get_episode_thumbnail_returns_499_when_client_already_disconnected(monkeypatch) -> None:
    dataset = _make_lerobot_dataset()
    dataset_service = _StubDatasetService(dataset)
    thumbnail_service = _StubThumbnailService(
        EpisodeThumbnail(content=b"thumb", etag='"etag"', last_modified="Thu, 01 Jan 1970 00:00:00 GMT")
    )

    app.dependency_overrides[get_dataset_service] = lambda: dataset_service
    app.dependency_overrides[get_episode_thumbnail_service] = lambda: thumbnail_service
    monkeypatch.setattr("api.dataset.InternalLeRobotDataset", _FakeInternalLeRobotDataset)
    monkeypatch.setattr("api.dataset.get_internal_dataset", lambda _dataset: _FakeInternalLeRobotDataset())

    async def disconnected(self):
        return True

    monkeypatch.setattr("starlette.requests.Request.is_disconnected", disconnected)

    try:
        client = TestClient(app)
        response = client.get(f"/api/dataset/{dataset.id}/episodes/0/thumbnail")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 499
    assert thumbnail_service.calls == 0


def test_get_episode_thumbnail_returns_304_when_cached_headers_match(monkeypatch) -> None:
    dataset = _make_lerobot_dataset()
    dataset_service = _StubDatasetService(dataset)
    thumbnail_service = _StubThumbnailService(
        EpisodeThumbnail(content=b"thumb", etag='"etag"', last_modified="Thu, 01 Jan 1970 00:00:00 GMT")
    )

    app.dependency_overrides[get_dataset_service] = lambda: dataset_service
    app.dependency_overrides[get_episode_thumbnail_service] = lambda: thumbnail_service
    monkeypatch.setattr("api.dataset.InternalLeRobotDataset", _FakeInternalLeRobotDataset)
    monkeypatch.setattr("api.dataset.get_internal_dataset", lambda _dataset: _FakeInternalLeRobotDataset())

    async def connected(self):
        return False

    monkeypatch.setattr("starlette.requests.Request.is_disconnected", connected)

    try:
        client = TestClient(app)
        response = client.get(
            f"/api/dataset/{dataset.id}/episodes/0/thumbnail",
            headers={"If-None-Match": '"etag"'},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == status.HTTP_304_NOT_MODIFIED
    assert thumbnail_service.calls == 1
