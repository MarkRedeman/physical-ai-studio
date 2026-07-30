from dataclasses import dataclass
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from api.dependencies import get_episode_thumbnail_service, get_project_service
from main import app
from schemas import Dataset, EpisodeInfo, Project


@dataclass
class _StubThumbnail:
    content: bytes
    etag: str
    last_modified: str


class _StubDatasetClient:
    def __init__(self, episodes: list[EpisodeInfo]) -> None:
        self._episodes = episodes

    def get_episode_infos(self) -> list[EpisodeInfo]:
        return self._episodes


class _StubProjectService:
    def __init__(self, project: Project) -> None:
        self._project = project

    async def get_project_by_id(self, project_id: UUID) -> Project:
        return self._project


class _StubEpisodeThumbnailService:
    def __init__(self, thumbnail: _StubThumbnail | None) -> None:
        self._thumbnail = thumbnail
        self.calls: list[dict[str, object]] = []

    def get_thumbnail(
        self,
        dataset_id: UUID,
        dataset: _StubDatasetClient,
        episode_index: int,
        camera: str | None = None,
        width: int = 320,
        height: int = 240,
    ) -> _StubThumbnail | None:
        self.calls.append(
            {
                "dataset_id": dataset_id,
                "dataset": dataset,
                "episode_index": episode_index,
                "camera": camera,
                "width": width,
                "height": height,
            }
        )
        return self._thumbnail


def _make_dataset() -> Dataset:
    return Dataset(
        id=uuid4(),
        name="Dataset 1",
        default_task="Task",
        project_id=uuid4(),
        environment_id=uuid4(),
    )


def _make_project(datasets: list[Dataset]) -> Project:
    return Project(
        id=uuid4(),
        name="Project",
        datasets=datasets,
    )


def test_project_thumbnail_returns_png(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = _make_dataset()
    project = _make_project([dataset])
    dataset_client = _StubDatasetClient(episodes=[EpisodeInfo(episode_index=7, tasks=["task"], length=20, fps=30)])
    thumbnail_service = _StubEpisodeThumbnailService(
        thumbnail=_StubThumbnail(
            content=b"png-bytes",
            etag='"project-etag"',
            last_modified="Wed, 06 Jan 2026 10:00:00 GMT",
        )
    )

    app.dependency_overrides[get_project_service] = lambda: _StubProjectService(project)
    app.dependency_overrides[get_episode_thumbnail_service] = lambda: thumbnail_service
    monkeypatch.setattr("api.project.get_internal_read_dataset", lambda _dataset: dataset_client)

    try:
        client = TestClient(app)
        response = client.get(f"/api/projects/{project.id}/thumbnail?width=640&height=360")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.headers["etag"] == '"project-etag"'
    assert response.content == b"png-bytes"
    assert thumbnail_service.calls == [
        {
            "dataset_id": dataset.id,
            "dataset": dataset_client,
            "episode_index": 7,
            "camera": None,
            "width": 640,
            "height": 360,
        }
    ]


def test_project_thumbnail_returns_304_for_matching_etag(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = _make_dataset()
    project = _make_project([dataset])
    dataset_client = _StubDatasetClient(episodes=[EpisodeInfo(episode_index=0, tasks=["task"], length=10, fps=30)])
    thumbnail_service = _StubEpisodeThumbnailService(
        thumbnail=_StubThumbnail(
            content=b"png-bytes",
            etag='"project-etag"',
            last_modified="Wed, 06 Jan 2026 10:00:00 GMT",
        )
    )

    app.dependency_overrides[get_project_service] = lambda: _StubProjectService(project)
    app.dependency_overrides[get_episode_thumbnail_service] = lambda: thumbnail_service
    monkeypatch.setattr("api.project.get_internal_read_dataset", lambda _dataset: dataset_client)

    try:
        client = TestClient(app)
        response = client.get(
            f"/api/projects/{project.id}/thumbnail",
            headers={"If-None-Match": '"project-etag"'},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 304


def test_project_thumbnail_returns_404_without_datasets() -> None:
    project = _make_project([])
    thumbnail_service = _StubEpisodeThumbnailService(
        thumbnail=_StubThumbnail(content=b"unused", etag='"unused"', last_modified="Wed, 06 Jan 2026 10:00:00 GMT")
    )

    app.dependency_overrides[get_project_service] = lambda: _StubProjectService(project)
    app.dependency_overrides[get_episode_thumbnail_service] = lambda: thumbnail_service

    try:
        client = TestClient(app)
        response = client.get(f"/api/projects/{project.id}/thumbnail")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 404


def test_project_thumbnail_returns_404_without_episodes(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = _make_dataset()
    project = _make_project([dataset])
    dataset_client = _StubDatasetClient(episodes=[])
    thumbnail_service = _StubEpisodeThumbnailService(
        thumbnail=_StubThumbnail(content=b"unused", etag='"unused"', last_modified="Wed, 06 Jan 2026 10:00:00 GMT")
    )

    app.dependency_overrides[get_project_service] = lambda: _StubProjectService(project)
    app.dependency_overrides[get_episode_thumbnail_service] = lambda: thumbnail_service
    monkeypatch.setattr("api.project.get_internal_read_dataset", lambda _dataset: dataset_client)

    try:
        client = TestClient(app)
        response = client.get(f"/api/projects/{project.id}/thumbnail")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 404
