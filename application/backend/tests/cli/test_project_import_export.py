import json
from pathlib import Path
from uuid import uuid4
from zipfile import ZipFile

from click.testing import CliRunner
from sqlalchemy import select

from cli import cli
from db.engine import get_sync_db_session, sync_engine
from db.schema import (
    Base,
    DatasetDB,
    EnvironmentCameraDB,
    EnvironmentRobotDB,
    ModelDB,
    ProjectCameraDB,
    ProjectDB,
    ProjectEnvironmentDB,
    ProjectRobotDB,
    SnapshotDB,
)
from settings import get_settings


def _create_project() -> tuple[str, str, str, str]:
    settings = get_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(sync_engine)
    project_id, environment_id, dataset_id, snapshot_id, parent_model_id, model_id, robot_id, camera_id = (
        str(uuid4()) for _ in range(8)
    )
    for path in (
        settings.datasets_dir / dataset_id,
        settings.snapshot_dir / snapshot_id,
        settings.models_dir / parent_model_id,
        settings.models_dir / model_id,
    ):
        path.mkdir(parents=True)
        (path / "payload.txt").write_text(path.name)
    with get_sync_db_session() as session:
        session.add(ProjectDB(id=project_id, name="Archive source"))
        session.add(ProjectEnvironmentDB(id=environment_id, project_id=project_id, name="Bench"))
        session.add(ProjectRobotDB(id=robot_id, project_id=project_id, name="Arm", type="so101", payload={}))
        session.add(
            ProjectCameraDB(
                id=camera_id,
                project_id=project_id,
                name="Camera",
                driver="usb_camera",
                fingerprint="/dev/video0",
                hardware_name=None,
                payload={},
            )
        )
        session.add(EnvironmentRobotDB(environment_id=environment_id, robot_id=robot_id, tele_operator_type="none"))
        session.add(EnvironmentCameraDB(environment_id=environment_id, camera_id=camera_id))
        session.add(
            DatasetDB(
                id=dataset_id,
                name="Data",
                path=str(settings.datasets_dir / dataset_id),
                project_id=project_id,
                environment_id=environment_id,
                default_task="Pick",
            )
        )
        session.add(SnapshotDB(id=snapshot_id, path=str(settings.snapshot_dir / snapshot_id), dataset_id=dataset_id))
        session.add(
            ModelDB(
                id=parent_model_id,
                name="Parent policy",
                path=str(settings.models_dir / parent_model_id),
                policy="act",
                properties={},
                project_id=project_id,
                dataset_id=dataset_id,
                version=1,
            )
        )
        session.add(
            ModelDB(
                id=model_id,
                name="Policy",
                path=str(settings.models_dir / model_id),
                policy="act",
                properties={},
                project_id=project_id,
                dataset_id=dataset_id,
                snapshot_id=snapshot_id,
                parent_model_id=parent_model_id,
                version=1,
            )
        )
        session.commit()
    return project_id, dataset_id, snapshot_id, model_id


def test_project_archive_round_trip_includes_files_and_remaps_ids(tmp_path: Path) -> None:
    project_id, dataset_id, snapshot_id, model_id = _create_project()
    archive = tmp_path / "project.zip"
    runner = CliRunner()

    result = runner.invoke(
        cli, ["projects", "export", "--project-id", project_id, "--output", str(archive), "--include-snapshots"]
    )
    assert result.exit_code == 0, result.output
    result = runner.invoke(cli, ["projects", "import", "--source", str(archive)])
    assert result.exit_code == 0, result.output
    imported_project_id = result.output.rsplit(": ", 1)[1].strip()

    with get_sync_db_session() as session:
        project = session.get(ProjectDB, imported_project_id)
        dataset = session.scalar(select(DatasetDB).where(DatasetDB.project_id == imported_project_id))
        snapshot = session.scalar(select(SnapshotDB).where(SnapshotDB.dataset_id == dataset.id))
        models = list(session.scalars(select(ModelDB).where(ModelDB.project_id == imported_project_id)))
        model = next(model for model in models if model.name == "Policy")
        robot = session.scalar(select(ProjectRobotDB).where(ProjectRobotDB.project_id == imported_project_id))
        camera = session.scalar(select(ProjectCameraDB).where(ProjectCameraDB.project_id == imported_project_id))
        robot_link = session.scalar(
            select(EnvironmentRobotDB).where(EnvironmentRobotDB.environment_id == dataset.environment_id)
        )
        camera_link = session.scalar(
            select(EnvironmentCameraDB).where(EnvironmentCameraDB.environment_id == dataset.environment_id)
        )

    assert project.name == "Archive source"
    assert dataset.id != dataset_id
    assert snapshot.id != snapshot_id
    assert model.id != model_id
    assert model.dataset_id == dataset.id
    assert model.snapshot_id == snapshot.id
    assert model.parent_model_id in {imported_model.id for imported_model in models}
    assert robot_link.robot_id == str(robot.id)
    assert camera_link.camera_id == str(camera.id)
    assert (Path(dataset.path) / "payload.txt").read_text() == dataset_id
    assert (Path(snapshot.path) / "payload.txt").read_text() == snapshot_id
    assert (Path(model.path) / "payload.txt").read_text() == model_id


def test_project_import_rejects_unknown_relationship(tmp_path: Path) -> None:
    archive = tmp_path / "invalid.zip"
    manifest = {
        "format": "physicalai-studio-project",
        "version": 1,
        "project": {"id": str(uuid4()), "name": "Bad"},
        "robots": [],
        "cameras": [],
        "environments": [],
        "environment_robots": [],
        "environment_cameras": [],
        "datasets": [{"id": str(uuid4()), "name": "Data", "environment_id": str(uuid4()), "default_task": ""}],
        "snapshots": [],
        "models": [],
    }
    with ZipFile(archive, "w") as zip_file:
        zip_file.writestr("manifest.json", json.dumps(manifest))
        zip_file.writestr(f"datasets/{manifest['datasets'][0]['id']}/data.txt", "data")

    result = CliRunner().invoke(cli, ["projects", "import", "--source", str(archive)])

    assert result.exit_code == 1
    assert "unknown ID" in result.output
