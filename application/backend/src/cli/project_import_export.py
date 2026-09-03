"""Developer-oriented project archive import and export commands.

Archives contain project-scoped configuration and the managed dataset, model,
and snapshot directories. They deliberately exclude jobs, logs, caches, global
settings, and credentials.
"""

import json
import shutil
import sys
import tempfile
from collections.abc import Iterable
from pathlib import Path
from uuid import UUID, uuid4
from zipfile import ZIP_DEFLATED, ZipFile

import click
from sqlalchemy import select

ARCHIVE_FORMAT = "physicalai-studio-project"
ARCHIVE_VERSION = 1


class ProjectArchiveError(Exception):
    """Raised when a project archive is incomplete or incompatible."""


@click.group("projects")
def projects() -> None:
    """Developer project archive commands."""


def _record(obj: object, fields: Iterable[str]) -> dict[str, object]:
    return {field: str(value) if isinstance(value := getattr(obj, field), UUID) else value for field in fields}


def _write_tree(archive: ZipFile, root: Path, prefix: str) -> None:
    if not root.is_dir():
        raise ProjectArchiveError(f"Managed directory is missing: {root}")
    for path in root.rglob("*"):
        if path.is_file():
            archive.write(path, Path(prefix) / path.relative_to(root))


def _manifest_for_project(
    session: object, project_id: str, include_snapshots: bool
) -> tuple[dict[str, object], list[tuple[Path, str]]]:
    from db.schema import (
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

    project = session.scalar(select(ProjectDB).where(ProjectDB.id == project_id))
    if project is None:
        raise ProjectArchiveError(f"Project not found: {project_id}")

    robots = list(session.scalars(select(ProjectRobotDB).where(ProjectRobotDB.project_id == project_id)))
    cameras = list(session.scalars(select(ProjectCameraDB).where(ProjectCameraDB.project_id == project_id)))
    environments = list(
        session.scalars(select(ProjectEnvironmentDB).where(ProjectEnvironmentDB.project_id == project_id))
    )
    environment_ids = [str(environment.id) for environment in environments]
    datasets = list(session.scalars(select(DatasetDB).where(DatasetDB.project_id == project_id)))
    models = list(session.scalars(select(ModelDB).where(ModelDB.project_id == project_id)))
    snapshots = list(
        session.scalars(select(SnapshotDB).where(SnapshotDB.dataset_id.in_([str(dataset.id) for dataset in datasets])))
        if include_snapshots and datasets
        else []
    )
    robot_links = list(
        session.scalars(select(EnvironmentRobotDB).where(EnvironmentRobotDB.environment_id.in_(environment_ids)))
        if environment_ids
        else []
    )
    camera_links = list(
        session.scalars(select(EnvironmentCameraDB).where(EnvironmentCameraDB.environment_id.in_(environment_ids)))
        if environment_ids
        else []
    )

    payloads: list[tuple[Path, str]] = [(Path(dataset.path), f"datasets/{dataset.id}") for dataset in datasets]
    payloads += [(Path(model.path), f"models/{model.id}") for model in models]
    payloads += [(Path(snapshot.path), f"snapshots/{snapshot.id}") for snapshot in snapshots]
    model_records = [
        _record(
            model,
            ("id", "name", "policy", "properties", "dataset_id", "snapshot_id", "parent_model_id", "version"),
        )
        for model in models
    ]
    if not include_snapshots:
        for record in model_records:
            record["snapshot_id"] = None
    manifest = {
        "format": ARCHIVE_FORMAT,
        "version": ARCHIVE_VERSION,
        "project": _record(project, ("id", "name")),
        "robots": [_record(robot, ("id", "name", "type", "payload")) for robot in robots],
        "cameras": [
            _record(camera, ("id", "name", "driver", "fingerprint", "hardware_name", "payload")) for camera in cameras
        ],
        "environments": [_record(environment, ("id", "name")) for environment in environments],
        "environment_robots": [
            _record(link, ("environment_id", "robot_id", "tele_operator_type", "tele_operator_robot_id"))
            for link in robot_links
        ],
        "environment_cameras": [_record(link, ("environment_id", "camera_id")) for link in camera_links],
        "datasets": [_record(dataset, ("id", "name", "environment_id", "default_task")) for dataset in datasets],
        "snapshots": [_record(snapshot, ("id", "dataset_id")) for snapshot in snapshots],
        "models": model_records,
    }
    return manifest, payloads


@projects.command("export")
@click.option("--project-id", required=True, type=click.UUID)
@click.option("--output", required=True, type=click.Path(dir_okay=False, path_type=Path))
@click.option("--include-snapshots", is_flag=True, help="Include training snapshot directories and metadata.")
def export_project(project_id: UUID, output: Path, include_snapshots: bool) -> None:
    """Export a project and its managed files to a portable ZIP archive."""
    from db.engine import get_sync_db_session

    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        with get_sync_db_session() as session:
            manifest, payloads = _manifest_for_project(session, str(project_id), include_snapshots)
        with ZipFile(output, "w", ZIP_DEFLATED) as archive:
            archive.writestr("manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
            for root, prefix in payloads:
                _write_tree(archive, root, prefix)
    except Exception as error:
        output.unlink(missing_ok=True)
        click.echo(f"Project export failed: {error}", err=True)
        sys.exit(1)
    click.echo(f"Project exported to: {output}")


def _require_records(manifest: dict[str, object], name: str) -> list[dict[str, object]]:
    records = manifest.get(name, [])
    if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
        raise ProjectArchiveError(f"Manifest field '{name}' must be a list of objects")
    return records


def _mapped(value: object, ids: dict[str, str], required: bool = False) -> str | None:
    if value is None and not required:
        return None
    try:
        return ids[str(value)]
    except KeyError as error:
        raise ProjectArchiveError(f"Manifest references unknown ID: {value}") from error


def _validate_manifest(manifest: object) -> dict[str, object]:
    if (
        not isinstance(manifest, dict)
        or manifest.get("format") != ARCHIVE_FORMAT
        or manifest.get("version") != ARCHIVE_VERSION
    ):
        raise ProjectArchiveError("Archive is not a supported Physical AI Studio project export")
    project = manifest.get("project")
    if not isinstance(project, dict) or not isinstance(project.get("name"), str):
        raise ProjectArchiveError("Manifest has no valid project")
    for name in (
        "robots",
        "cameras",
        "environments",
        "environment_robots",
        "environment_cameras",
        "datasets",
        "snapshots",
        "models",
    ):
        _require_records(manifest, name)
    return manifest


def _copy_payloads(stage: Path, settings: object, ids: dict[str, str], manifest: dict[str, object]) -> list[Path]:
    created: list[Path] = []
    try:
        for collection, destination_root in (
            ("datasets", settings.datasets_dir),
            ("models", settings.models_dir),
            ("snapshots", settings.snapshot_dir),
        ):
            for record in _require_records(manifest, collection):
                source = stage / collection / str(record["id"])
                destination = destination_root / ids[str(record["id"])]
                if not source.is_dir():
                    raise ProjectArchiveError(f"Archive is missing payload directory: {collection}/{record['id']}")
                if destination.exists():
                    raise ProjectArchiveError(f"Destination already exists: {destination}")
                shutil.copytree(source, destination)
                created.append(destination)
    except Exception:
        for path in created:
            shutil.rmtree(path, ignore_errors=True)
        raise
    return created


def _import_manifest(session: object, manifest: dict[str, object], settings: object, stage: Path) -> str:
    from db.schema import (
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

    collections = ("robots", "cameras", "environments", "datasets", "snapshots", "models")
    ids = {str(manifest["project"]["id"]): str(uuid4())}
    for collection in collections:
        ids.update({str(record["id"]): str(uuid4()) for record in _require_records(manifest, collection)})
    created = _copy_payloads(stage, settings, ids, manifest)
    try:
        project = manifest["project"]
        session.add(ProjectDB(id=ids[str(project["id"])], name=project["name"]))
        for record in _require_records(manifest, "robots"):
            session.add(
                ProjectRobotDB(
                    id=ids[str(record["id"])],
                    project_id=ids[str(project["id"])],
                    name=record["name"],
                    type=record["type"],
                    payload=record["payload"],
                )
            )
        for record in _require_records(manifest, "cameras"):
            session.add(
                ProjectCameraDB(
                    id=ids[str(record["id"])],
                    project_id=ids[str(project["id"])],
                    name=record["name"],
                    driver=record["driver"],
                    fingerprint=record["fingerprint"],
                    hardware_name=record["hardware_name"],
                    payload=record["payload"],
                )
            )
        for record in _require_records(manifest, "environments"):
            session.add(
                ProjectEnvironmentDB(id=ids[str(record["id"])], project_id=ids[str(project["id"])], name=record["name"])
            )
        for record in _require_records(manifest, "environment_robots"):
            session.add(
                EnvironmentRobotDB(
                    environment_id=_mapped(record.get("environment_id"), ids, True),
                    robot_id=_mapped(record.get("robot_id"), ids, True),
                    tele_operator_type=record["tele_operator_type"],
                    tele_operator_robot_id=_mapped(record.get("tele_operator_robot_id"), ids),
                )
            )
        for record in _require_records(manifest, "environment_cameras"):
            session.add(
                EnvironmentCameraDB(
                    environment_id=_mapped(record.get("environment_id"), ids, True),
                    camera_id=_mapped(record.get("camera_id"), ids, True),
                )
            )
        for record in _require_records(manifest, "datasets"):
            session.add(
                DatasetDB(
                    id=ids[str(record["id"])],
                    name=record["name"],
                    path=str(settings.datasets_dir / ids[str(record["id"])]),
                    project_id=ids[str(project["id"])],
                    environment_id=_mapped(record.get("environment_id"), ids, True),
                    default_task=record["default_task"],
                )
            )
        for record in _require_records(manifest, "snapshots"):
            session.add(
                SnapshotDB(
                    id=ids[str(record["id"])],
                    path=str(settings.snapshot_dir / ids[str(record["id"])]),
                    dataset_id=_mapped(record.get("dataset_id"), ids, True),
                )
            )
        for record in _require_records(manifest, "models"):
            session.add(
                ModelDB(
                    id=ids[str(record["id"])],
                    name=record["name"],
                    path=str(settings.models_dir / ids[str(record["id"])]),
                    policy=record["policy"],
                    properties=record["properties"],
                    project_id=ids[str(project["id"])],
                    dataset_id=_mapped(record.get("dataset_id"), ids),
                    snapshot_id=_mapped(record.get("snapshot_id"), ids),
                    parent_model_id=_mapped(record.get("parent_model_id"), ids),
                    version=record["version"],
                )
            )
        session.commit()
    except Exception:
        session.rollback()
        for path in created:
            shutil.rmtree(path, ignore_errors=True)
        raise
    return ids[str(project["id"])]


@projects.command("import")
@click.option("--source", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
def import_project(source: Path) -> None:
    """Import a project archive, creating a project with new resource IDs."""
    from physicalai.data.archive_safety import SafeZipArchive

    from db.engine import get_sync_db_session
    from settings import get_settings

    settings = get_settings()
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="project-import-", dir=settings.cache_dir))
    try:
        archive = SafeZipArchive(source, max_uncompressed_bytes=settings.data_import_max_uncompressed_bytes)
        archive.validate()
        manifest = _validate_manifest(archive.read_json("manifest.json"))
        archive.extract_to(stage, min_free_bytes=settings.data_import_min_free_bytes)
        with get_sync_db_session() as session:
            project_id = _import_manifest(session, manifest, settings, stage)
    except Exception as error:
        click.echo(f"Project import failed: {error}", err=True)
        sys.exit(1)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    click.echo(f"Project imported successfully. Project ID: {project_id}")
