import json
import zipfile
from pathlib import Path
from unittest.mock import MagicMock

from schemas.dataset_import_job import ImportValidationSeverity
from services.archive_safety import SafeZipArchive
from services.dataset_import.adapters.lerobot_v2 import LeRobotV2Adapter


def _write_zip(path: Path, files: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)


def test_lerobot_v2_detect_returns_true_for_legacy_layout(tmp_path: Path) -> None:
    adapter = LeRobotV2Adapter()
    archive_path = tmp_path / "legacy-v2.zip"
    _write_zip(
        archive_path,
        {
            "meta/info.json": b'{"codebase_version":"v2.1"}',
            "meta/tasks.jsonl": b'{"task_index":0,"task":"pick"}\n',
            "meta/episodes.jsonl": b'{"episode_index":0,"tasks":["pick"],"length":1}\n',
            "data/chunk-000/episode_000000.parquet": b"PAR1",  # marker bytes are sufficient for detect()
        },
    )

    safe_archive = SafeZipArchive(archive_path, max_uncompressed_bytes=5 * 1024 * 1024 * 1024)
    assert adapter.detect(safe_archive) is True


def test_lerobot_v2_detect_returns_false_without_legacy_markers(tmp_path: Path) -> None:
    adapter = LeRobotV2Adapter()
    archive_path = tmp_path / "not-v2.zip"
    _write_zip(
        archive_path,
        {
            "meta/info.json": b'{"codebase_version":"v3.0"}',
            "meta/tasks.parquet": b"PAR1",
            "data/chunk-000/file-000.parquet": b"PAR1",
        },
    )

    safe_archive = SafeZipArchive(archive_path, max_uncompressed_bytes=5 * 1024 * 1024 * 1024)
    assert adapter.detect(safe_archive) is False


def test_lerobot_v2_parse_manifest_reads_stats_from_nested_root_zip(tmp_path: Path) -> None:
    adapter = LeRobotV2Adapter()
    archive_path = tmp_path / "legacy-v2-nested.zip"
    episodes = [
        {"episode_index": 0, "tasks": ["pick"], "length": 10},
        {"episode_index": 1, "tasks": ["pick"], "length": 15},
    ]
    _write_zip(
        archive_path,
        {
            "dataset/meta/info.json": json.dumps({"codebase_version": "v2.1", "fps": 30}).encode("utf-8"),
            "dataset/meta/tasks.jsonl": b'{"task_index":0,"task":"pick"}\n',
            "dataset/meta/episodes.jsonl": "\n".join(json.dumps(item) for item in episodes).encode("utf-8") + b"\n",
            "dataset/data/chunk-000/episode_000000.parquet": b"PAR1",
        },
    )

    safe_archive = SafeZipArchive(archive_path, max_uncompressed_bytes=5 * 1024 * 1024 * 1024)
    manifest, _report = adapter.build_draft(safe_archive, payload=MagicMock())

    assert manifest.capture.fps == 30
    assert manifest.capture.episode_count == 2
    assert manifest.capture.frame_count == 25


def test_lerobot_v2_parse_manifest_schema_empty_when_no_features(tmp_path: Path) -> None:
    """When info.json has no features the recording schema is empty."""
    adapter = LeRobotV2Adapter()
    archive_path = tmp_path / "v2-no-features.zip"
    episodes = [{"episode_index": 0, "tasks": ["pick"], "length": 5}]
    _write_zip(
        archive_path,
        {
            "meta/info.json": json.dumps({"codebase_version": "v2.1", "fps": 10}).encode("utf-8"),
            "meta/tasks.jsonl": b'{"task_index":0,"task":"pick"}\n',
            "meta/episodes.jsonl": "\n".join(json.dumps(item) for item in episodes).encode("utf-8") + b"\n",
            "data/chunk-000/episode_000000.parquet": b"PAR1",
        },
    )

    safe_archive = SafeZipArchive(archive_path, max_uncompressed_bytes=5 * 1024 * 1024 * 1024)
    manifest, _report = adapter.build_draft(safe_archive, payload=MagicMock())

    assert manifest.schema_.cameras == []
    assert manifest.schema_.robots == []


def test_lerobot_v2_parse_manifest_schema_cameras_and_joints(tmp_path: Path) -> None:
    """V2 adapter extracts camera entries and robot joints from info.json features."""
    adapter = LeRobotV2Adapter()
    archive_path = tmp_path / "v2-with-schema.zip"
    episodes = [{"episode_index": 0, "tasks": ["pick"], "length": 3}]

    info = {
        "codebase_version": "v2.1",
        "fps": 25,
        "robot_type": "panda",
        "features": {
            "observation.images.left_cam": {
                "dtype": "video",
                "shape": [480, 640, 3],
                "info": {"video": {"fps": 25, "width": 640, "height": 480}},
            },
            "action": {
                "dtype": "float32",
                "shape": [4],
                "names": ["joint1.pos", "joint2.pos", "joint3.pos", "gripper.pos"],
            },
        },
    }

    _write_zip(
        archive_path,
        {
            "meta/info.json": json.dumps(info).encode("utf-8"),
            "meta/tasks.jsonl": b'{"task_index":0,"task":"pick"}\n',
            "meta/episodes.jsonl": "\n".join(json.dumps(item) for item in episodes).encode("utf-8") + b"\n",
            "data/chunk-000/episode_000000.parquet": b"PAR1",
        },
    )

    safe_archive = SafeZipArchive(archive_path, max_uncompressed_bytes=5 * 1024 * 1024 * 1024)
    manifest, _report = adapter.build_draft(safe_archive, payload=MagicMock())

    schema = manifest.schema_

    assert len(schema.cameras) == 1
    cam = schema.cameras[0]
    assert cam.name == "left_cam"
    assert cam.width == 640
    assert cam.height == 480
    assert cam.fps == 25

    assert len(schema.robots) == 1
    robot = schema.robots[0]
    assert robot.name == "panda"
    assert robot.type == "panda"
    assert robot.joints == ["joint1", "joint2", "joint3", "gripper"]


def test_lerobot_v2_parse_manifest_prefers_uploaded_archive_name_for_suggested_name(tmp_path: Path) -> None:
    adapter = LeRobotV2Adapter()
    archive_path = tmp_path / "opaque-staging-id.zip"
    _write_zip(
        archive_path,
        {
            "meta/info.json": b'{"fps":30}',
            "meta/tasks.jsonl": b'{"task_index":0,"task":"pick"}\n',
            "meta/episodes.jsonl": b'{"episode_index":0,"tasks":["pick"],"length":1}\n',
            "data/chunk-000/episode_000000.parquet": b"PAR1",
        },
    )

    payload = MagicMock()
    payload.uploaded_archive_name = "my-cool-dataset.zip"

    safe_archive = SafeZipArchive(archive_path, max_uncompressed_bytes=5 * 1024 * 1024 * 1024)
    manifest, _report = adapter.build_draft(safe_archive, payload=payload)

    assert manifest.suggested_name == "my-cool-dataset"


def test_lerobot_v2_build_draft_reports_error_when_episodes_metadata_missing(tmp_path: Path) -> None:
    adapter = LeRobotV2Adapter()
    archive_path = tmp_path / "v2-missing-episodes.zip"
    _write_zip(
        archive_path,
        {
            "meta/info.json": b'{"codebase_version":"v2.1","fps":30}',
            "meta/tasks.jsonl": b'{"task_index":0,"task":"pick"}\n',
        },
    )

    safe_archive = SafeZipArchive(archive_path, max_uncompressed_bytes=5 * 1024 * 1024 * 1024)
    _manifest, report = adapter.build_draft(safe_archive, payload=MagicMock())

    assert any(msg.severity == ImportValidationSeverity.ERROR for msg in report.messages)
    assert any("episodes.jsonl" in msg.message for msg in report.messages)


def test_lerobot_v2_build_draft_reports_warnings_for_missing_schema(tmp_path: Path) -> None:
    adapter = LeRobotV2Adapter()
    archive_path = tmp_path / "v2-no-features-report.zip"
    _write_zip(
        archive_path,
        {
            "meta/info.json": json.dumps({"codebase_version": "v2.1", "fps": 10}).encode("utf-8"),
            "meta/tasks.jsonl": b'{"task_index":0,"task":"pick"}\n',
            "meta/episodes.jsonl": b'{"episode_index":0,"tasks":["pick"],"length":1}\n',
            "data/chunk-000/episode_000000.parquet": b"PAR1",
        },
    )

    safe_archive = SafeZipArchive(archive_path, max_uncompressed_bytes=5 * 1024 * 1024 * 1024)
    _manifest, report = adapter.build_draft(safe_archive, payload=MagicMock())

    warning_messages = [msg.message for msg in report.messages if msg.severity == ImportValidationSeverity.WARNING]
    assert any("No camera streams" in message for message in warning_messages)
    assert any("No robot schema" in message for message in warning_messages)
