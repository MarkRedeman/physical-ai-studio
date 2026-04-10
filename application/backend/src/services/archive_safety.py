import json
import os
import shutil
from collections.abc import Iterator
from pathlib import Path
from zipfile import BadZipFile, ZipFile, ZipInfo

from loguru import logger

from exceptions import InsufficientDiskSpaceError, InvalidArchiveError, ZipBombDetectedError

DEFAULT_MAX_FILE_COUNT = 200_000
ALLOW_NESTED_ZIP = False


def _collect_total_uncompressed(members: list[ZipInfo]) -> int:
    total_uncompressed = 0
    for member in members:
        total_uncompressed += member.file_size
    return total_uncompressed


def _normalize_zip_member_name(name: str) -> str:
    normalized = name.replace("\\", "/").strip("/")
    if normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


class SafeZipArchive:
    """Security-aware ZIP access wrapper.

    This class validates archive entries once and provides safe helper methods
    for member lookup and extraction so callers don't have to manually invoke
    low-level safety functions.
    """

    def __init__(
        self,
        archive_path: str | Path,
        *,
        max_uncompressed_bytes: int,
        max_file_count: int | None = None,
    ) -> None:
        self.path = Path(archive_path)
        self.max_uncompressed_bytes = max_uncompressed_bytes
        self.max_file_count = max_file_count
        self._validated_members: list[ZipInfo] | None = None

    def validate(self) -> None:
        self._get_validated_members()

    def estimated_uncompressed_size(self) -> int:
        members = self._get_validated_members()
        return _collect_total_uncompressed(members)

    def iter_normalized_names(self) -> Iterator[str]:
        for member in self._get_validated_members():
            yield _normalize_zip_member_name(member.filename)

    def resolve_member_name(self, target_name: str) -> str | None:
        normalized_target = _normalize_zip_member_name(target_name)
        for member in self._get_validated_members():
            normalized = _normalize_zip_member_name(member.filename)
            if normalized == normalized_target or normalized.endswith(f"/{normalized_target}"):
                return member.filename
        return None

    def read_json(self, target_name: str) -> dict | None:
        member_name = self.resolve_member_name(target_name)
        if member_name is None:
            return None

        try:
            with ZipFile(self.path, mode="r") as archive, archive.open(member_name) as file_obj:
                return json.loads(file_obj.read().decode("utf-8"))
        except (BadZipFile, OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise InvalidArchiveError(f"Unable to read JSON '{target_name}' from archive") from error

    def read_jsonl(self, target_name: str) -> list[dict]:
        member_name = self.resolve_member_name(target_name)
        if member_name is None:
            return []

        try:
            with ZipFile(self.path, mode="r") as archive, archive.open(member_name) as file_obj:
                lines = file_obj.read().decode("utf-8").splitlines()
        except (BadZipFile, OSError, UnicodeDecodeError) as error:
            raise InvalidArchiveError(f"Unable to read JSONL '{target_name}' from archive") from error

        rows: list[dict] = []
        for line in lines:
            normalized = line.strip()
            if not normalized:
                continue
            rows.append(json.loads(normalized))
        return rows

    def read_bytes(self, target_name: str) -> bytes | None:
        member_name = self.resolve_member_name(target_name)
        if member_name is None:
            return None

        try:
            with ZipFile(self.path, mode="r") as archive, archive.open(member_name) as file_obj:
                return file_obj.read()
        except (BadZipFile, OSError) as error:
            raise InvalidArchiveError(f"Unable to read '{target_name}' from archive") from error

    def extract_to(
        self,
        destination_dir: str | Path,
        *,
        min_free_bytes: int = 0,
    ) -> int:
        destination_root = Path(destination_dir)
        if min_free_bytes > 0:
            check_disk_headroom(destination_root, self.estimated_uncompressed_size(), min_free_bytes)

        members = self._get_validated_members()
        extracted_count = 0
        resolved_destination = destination_root.resolve()

        try:
            with ZipFile(self.path, mode="r") as archive:
                for member in members:
                    member_name = _normalize_zip_member_name(member.filename)
                    target_path = (resolved_destination / member_name).resolve()
                    if resolved_destination not in target_path.parents and target_path != resolved_destination:
                        raise ZipBombDetectedError(f"Archive contains unsafe entry path '{member.filename}'")

                    archive.extract(member, resolved_destination)
                    if not member.is_dir():
                        extracted_count += 1
        except BadZipFile as error:
            raise InvalidArchiveError("Uploaded file is not a valid ZIP archive") from error

        return extracted_count

    def _get_validated_members(self) -> list[ZipInfo]:
        if self._validated_members is None:
            try:
                with ZipFile(self.path, mode="r") as archive:
                    members = archive.infolist()
            except BadZipFile as error:
                raise InvalidArchiveError("Uploaded file is not a valid ZIP archive") from error

            validate_zip_entries(
                members,
                max_file_count=self.max_file_count,
                max_uncompressed_bytes=self.max_uncompressed_bytes,
            )
            self._validated_members = members

        return self._validated_members


def _is_symlink(member_external_attr: int) -> bool:
    # Upper 16 bits contain Unix mode when created on Unix systems
    mode = member_external_attr >> 16
    return (mode & 0o170000) == 0o120000


def validate_zip_entries(
    members: list[ZipInfo],
    *,
    max_file_count: int | None,
    max_uncompressed_bytes: int,
) -> int:
    """Validate ZIP entries for safety constraints and return total uncompressed bytes."""
    if max_file_count is None:
        max_file_count = DEFAULT_MAX_FILE_COUNT

    if len(members) > max_file_count:
        raise ZipBombDetectedError(f"Archive contains too many entries ({len(members)} > {max_file_count})")

    total_uncompressed = _collect_total_uncompressed(members)
    if total_uncompressed > max_uncompressed_bytes:
        raise ZipBombDetectedError(
            f"Archive uncompressed size exceeds allowed limit ({total_uncompressed} > {max_uncompressed_bytes} bytes)"
        )

    for member in members:
        name = _normalize_zip_member_name(member.filename)

        if _is_symlink(member.external_attr):
            raise ZipBombDetectedError(f"Archive contains symlink entry '{name}', which is not allowed")

        # Reject path traversal and absolute paths
        normalized_path = Path(name)
        if normalized_path.is_absolute() or ".." in normalized_path.parts:
            raise ZipBombDetectedError(f"Archive contains unsafe entry path '{member.filename}'")

        if not ALLOW_NESTED_ZIP and os.path.splitext(name)[1].lower() == ".zip":
            raise ZipBombDetectedError(f"Archive contains nested zip entry '{member.filename}', which is not allowed")

    return total_uncompressed


def check_disk_headroom(directory: Path, required_bytes: int, min_free_bytes: int) -> None:
    """Ensure *directory*'s filesystem has enough headroom for the pending operation.

    :param directory: Directory (or any path on the target filesystem) to
        inspect.  The directory is created if it does not yet exist so that
        ``shutil.disk_usage`` has a concrete path to query.
    :param required_bytes: Bytes the caller expects to write (e.g. upload size
        or estimated uncompressed extraction size).
    :param min_free_bytes: Minimum headroom that must remain *after* writing
        *required_bytes*.
    :raises InsufficientDiskSpaceError: When free space would fall below the
        minimum headroom.
    """
    directory.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(directory)
    needed = required_bytes + min_free_bytes
    if usage.free < needed:
        raise InsufficientDiskSpaceError(
            f"Insufficient disk space on '{directory}': "
            f"{usage.free} bytes free, need {needed} bytes "
            f"({required_bytes} for data + {min_free_bytes} headroom)"
        )


def cleanup_staged_archive(uploaded_archive_path: str | Path | None) -> None:
    """Best-effort staged archive cleanup used by API/service/worker paths."""
    if uploaded_archive_path is None:
        return

    try:
        Path(uploaded_archive_path).unlink(missing_ok=True)
    except Exception as error:
        logger.warning("Failed to clean staged dataset archive '{}': {}", uploaded_archive_path, error)


def flatten_single_root_directory(destination_dir: str | Path) -> None:
    """Flatten extracted archive when it contains exactly one root directory.

    Supports archives where dataset files are nested under a single top-level
    folder by moving that folder's contents to *destination_dir*.
    """
    root = Path(destination_dir)
    entries = list(root.iterdir())
    if len(entries) != 1 or not entries[0].is_dir():
        return

    nested_root = entries[0]
    nested_entries = list(nested_root.iterdir())
    for child in nested_entries:
        shutil.move(str(child), str(root / child.name))
    nested_root.rmdir()
