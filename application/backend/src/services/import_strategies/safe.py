"""Safe import strategy decorator.

Wraps any ImportStrategy with security checks against malicious zip archives.
All validation uses zipfile metadata (infolist()) — no extraction is performed
during the security scan.
"""

import logging
import stat
import zipfile

from exceptions import ImportValidationError
from schemas import Model
from services.import_strategies.base import ImportStrategy

logger = logging.getLogger(__name__)

# Characters that should never appear in zip entry filenames.
_CONTROL_CHARS = set(range(0x00, 0x20)) - {ord("\n"), ord("\r")}
_DANGEROUS_UNICODE = {
    "\u202e",  # RTL override — can visually disguise filenames
}


class SafeImportStrategy(ImportStrategy):
    """Decorator that validates zip archive safety before delegating to an inner strategy.

    Checks performed (all metadata-only via ``infolist()``):
    - Encrypted entries (cannot be inspected)
    - Path traversal (``..`` components)
    - Symlinks (Unix symlink external attributes)
    - Filename attacks (null bytes, control characters, RTL override, excessive length)
    - Zip bombs (per-entry compression ratio, per-entry size, total size)
    - Excessive file count
    - Excessive directory depth
    """

    def __init__(
        self,
        inner: ImportStrategy,
        *,
        max_compression_ratio: float = 100.0,
        max_uncompressed_entry_size: int = 2 * 1024**3,  # 2 GB
        max_total_uncompressed_size: int = 5 * 1024**3,  # 5 GB
        max_file_count: int = 10_000,
        max_directory_depth: int = 20,
        max_filename_length: int = 1024,
        max_component_length: int = 255,
    ) -> None:
        self._inner = inner
        self._max_compression_ratio = max_compression_ratio
        self._max_uncompressed_entry_size = max_uncompressed_entry_size
        self._max_total_uncompressed_size = max_total_uncompressed_size
        self._max_file_count = max_file_count
        self._max_directory_depth = max_directory_depth
        self._max_filename_length = max_filename_length
        self._max_component_length = max_component_length

    async def import_model(
        self,
        zf: zipfile.ZipFile,
        all_names: list[str],
        name: str,
        project_id: str,
    ) -> Model:
        """Validate the archive for security threats, then delegate to the inner strategy."""
        self._validate_archive(zf)
        return await self._inner.import_model(zf, all_names, name, project_id)

    # ------------------------------------------------------------------
    # Archive validation
    # ------------------------------------------------------------------

    def _validate_archive(self, zf: zipfile.ZipFile) -> None:
        """Run all security checks against the zip's metadata.

        Iterates ``zf.infolist()`` exactly once, accumulating totals and
        checking each entry. Raises ``ImportValidationError`` on the first
        threat detected.
        """
        entries = zf.infolist()
        total_uncompressed: int = 0

        if len(entries) > self._max_file_count:
            raise ImportValidationError(
                f"Archive contains {len(entries):,} entries, exceeding the "
                f"limit of {self._max_file_count:,}. This may indicate a "
                f"malicious archive."
            )

        for info in entries:
            self._check_encryption(info)
            self._check_path_traversal(info)
            self._check_symlink(info)
            self._check_filename(info)
            self._check_directory_depth(info)
            self._check_entry_size(info)
            self._check_compression_ratio(info)
            total_uncompressed += info.file_size

        if total_uncompressed > self._max_total_uncompressed_size:
            total_gb = total_uncompressed / (1024**3)
            limit_gb = self._max_total_uncompressed_size / (1024**3)
            raise ImportValidationError(
                f"Archive total uncompressed size ({total_gb:.1f} GB) exceeds the {limit_gb:.1f} GB limit."
            )

        logger.info(
            "Archive security scan passed: %d entries, %.1f MB total uncompressed",
            len(entries),
            total_uncompressed / (1024**2),
        )

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    @staticmethod
    def _check_encryption(info: zipfile.ZipInfo) -> None:
        if info.flag_bits & 0x1:
            raise ImportValidationError(
                f"Archive entry '{info.filename}' is encrypted. "
                f"Encrypted archives cannot be safely inspected and are not supported."
            )

    @staticmethod
    def _check_path_traversal(info: zipfile.ZipInfo) -> None:
        if ".." in info.filename.split("/"):
            raise ImportValidationError(
                f"Archive entry '{info.filename}' contains a path traversal "
                f"component ('..'). This is a potential security threat."
            )

    @staticmethod
    def _check_symlink(info: zipfile.ZipInfo) -> None:
        unix_mode = info.external_attr >> 16
        if unix_mode and stat.S_ISLNK(unix_mode):
            raise ImportValidationError(
                f"Archive entry '{info.filename}' is a symbolic link. Symlinks in zip archives are not supported."
            )

    def _check_filename(self, info: zipfile.ZipInfo) -> None:
        filename = info.filename

        if len(filename) > self._max_filename_length:
            raise ImportValidationError(
                f"Archive entry filename is {len(filename)} characters long, "
                f"exceeding the {self._max_filename_length} character limit."
            )

        for component in filename.split("/"):
            if len(component) > self._max_component_length:
                raise ImportValidationError(
                    f"Archive entry '{filename}' has a path component "
                    f"exceeding {self._max_component_length} characters."
                )

        if any(ord(c) in _CONTROL_CHARS for c in filename):
            raise ImportValidationError(
                f"Archive entry '{filename!r}' contains control characters. This is a potential security threat."
            )

        if any(c in filename for c in _DANGEROUS_UNICODE):
            raise ImportValidationError(
                f"Archive entry '{filename!r}' contains dangerous Unicode "
                f"characters (e.g., RTL override). This is a potential security threat."
            )

    def _check_directory_depth(self, info: zipfile.ZipInfo) -> None:
        depth = len(info.filename.split("/"))
        if depth > self._max_directory_depth:
            raise ImportValidationError(
                f"Archive entry '{info.filename}' has a directory depth of "
                f"{depth}, exceeding the limit of {self._max_directory_depth}."
            )

    def _check_entry_size(self, info: zipfile.ZipInfo) -> None:
        if info.file_size > self._max_uncompressed_entry_size:
            size_mb = info.file_size / (1024**2)
            limit_mb = self._max_uncompressed_entry_size / (1024**2)
            raise ImportValidationError(
                f"Archive entry '{info.filename}' has an uncompressed size of "
                f"{size_mb:,.0f} MB, exceeding the {limit_mb:,.0f} MB limit."
            )

    def _check_compression_ratio(self, info: zipfile.ZipInfo) -> None:
        # Skip entries with zero compressed size (stored or empty files).
        if info.compress_size == 0:
            return
        ratio = info.file_size / info.compress_size
        if ratio > self._max_compression_ratio:
            raise ImportValidationError(
                f"Archive entry '{info.filename}' has a compression ratio of "
                f"{ratio:.0f}:1, exceeding the {self._max_compression_ratio:.0f}:1 "
                f"limit. This may indicate a zip bomb."
            )
