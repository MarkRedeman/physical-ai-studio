from __future__ import annotations

from typing import TYPE_CHECKING

from .base import DatasetImportAdapter

if TYPE_CHECKING:
    from services.archive_safety import SafeZipArchive

REGISTERED_DATASET_IMPORT_ADAPTERS: tuple[DatasetImportAdapter, ...] = (
    # Will be added in separate PR
)


def get_registered_dataset_import_adapters() -> list[DatasetImportAdapter]:
    """Return fresh adapter list for worker/runtime usage."""
    return list(REGISTERED_DATASET_IMPORT_ADAPTERS)


def get_supported_dataset_import_sources() -> set[str]:
    """Single source of truth for currently supported import source hints."""
    return {"auto", *[adapter.source.value for adapter in REGISTERED_DATASET_IMPORT_ADAPTERS]}


def select_dataset_import_adapter(
    adapters: list[DatasetImportAdapter],
    source_hint: str,
    archive: SafeZipArchive,
) -> DatasetImportAdapter | None:
    """Select the appropriate adapter given a *source_hint* and an open *archive*.

    Behavior:
    - If *source_hint* is ``"auto"``: iterate adapters and return the first
      whose :meth:`detect` returns ``True``, or ``None`` if none match.
    - If *source_hint* is a known adapter source value: require that adapter's
      :meth:`detect` to pass; raise :class:`ValueError` if it does not.
    - If *source_hint* is not recognized: raise :class:`ValueError` listing
      the known source values.
    """
    if source_hint == "auto":
        for adapter in adapters:
            if adapter.detect(archive):
                return adapter
        return None

    hinted = next((a for a in adapters if a.source == source_hint), None)
    if hinted is None:
        known = [a.source.value for a in adapters]
        raise ValueError(f"Source hint '{source_hint}' is not recognized. Known adapters: {known}")

    if not hinted.detect(archive):
        raise ValueError(f"The uploaded archive is not compatible with the selected format '{source_hint}'")
    return hinted


__all__ = [
    "DatasetImportAdapter",
    "get_registered_dataset_import_adapters",
    "get_supported_dataset_import_sources",
    "select_dataset_import_adapter",
]
