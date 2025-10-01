"""Path utilities for dataset and artifact resolution."""
from __future__ import annotations

import os
import tarfile
import zipfile
from pathlib import Path
from typing import Iterable, List, Optional

from .logging import get_logger


LOGGER = get_logger(__name__)


class DatasetRootResolutionError(FileNotFoundError):
    """Raised when dataset root resolution fails."""

    def __init__(self, root_hint: str, attempted: Iterable[Path]) -> None:
        attempts = ",\n".join(f"  - {path}" for path in attempted)
        message = (
            "Unable to resolve dataset root from hint '"
            f"{root_hint}'.\nChecked the following locations:\n{attempts}"
        )
        super().__init__(message)
        self.root_hint = root_hint
        self.attempted = list(attempted)


def _normalise(path: Path) -> Path:
    """Return a normalised version of ``path`` without resolving symlinks."""

    expanded = path.expanduser()
    try:
        return expanded.resolve(strict=False)
    except RuntimeError:
        # ``resolve`` can fail on some network mounts; fall back to ``absolute``.
        return expanded.absolute()


def _append_candidate(candidates: List[Path], seen: set[Path], candidate: Path) -> None:
    normalised = _normalise(candidate)
    if normalised not in seen:
        candidates.append(normalised)
        seen.add(normalised)


def _extract_archive(archive_path: Path, destination: Path) -> Optional[Path]:
    """Extract ``archive_path`` into ``destination`` if possible."""

    if destination.exists() and any(destination.iterdir()):
        return destination

    destination.mkdir(parents=True, exist_ok=True)

    try:
        if zipfile.is_zipfile(archive_path):
            with zipfile.ZipFile(archive_path) as archive:
                archive.extractall(destination)
        elif tarfile.is_tarfile(archive_path):
            with tarfile.open(archive_path) as archive:
                archive.extractall(destination)
        else:
            return None
    except (OSError, zipfile.BadZipFile, tarfile.TarError) as exc:  # pragma: no cover - logging path
        LOGGER.warning("Failed to extract dataset archive %s: %s", archive_path, exc)
        return None

    # Some archives contain a top-level directory matching the destination.
    # Flatten the structure so that ``destination`` always holds the dataset
    # contents directly.
    entries = [entry for entry in destination.iterdir()]
    if len(entries) == 1 and entries[0].is_dir():
        inner = entries[0]
        if inner.name == destination.name:
            for child in list(inner.iterdir()):
                child.rename(destination / child.name)
            inner.rmdir()
            return destination
        return inner

    return destination


def _maybe_extract_candidate(candidate: Path) -> Optional[Path]:
    """Attempt to extract an archive that matches ``candidate``."""

    archive_suffixes = [".zip", ".tar.gz", ".tgz", ".tar"]
    for suffix in archive_suffixes:
        if suffix == ".tar.gz":
            archive_path = candidate.parent / f"{candidate.name}.tar.gz"
        else:
            archive_path = candidate.with_suffix(suffix)

        if archive_path.exists():
            extracted = _extract_archive(archive_path, candidate)
            if extracted and extracted.exists():
                LOGGER.info("Extracted dataset archive %s to %s", archive_path, extracted)
                return extracted
    return None


def resolve_dataset_root(
    root_hint: str | Path,
    project_root: Path,
    *,
    config_path: Optional[Path] = None,
    env_var: str = "HGNN_DATA_ROOT",
) -> Path:
    """Resolve a dataset root from a configuration hint.

    This helper looks for the dataset directory in a series of locations to
    accommodate different project layouts:

    1. The hint itself when it points to an existing absolute path.
    2. The directory that contains the configuration file.
    3. A project-level override provided via the ``HGNN_DATA_ROOT`` environment
       variable (both as a direct path and as a prefix).
    4. The project root.
    5. A legacy ``src/`` sub-tree within the project root.
    6. The current working directory.

    Args:
        root_hint: Value coming from the configuration file.
        project_root: Repository root.
        config_path: Absolute path to the resolved configuration file, if
            available.
        env_var: Environment variable name for dataset root overrides.

    Returns:
        A resolved :class:`pathlib.Path` pointing to the dataset directory.

    Raises:
        DatasetRootResolutionError: If none of the candidate paths exists.
    """

    hint_path = Path(root_hint).expanduser()
    candidates: List[Path] = []
    seen: set[Path] = set()

    def add_candidate(path: Path) -> None:
        _append_candidate(candidates, seen, path)

    if hint_path.is_absolute():
        add_candidate(hint_path)
    else:
        if config_path is not None:
            add_candidate(config_path.parent / hint_path)

    env_override = os.getenv(env_var)
    if env_override:
        env_path = Path(env_override).expanduser()
        if not hint_path.is_absolute():
            add_candidate(env_path / hint_path)
        add_candidate(env_path)

    if not hint_path.is_absolute():
        add_candidate(project_root / hint_path)
        add_candidate(project_root / "src" / hint_path)
        add_candidate(Path.cwd() / hint_path)

    # As a last resort, include the hint itself relative to the current
    # working directory for clarity in the error message.
    add_candidate(hint_path)

    for candidate in candidates:
        if not candidate.exists():
            extracted = _maybe_extract_candidate(candidate)
            if extracted is not None:
                return extracted
        if candidate.exists():
            return candidate

    raise DatasetRootResolutionError(str(root_hint), candidates)


__all__ = ["DatasetRootResolutionError", "resolve_dataset_root"]

