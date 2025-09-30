"""Tests for dataset root resolution utilities."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.common.path import DatasetRootResolutionError, resolve_dataset_root


def test_absolute_hint(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "email"
    dataset_dir.mkdir()

    resolved = resolve_dataset_root(str(dataset_dir), project_root=tmp_path)

    assert resolved == dataset_dir


def test_relative_to_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    cfg_dir = project_root / "configs"
    cfg_dir.mkdir()
    dataset_dir = cfg_dir / "data" / "email"
    dataset_dir.mkdir(parents=True)

    resolved = resolve_dataset_root(
        "data/email",
        project_root=project_root,
        config_path=cfg_dir / "default.yaml",
    )

    assert resolved == dataset_dir


def test_environment_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    env_root = tmp_path / "datasets"
    env_root.mkdir()
    dataset_dir = env_root / "email"
    dataset_dir.mkdir()

    monkeypatch.setenv("HGNN_DATA_ROOT", str(env_root))

    resolved = resolve_dataset_root("email", project_root=project_root)

    assert resolved == dataset_dir


def test_project_root_fallback(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    dataset_dir = project_root / "data" / "raw" / "email"
    dataset_dir.mkdir(parents=True)

    resolved = resolve_dataset_root(
        "data/raw/email",
        project_root=project_root,
    )

    assert resolved == dataset_dir


def test_legacy_src_fallback(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    dataset_dir = project_root / "src" / "data" / "raw" / "email"
    dataset_dir.mkdir(parents=True)

    resolved = resolve_dataset_root(
        "data/raw/email",
        project_root=project_root,
    )

    assert resolved == dataset_dir


def test_resolution_error_lists_attempts(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()

    with pytest.raises(DatasetRootResolutionError) as exc_info:
        resolve_dataset_root("missing", project_root=project_root)

    message = str(exc_info.value)
    assert "missing" in message
    assert "project" in message

