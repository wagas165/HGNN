"""Tests for the generic hypergraph loader utilities."""
from __future__ import annotations

import logging
from pathlib import Path

import pytest

from src.data.loaders.generic import EdgeListHypergraphConfig, EdgeListHypergraphLoader


def _write_lines(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines))


def test_generic_loader_skips_small_hyperedges(tmp_path, caplog: pytest.LogCaptureFixture) -> None:
    simplices_path = Path(tmp_path) / "simplices.txt"
    _write_lines(simplices_path, ["0 1", "2", "3 4 5", "# comment"])

    config = EdgeListHypergraphConfig(simplices_file="simplices.txt", min_cardinality=2)
    loader = EdgeListHypergraphLoader(str(tmp_path), config)

    with caplog.at_level(logging.WARNING):
        simplices = loader._read_simplices(simplices_path)

    assert simplices == [[0, 1], [3, 4, 5]]
    assert "Dropped 1 hyperedges" in caplog.text


def test_generic_loader_detects_git_lfs_pointer(tmp_path) -> None:
    simplices_path = Path(tmp_path) / "simplices.txt"
    _write_lines(
        simplices_path,
        ["version https://git-lfs.github.com/spec/v1", "oid sha256:123", "size 42"],
    )

    config = EdgeListHypergraphConfig(simplices_file="simplices.txt")
    loader = EdgeListHypergraphLoader(str(tmp_path), config)

    with pytest.raises(RuntimeError, match="Git LFS"):
        loader._read_simplices(simplices_path)
