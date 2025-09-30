from __future__ import annotations

import numpy as np
import pytest

from src.data.loaders.email_eu_full import EmailEuFullConfig, EmailEuFullLoader


def _write_lines(path, lines):
    path.write_text("\n".join(str(line) for line in lines))


def test_email_eu_full_loader_parses_flat_simplices(tmp_path):
    root = tmp_path
    nverts_path = root / "email-Eu-full-nverts.txt"
    simplices_path = root / "email-Eu-full-simplices.txt"

    cardinalities = [2, 1, 0, 3]
    _write_lines(nverts_path, cardinalities)

    simplices_lines = ["0 1", "2", "", "3 4", "5"]
    simplices_path.write_text("\n".join(simplices_lines))

    config = EmailEuFullConfig(
        vertices_file="email-Eu-full-nverts.txt",
        simplices_file="email-Eu-full-simplices.txt",
    )

    loader = EmailEuFullLoader(str(root), config)
    data = loader.load()

    assert data.num_nodes == 6
    assert data.incidence.shape == (6, 4)
    assert data.metadata["num_hyperedges"] == len(cardinalities)
    assert data.metadata["avg_cardinality"] == pytest.approx(float(np.mean(cardinalities)))

    # The third hyperedge is empty and should produce a zero column.
    assert data.incidence[:, 2].sum().item() == pytest.approx(0.0)

    # Node features default to zeros with one feature per node.
    assert data.node_features.shape == (6, 1)
    assert data.edge_weights.shape == (4,)
    assert data.timestamps is None


def test_email_eu_full_loader_validates_entry_counts(tmp_path):
    root = tmp_path
    (root / "email-Eu-full-nverts.txt").write_text("\n".join(["2", "2"]))
    (root / "email-Eu-full-simplices.txt").write_text("0 1 2")

    config = EmailEuFullConfig(
        vertices_file="email-Eu-full-nverts.txt",
        simplices_file="email-Eu-full-simplices.txt",
    )

    loader = EmailEuFullLoader(str(root), config)

    with pytest.raises(ValueError):
        loader.load()
