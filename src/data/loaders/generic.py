"""Generic loaders for edge-list style hypergraph datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import torch

from .base import HypergraphData, HypergraphDatasetLoader


@dataclass
class EdgeListHypergraphConfig:
    """Configuration for edge-list based hypergraph datasets.

    Attributes:
        simplices_file: Text file where each row describes a hyperedge via
            whitespace- or delimiter-separated integer node ids.
        num_nodes: Optional explicit node count. When omitted the loader will
            infer it from the maximum node id observed in the simplices file.
        node_features_file: Optional path to a dense feature matrix stored as
            ``.npy``/``.npz``/``.txt``/``.csv``.
        label_file: Optional path to node labels stored as a vector in the same
            formats accepted by :attr:`node_features_file`.
        times_file: Optional path to hyperedge timestamps, one floating point
            value per hyperedge.
        delimiter: Delimiter used to split columns in :attr:`simplices_file`.
            ``None`` delegates to ``str.split``'s default whitespace handling.
        comment_prefix: Lines starting with this prefix are skipped.
        min_cardinality: Minimum allowed number of nodes per hyperedge. Smaller
            hyperedges will trigger a ``ValueError``.
    """

    simplices_file: str
    num_nodes: Optional[int] = None
    node_features_file: Optional[str] = None
    label_file: Optional[str] = None
    times_file: Optional[str] = None
    delimiter: Optional[str] = None
    comment_prefix: str = "#"
    min_cardinality: int = 1


class EdgeListHypergraphLoader(HypergraphDatasetLoader):
    """Loader for hypergraphs described by simple edge-list text files."""

    def __init__(self, root: str, config: EdgeListHypergraphConfig) -> None:
        super().__init__(root)
        self.config = config

    def load(self) -> HypergraphData:
        simplices_path = self.root / self.config.simplices_file
        self._check_exists(self.config.simplices_file)

        simplices = self._read_simplices(simplices_path)
        if not simplices:
            raise ValueError(f"No simplices found in {simplices_path}")

        num_nodes = self.config.num_nodes or self._infer_num_nodes(simplices)
        if num_nodes <= 0:
            raise ValueError("Inferred number of nodes must be positive")

        incidence = self._build_incidence(num_nodes, simplices)
        edge_weights = torch.ones(incidence.shape[1], dtype=torch.float32)

        node_features = self._load_array(
            self.config.node_features_file,
            num_nodes,
            allow_zero_cols=True,
            dtype=torch.float32,
            ensure_2d=True,
        )
        labels = self._load_array(
            self.config.label_file,
            num_nodes,
            allow_zero_cols=True,
            dtype=torch.long,
            ensure_2d=False,
        )

        timestamps = None
        if self.config.times_file:
            timestamps = self._read_times(self.root / self.config.times_file, len(simplices))

        metadata = {
            "num_hyperedges": incidence.shape[1],
            "avg_cardinality": float(np.mean([len(s) for s in simplices])),
        }

        return HypergraphData(
            num_nodes=num_nodes,
            incidence=incidence,
            edge_weights=edge_weights,
            node_features=node_features,
            labels=labels,
            timestamps=timestamps,
            metadata=metadata,
        )

    def _read_simplices(self, path: Path) -> List[List[int]]:
        simplices: List[List[int]] = []
        with path.open("r", encoding="utf-8") as f:
            for line_num, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line or line.startswith(self.config.comment_prefix):
                    continue
                tokens = line.split(self.config.delimiter)
                try:
                    simplex = [int(tok) for tok in tokens if tok]
                except ValueError as exc:
                    raise ValueError(
                        f"Failed to parse simplices line {line_num} in {path}: {raw_line!r}"
                    ) from exc
                if len(simplex) < self.config.min_cardinality:
                    raise ValueError(
                        "Encountered hyperedge with cardinality "
                        f"{len(simplex)} (minimum allowed {self.config.min_cardinality})"
                    )
                simplices.append(simplex)
        return simplices

    def _infer_num_nodes(self, simplices: Iterable[Iterable[int]]) -> int:
        max_idx = -1
        for simplex in simplices:
            if simplex:
                max_idx = max(max_idx, max(simplex))
        return max_idx + 1

    def _build_incidence(self, num_nodes: int, simplices: List[List[int]]) -> torch.Tensor:
        num_edges = len(simplices)
        incidence = torch.zeros((num_nodes, num_edges), dtype=torch.float32)
        for e_idx, simplex in enumerate(simplices):
            for v in simplex:
                if v < 0 or v >= num_nodes:
                    raise ValueError(
                        f"Node index {v} in simplex {e_idx} exceeds inferred node count {num_nodes}"
                    )
                incidence[v, e_idx] = 1.0
        return incidence

    def _load_array(
        self,
        filename: Optional[str],
        expected_rows: int,
        allow_zero_cols: bool = False,
        dtype: torch.dtype = torch.float32,
        ensure_2d: bool = True,
    ) -> Optional[torch.Tensor]:
        if not filename:
            return None
        path = self.root / filename
        if not path.exists():
            raise FileNotFoundError(f"Required dataset artifact not found: {path}")

        data = self._read_numerical_array(path)
        if data.shape[0] != expected_rows:
            raise ValueError(
                f"Array {path} has {data.shape[0]} rows but expected {expected_rows}"
            )
        if ensure_2d:
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            if data.shape[1] == 0 and not allow_zero_cols:
                raise ValueError(f"Array {path} has zero feature columns")
        else:
            data = data.reshape(expected_rows)
        return torch.tensor(data, dtype=dtype)

    def _read_times(self, path: Path, expected: int) -> torch.Tensor:
        values: List[float] = []
        with path.open("r", encoding="utf-8") as f:
            for line_num, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line or line.startswith(self.config.comment_prefix):
                    continue
                try:
                    values.append(float(line))
                except ValueError as exc:
                    raise ValueError(
                        f"Failed to parse timestamp line {line_num} in {path}: {raw_line!r}"
                    ) from exc
        if len(values) != expected:
            raise ValueError(
                f"Timestamp file {path} length {len(values)} does not match number of simplices {expected}"
            )
        return torch.tensor(values, dtype=torch.float32)

    def _read_numerical_array(self, path: Path) -> np.ndarray:
        suffix = path.suffix.lower()
        if suffix == ".npy":
            return np.load(path)
        if suffix == ".npz":
            data = np.load(path)
            if len(data.files) == 1:
                return np.asarray(data[data.files[0]])
            for key in ("X", "features", "data", "array"):
                if key in data:
                    return np.asarray(data[key])
            raise ValueError(f"Unable to infer array entry from {path}")
        if suffix in {".csv", ".txt", ".tsv"}:
            delimiter = "," if suffix == ".csv" else None
            return np.loadtxt(path, delimiter=delimiter)
        raise ValueError(f"Unsupported array format for {path}")
