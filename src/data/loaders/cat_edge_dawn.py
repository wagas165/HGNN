"""Loader for the Cat-Edge-DAWN hypergraph dataset."""
from __future__ import annotations

from dataclasses import dataclass

from .generic import EdgeListHypergraphConfig, EdgeListHypergraphLoader


@dataclass
class CatEdgeDawnConfig(EdgeListHypergraphConfig):
    """Configuration for Cat-Edge-DAWN."""

    min_cardinality: int = 2


class CatEdgeDawnLoader(EdgeListHypergraphLoader):
    """Hypergraph loader tailored to Cat-Edge-DAWN conventions."""

    def __init__(self, root: str, config: CatEdgeDawnConfig) -> None:
        super().__init__(root, config)
