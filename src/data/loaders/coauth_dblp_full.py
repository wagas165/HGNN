"""Loader for the coauth-DBLP-full hypergraph dataset."""
from __future__ import annotations

from dataclasses import dataclass

from .generic import EdgeListHypergraphConfig, EdgeListHypergraphLoader


@dataclass
class CoauthDblpFullConfig(EdgeListHypergraphConfig):
    """Configuration defaults for coauth-DBLP-full."""

    min_cardinality: int = 2


class CoauthDblpFullLoader(EdgeListHypergraphLoader):
    """Hypergraph loader tailored to coauth-DBLP-full."""

    def __init__(self, root: str, config: CoauthDblpFullConfig) -> None:
        super().__init__(root, config)
