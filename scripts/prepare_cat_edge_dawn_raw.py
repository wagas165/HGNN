"""Utilities to normalize Cat-Edge-DAWN raw files for training."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.common.logging import get_logger, setup_logging

LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize Cat-Edge-DAWN raw artifacts")
    parser.add_argument("--root", type=str, default="data/raw/cat-edge-DAWN", help="Dataset root directory")
    parser.add_argument(
        "--simplices-input",
        type=str,
        default="cat-edge-DAWN-simplices.txt",
        help="Relative path to the original (1-indexed) hyperedge list",
    )
    parser.add_argument(
        "--simplices-output",
        type=str,
        default="cat-edge-DAWN-simplices-zero-based.txt",
        help="Destination filename for the zero-based hyperedge list",
    )
    parser.add_argument(
        "--hyperedge-labels",
        type=str,
        default="cat-edge-DAWN-hyperedge-labels.txt",
        help="Relative path to the hyperedge disposition labels",
    )
    parser.add_argument(
        "--label-identities",
        type=str,
        default="cat-edge-DAWN-hyperedge-label-identities.txt",
        help="Relative path mapping disposition ids to human-readable names",
    )
    parser.add_argument(
        "--node-metadata",
        type=str,
        default="cat-edge-DAWN-node-labels.txt",
        help="Relative path listing node ids and their drug names",
    )
    parser.add_argument(
        "--node-label-output",
        type=str,
        default="cat-edge-DAWN-node-labels-int.txt",
        help="Destination filename for the derived node label array (text format)",
    )
    parser.add_argument(
        "--node-label-output-binary",
        type=str,
        default=None,
        help="Optional destination filename for an additional binary (.npy) label dump",
    )
    parser.add_argument(
        "--label-map-output",
        type=str,
        default="cat-edge-DAWN-node-label-map.json",
        help="Destination filename for the class mapping metadata",
    )
    return parser.parse_args()


def _read_simplices(path: Path) -> List[List[int]]:
    simplices: List[List[int]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            try:
                simplex = [int(tok) for tok in parts]
            except ValueError as exc:
                raise ValueError(f"Failed to parse simplices line {line_num} in {path}: {raw_line!r}") from exc
            if any(v <= 0 for v in simplex):
                raise ValueError(f"Encountered non-positive node id in line {line_num}: {simplex}")
            simplices.append(simplex)
    return simplices


def _read_hyperedge_labels(path: Path) -> List[int]:
    labels: List[int] = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                labels.append(int(line))
            except ValueError as exc:
                raise ValueError(f"Failed to parse hyperedge label line {line_num} in {path}: {raw_line!r}") from exc
    return labels


def _read_node_metadata(path: Path) -> Dict[int, str]:
    metadata: Dict[int, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 1)
            if not parts:
                continue
            try:
                node_id = int(parts[0])
            except ValueError:
                # Header row (e.g., "node_id name")
                continue
            name = parts[1].strip() if len(parts) > 1 else ""
            metadata[node_id] = name
    if not metadata:
        raise ValueError(f"No node metadata parsed from {path}")
    return metadata


def _read_label_names(path: Path) -> Dict[int, str]:
    names: Dict[int, str] = {}
    if not path.exists():
        return names
    with path.open("r", encoding="utf-8") as f:
        for idx, raw_line in enumerate(f, start=1):
            label = raw_line.strip()
            if not label:
                continue
            names[idx] = label
    return names


def main() -> None:
    setup_logging()
    args = parse_args()
    root = Path(args.root)

    simplices_path = root / args.simplices_input
    label_path = root / args.hyperedge_labels
    node_meta_path = root / args.node_metadata
    label_identities_path = root / args.label_identities

    LOGGER.info("Reading simplices from %s", simplices_path)
    simplices = _read_simplices(simplices_path)
    LOGGER.info("Loaded %d hyperedges", len(simplices))

    LOGGER.info("Reading hyperedge labels from %s", label_path)
    hyperedge_labels = _read_hyperedge_labels(label_path)
    if len(hyperedge_labels) != len(simplices):
        raise ValueError(
            "Mismatch between number of hyperedges (%d) and label entries (%d)" % (len(simplices), len(hyperedge_labels))
        )

    LOGGER.info("Reading node metadata from %s", node_meta_path)
    node_metadata = _read_node_metadata(node_meta_path)
    num_nodes = max(node_metadata)  # Node ids are 1-indexed
    LOGGER.info("Detected %d nodes", num_nodes)

    label_names = _read_label_names(label_identities_path)
    if label_names:
        LOGGER.info("Loaded %d disposition names", len(label_names))
    else:
        LOGGER.warning("Label identity file %s missing or empty", label_identities_path)

    zero_based_simplices = [[node - 1 for node in simplex] for simplex in simplices]
    simplices_output_path = root / args.simplices_output
    LOGGER.info("Writing zero-based hyperedges to %s", simplices_output_path)
    with simplices_output_path.open("w", encoding="utf-8") as f:
        for simplex in zero_based_simplices:
            f.write("\t".join(str(node) for node in simplex))
            f.write("\n")

    node_label_votes: List[Counter[int]] = [Counter() for _ in range(num_nodes)]
    for hyperedge, label in zip(zero_based_simplices, hyperedge_labels):
        for node in hyperedge:
            node_label_votes[node][label] += 1

    unknown_label = 0
    unknown_assigned = 0
    node_labels_original = np.zeros(num_nodes, dtype=np.int64)
    for node_idx, votes in enumerate(node_label_votes):
        if not votes:
            node_labels_original[node_idx] = unknown_label
            unknown_assigned += 1
            continue
        most_common_count = max(votes.values())
        candidates = [label for label, count in votes.items() if count == most_common_count]
        chosen_label = min(candidates)
        node_labels_original[node_idx] = chosen_label

    if unknown_assigned:
        LOGGER.warning("Assigned 'Unknown' label to %d nodes with no incident hyperedges", unknown_assigned)

    unique_original_labels = sorted(set(node_labels_original.tolist()))
    label_to_index = {label: idx for idx, label in enumerate(unique_original_labels)}
    node_labels = np.array([label_to_index[label] for label in node_labels_original], dtype=np.int64)

    class_counts: Dict[int, int] = defaultdict(int)
    for label_idx in node_labels:
        class_counts[int(label_idx)] += 1

    label_map_entries = []
    for original_label in unique_original_labels:
        index = label_to_index[original_label]
        name = label_names.get(original_label)
        if original_label == unknown_label and unknown_assigned > 0:
            name = name or "Unknown"
        elif not name:
            name = f"Label {original_label}"
        label_map_entries.append(
            {
                "index": int(index),
                "original_label": int(original_label),
                "name": name,
                "count": class_counts[int(index)],
            }
        )

    labels_output_path = root / args.node_label_output
    LOGGER.info("Saving node labels to %s", labels_output_path)
    np.savetxt(labels_output_path, node_labels, fmt="%d")

    if args.node_label_output_binary:
        binary_output_path = root / args.node_label_output_binary
        LOGGER.info("Saving binary node labels to %s", binary_output_path)
        np.save(binary_output_path, node_labels)

    mapping_output_path = root / args.label_map_output
    LOGGER.info("Writing label mapping metadata to %s", mapping_output_path)
    metadata = {
        "num_nodes": num_nodes,
        "num_classes": len(unique_original_labels),
        "unknown_assigned": unknown_assigned,
        "classes": label_map_entries,
    }
    with mapping_output_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    LOGGER.info(
        "Finished preprocessing Cat-Edge-DAWN raw files: %d classes, distribution=%s",
        len(unique_original_labels),
        {entry["index"]: entry["count"] for entry in label_map_entries},
    )


if __name__ == "__main__":
    main()
