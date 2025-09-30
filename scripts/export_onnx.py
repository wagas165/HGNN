"""Export trained DF-HGNN model to ONNX."""
from __future__ import annotations

import argparse
import torch

from src.common.logging import setup_logging, get_logger
from src.models.df_hgnn import DFHGNN, DFHGNNConfig

LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export DF-HGNN to ONNX")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--out", type=str, default="df_hgnn.onnx")
    parser.add_argument("--in-dim", type=int, required=True)
    parser.add_argument("--det-dim", type=int, required=True)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--out-dim", type=int, required=True)
    parser.add_argument("--conv-type", type=str, default="mp")
    parser.add_argument("--cheb-order", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    model = DFHGNN(
        DFHGNNConfig(
            in_dim=args.in_dim,
            det_dim=args.det_dim,
            hidden_dim=args.hidden,
            out_dim=args.out_dim,
            conv_type=args.conv_type,
            chebyshev_order=args.cheb_order,
        )
    )
    state_dict = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    dummy_x = torch.randn(1, args.in_dim)
    dummy_z = torch.randn(1, args.det_dim)
    dummy_incidence = torch.randn(1, 1)
    dummy_weights = torch.ones(1)

    torch.onnx.export(
        model,
        (dummy_x, dummy_z, dummy_incidence, dummy_weights),
        args.out,
        input_names=["x", "z", "incidence", "edge_weights"],
        output_names=["logits", "gate"],
        opset_version=17,
    )
    LOGGER.info("Exported ONNX model to %s", args.out)


if __name__ == "__main__":
    main()
