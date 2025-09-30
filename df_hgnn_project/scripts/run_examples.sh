
#!/usr/bin/env bash
set -e

# Example (assuming you prepared datasets/stackoverflow/processed/data.pt)
python -m src.train --data ./datasets/stackoverflow/processed/data.pt --model hgnn --epochs 150
python -m src.train --data ./datasets/stackoverflow/processed/data.pt --model hnhn --epochs 150
python -m src.train --data ./datasets/stackoverflow/processed/data.pt --model hypergcn --epochs 150
python -m src.train --data ./datasets/stackoverflow/processed/data.pt --model df_hgnn --epochs 120 --lbfgs_epochs 30 --topk_spectral 16
