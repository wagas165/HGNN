# DF-HGNN Project

DF-HGNN (Deterministic Feature Hypergraph Neural Network) is a research prototype that
explores fusing deterministic hypergraph statistics with learnable neural components
for classification and regression tasks on higher-order relational data.

## Repository layout

- `configs/` – Hydra-compatible experiment configurations (see `configs/default.yaml`).
- `scripts/` – entry-points for training and evaluation (`train_df_hgnn.py`).
- `src/` – library code for data loading, feature engineering, models, and training.
- `docs/` – background material, including the architecture rationale and task plan.
- `tests/` – unit tests for selected components.

## Getting started

### 1. Create an environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

`pip install -e .` relies on `pyproject.toml` and installs the project in editable mode with
its Python dependencies. PyTorch is required but not pinned; install the version that matches
your CUDA setup, e.g.:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 2. Prepare data

The default experiments target the Email-Eu hypergraph from the SNAP collection. Download
and unpack the dataset into `data/raw/email-Eu-full` so that the directory contains:

- `email-Eu-full-nverts.txt`
- `email-Eu-full-simplices.txt`
- `email-Eu-full-times.txt`
- `email-Eu-full-labels.json` (textual copy of the degree-quantile node classes)
- `email-Eu-full-labels.npy` (materialised automatically from the JSON when needed)

> **Why the label file matters:**
> DF-HGNN's node-classification recipes assume supervised targets. We provide
> `email-Eu-full-labels.json`, which bins nodes into four classes by email hyperedge
> participation. Custom labels may be substituted, but the preprocessing script will
> materialise the binary NumPy array (`email-Eu-full-labels.npy`) from the JSON source
> and raise an error if neither artifact is present. This avoids silently training
> without supervision while keeping the tracked label data human-readable.

The training script resolves dataset directories in the following order:

1. An absolute path provided in the configuration.
2. The directory that contains the referenced configuration file (allowing
   per-experiment data folders).
3. A path pointed to by the `HGNN_DATA_ROOT` environment variable (either the
   dataset root itself or a parent directory).
4. `data/raw/...` under the project root.
5. The legacy `src/data/raw/...` tree for backwards compatibility.

If none of these locations exists, the script will list every attempted path in
the error message to aid debugging. Adjust `configs/default.yaml` (see the
`data` section) or set `HGNN_DATA_ROOT` if your datasets live elsewhere.


### 3. Run training

Train DF-HGNN with the default configuration:

```bash
python scripts/train_df_hgnn.py --config configs/default.yaml
```

Training logs and checkpoints are written under `outputs/` by default. The script produces
a metrics report (`.json`) summarising train/validation/test performance.

### 4. Customising experiments

- Adjust model hyperparameters (hidden size, convolution type, alignment/gating penalties)
in the `model` section of the config.
- Enable or disable deterministic feature families under `features.deterministic` using
  the `enabled` flag, and control where they are computed via `device` (use `auto` to
  prefer CUDA) and
  `expansion_chunk_size` for large hypergraphs.
- Edit `trainer` to change the Adam/L-BFGS schedule or gradient clipping.
- Toggle `trainer.pin_memory` to optimise host-to-device transfers when running on GPU.
- Modify `data.split` to alter the train/validation/test ratios or use random splits.

### 5. Documentation & further reading

See the documents in `docs/` for the theoretical motivation (`architecture.md`), planned
experiments (`task_tech_plan.md`), and dataset assumptions (`data-spec.md`).

## Testing

Run the unit test suite (requires PyTorch and dataset fixtures):

```bash
pytest
```

You can also perform a quick syntax/bytecode check without full dependencies:

```bash
python -m compileall src
```

## License

This project is research code; please consult the authors before redistribution.
