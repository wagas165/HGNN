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

> **Cat-Edge-DAWN preparation:**
> The raw Cat-Edge-DAWN dump ships with 1-indexed hyperedges and textual node
> annotations. Run `PYTHONPATH=. python scripts/prepare_cat_edge_dawn_raw.py`
> once after downloading the archive to materialise
> `cat-edge-DAWN-simplices-zero-based.txt`, `cat-edge-DAWN-node-labels-int.txt`,
> and the accompanying label map JSON (pass `--node-label-output-binary` if you
> also need a `.npy` dump). The standard preprocessing script
> (`scripts/preprocess_cat_edge_dawn.py`) then consumes these normalised files
> to build the tensor bundle used during training.


### 3. Run training

Train DF-HGNN with the default configuration:

```bash
python scripts/train_df_hgnn.py --config configs/default.yaml
```

The script now derives the experiment name from the configuration filename and writes
artefacts to `results/<config_name>/`. For example, running with `configs/email_config.yaml`
creates `results/email_config/` with the following structure:

```
results/
└── email_config/
    ├── train.log
    ├── reports/
    │   ├── metrics.json
    │   ├── metrics_bar.png
    │   └── … (ROC/PR/Confusion plots when probabilities are cached)
    ├── features/
    │   └── … (deterministic feature cache)
    └── checkpoints/
        └── … (if checkpointing is enabled in the config)
```

`train.log` mirrors the console output so that every run keeps its own log snapshot.
You can safely launch several experiments without overwriting previous results.

### 4. One-click execution helper

For remote servers or batch experimentation, use the `run.py` convenience wrapper:

```bash
python run.py --config configs/email_config.yaml
```

The wrapper tees console output into `results/<config_name>/train.log` and guarantees that
all required folders exist. Add `--save-plots` to invoke the analysis script after training;
pass extra experiment directories (e.g. previous runs) via `--compare` to include them in
the comparison plots.

### 5. Visualise and compare experiments

`scripts/analyze_results.py` reads one or more `metrics.json` files and generates
publication-ready charts:

```bash
python scripts/analyze_results.py \
    --inputs results/baseline results/our_method results/ablation \
    --metrics-keys test_accuracy test_macro_f1 test_roc_auc \
    --boxplot-metric test_accuracy \
    --output-dir results/analysis
```

The script produces per-method bar charts, a combined comparison bar plot, an optional
box plot (when multiple runs per method exist), and a CSV table with summary statistics.
Use `label=path` (for example, `Baseline=results/baseline`) to override legend names when
necessary. Combine this tool with `run.py --save-plots` for end-to-end automation.

### 6. Customising experiments

- Adjust model hyperparameters (hidden size, convolution type, alignment/gating penalties)
in the `model` section of the config.
- Enable or disable deterministic feature families under `features.deterministic` using
  the `enabled` flag, and control where they are computed via `device` (use `auto` to
  prefer CUDA) and
  `expansion_chunk_size` for large hypergraphs.
- Edit `trainer` to change the Adam/L-BFGS schedule or gradient clipping.
- Toggle `trainer.pin_memory` to optimise host-to-device transfers when running on GPU.
- Modify `data.split` to alter the train/validation/test ratios or use random splits.

## Baseline implementations

Alongside DF-HGNN we provide three baseline encoders that share the same feature pipeline and
training loop (invoke them with `python scripts/train_df_hgnn.py --config <experiment.yaml>`):

- **AllSet Transformer** (Chien et al., NeurIPS 2022) relies on multi-head attention between
  node and hyperedge representations. Our default run uses a 160-dimensional hidden size, three
  transformer blocks with four heads, and an MLP expansion ratio of 2.0.【F:configs/experiment/allset_transformer.yaml†L1-L15】
- **UniGNN** (Huang et al., NeurIPS 2021) couples edge and node updates with residual diffusion.
  The reference configuration keeps three stacked layers with dropout 0.3 to mimic the published
  setup.【F:configs/experiment/unignn.yaml†L1-L14】
- **HyperGCN** (Yadati et al., NeurIPS 2019) builds a symmetric normalised Laplacian from the
  incidence matrix and applies three diffusion layers with 0.25 dropout.【F:configs/experiment/hypergcn.yaml†L1-L14】

All three variants depend only on PyTorch (no extra third-party packages) and reuse the deterministic
feature computation shipped with DF-HGNN, enabling apples-to-apples comparisons across baselines.

### 7. Documentation & further reading

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
