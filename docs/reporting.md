# Evaluation Reporting Workflow

The DF-HGNN training script produces publication-ready metrics summaries and
visual diagnostics. After `scripts/train_df_hgnn.py` completes, the
`save_metrics_report` helper writes a JSON file with scalar metrics and renders a
coherent suite of visualisations for every evaluation split (training, validation,
test, out-of-distribution, and transfer splits when present).

## Generated artefacts

All artefacts are stored under `outputs/reports/` by default. For each run the
following files are created:

- `metrics.json` – scalar metrics in JSON format.
- `metrics_bar.png` – grouped bar chart summarising accuracy/F1/AUC scores
  across available splits.
- `roc_<split>.png` – receiver operating characteristic curves per split.
- `pr_<split>.png` – precision–recall curves per split.
- `confusion_matrix_<split>.png` – annotated confusion matrices per split.

The `<split>` suffix matches the split identifier (e.g. `train`, `val`,
`test`, `transfer_test`).

## Plot configuration

To satisfy paper-quality typography and readability standards, the plotting
utilities enforce the following style rules:

- **Resolution:** figures are rendered at 300 DPI via `matplotlib`'s
  `savefig.dpi` configuration.
- **Canvas size:** each figure uses a 7×5 inch canvas to maintain consistent
  aspect ratios for grid layouts in publications.
- **Palette:** seaborn's `colorblind` palette is applied for colour-safe, high
  contrast curves and bars.
- **Typography:** the global font family is set to *DejaVu Sans* with axes titles
  at 18 pt, axis labels at 14 pt, legends at 12 pt, and tick labels at 12 pt.
- **Background:** plots use the `whitegrid` style for legible axes while
  retaining a neutral background suitable for print.

Macro- and micro-average curves are included for multi-class ROC/PR diagrams,
ensuring that both per-class performance and holistic trends are visible in a
single figure.

## Custom report locations

The destination directory can be overridden via the `reporting.dir` field in the
training configuration. All metrics and figures generated in the same run are
kept together in this directory to simplify manuscript asset management.
