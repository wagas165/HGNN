
# DF-HGNN Project (Deterministic-Feature–Augmented Hypergraph Neural Networks)

> 一套可复现的**超图**学习项目模板，聚焦“预测性能”提升：包含数据预处理、三类基线（HGNN/HNHN/HyperGCN-扩展图）、以及确定性特征增强（DF-HGNN）。
>
> - 任务自动识别：按数据可用性选择 *节点分类* / *超边分类* / *超边补全（link prediction）*。
> - 预处理：统一把数据转存为 `processed/data.pt`（节点数、超边列表、特征、标签、划分）。
> - 增强特征：结构确定性特征（超度、超边基数统计、谱位置编码），**自动维度对齐**与门控融合。

## 安装

```bash
conda create -n dfhgnn python=3.10 -y
conda activate dfhgnn
pip install -r requirements.txt
```

## 数据准备（两种方式）

1) **推荐：YAML 配置**（见 `configs/` 示例）指定文件路径：
- `node_features`: `.npy` 或 `.csv`（行对应节点，列为特征）；若无则自动以确定性特征代替。
- `node_labels`: `.npy` 或 `.csv`（长度 N）；可缺省。
- `edge_labels`: `.npy` 或 `.csv`（长度 M）；可缺省。
- `hyperedges`: 支持
  - `txt`：每行一个超边，节点 id 以空格/逗号分割
  - `json`：`[[nodes...], [nodes...], ...]`
  - `npz`：稀疏入射矩阵（N×M）

2) **直接放置**文件到数据目录，并用 `--data_dir` 调用（自动尝试常见命名）。

预处理：
```bash
python -m src.data.preprocess --config configs/example_stackoverflow.yaml --out_dir ./datasets/stackoverflow
# 或者
python -m src.data.preprocess --data_dir ./datasets/stackoverflow_raw --out_dir ./datasets/stackoverflow
```

## 训练与评测

```bash
# HGNN 基线
python -m src.train --data ./datasets/stackoverflow/processed/data.pt --model hgnn --epochs 150

# HNHN 基线
python -m src.train --data ./datasets/stackoverflow/processed/data.pt --model hnhn --epochs 150

# HyperGCN（超边 -> 完全图近似 + GCN）
python -m src.train --data ./datasets/stackoverflow/processed/data.pt --model hypergcn --epochs 150

# DF-HGNN（确定性特征增强 + 门控融合 + 对齐正则 + Adam→L-BFGS）
python -m src.train --data ./datasets/stackoverflow/processed/data.pt --model df_hgnn --epochs 120 --lbfgs_epochs 30 --topk_spectral 16
```

### 任务选择规则

- 若存在 `node_labels` → 执行**节点分类**（优先）。
- 否则若存在 `edge_labels` → 执行**超边分类**。
- 否则 → 执行**超边补全**（负采样 + AUC/AP）。

## 目录结构

```
df_hgnn_project/
  configs/
    example_stackoverflow.yaml
    example_dawn.yaml
  src/
    data/
      preprocess.py   # 将原始文件转为统一格式 data.pt
      loaders.py      # 读写统一数据格式
      registry.py     # 根据数据可用性选择任务
      splits.py       # 划分 train/val/test（分层）
      utils_io.py     # 常用 IO
    models/
      layers.py       # 超图算子与基础层
      hgnn.py         # HGNN 基线
      hnhn.py         # HNHN 基线（节点-超边二部图传递）
      hypergcn.py     # 超边完全图近似 + 简单 GCN
      det_features.py # 确定性特征（超度/基数统计/谱位置编码）
      df_hgnn.py      # 增强模型（门控融合+对齐正则）
    metrics.py
    train.py
  scripts/
    run_examples.sh
  requirements.txt
  README.md
```

## 注意事项
- 若特征缺失：训练时自动以确定性特征作为输入；`--use_raw_features False` 可强制仅用确定性特征。
- 统一**维度对齐**：所有输入先投影到相同维度后再门控融合，避免维度不一致。
- 光谱特征在大图上计算成本较高：可调小 `--topk_spectral` 或置 `--use_spectral False`。

## 引用基线与算子来源
- HGNN / HypergraphConv（Bai et al., AAAI 2019；PyTorch Geometric 文档）
- HNHN（Dong et al., 2020）
- HyperGCN（Yadati et al., NeurIPS 2019）
```

(open-source papers and docs; see training script comments for pointers.)
