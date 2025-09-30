# 数据规范

本文档定义 DF-HGNN 项目中超图数据的统一格式与质量要求。

## 原始文件

- **vertices (`*-nverts.txt`)**: 每行一个节点标识，按整数索引排序。
- **simplices (`*-simplices.txt`)**: 每行描述一个超边，使用空格分隔的节点索引。
- **times (`*-times.txt`)**: 每行一个时间戳（可选），与 `simplices` 行号对应。
- **edge list (`*-hyperedges.txt`)**: StackOverflow、Cat-Edge-DAWN、coauth-DBLP-full 等新增数据集以该格式提供，换行分隔超边、同一行内空格分隔节点索引。
- **node features (`*-node-features.npy`)**: NumPy 数组，行对应节点、列对应特征，可选。
- **node labels (`*-node-labels.*`)**: 节点标签，支持 `.npy`、`.npz`、`.txt` 或 `.csv`。StackOverflow 的文本标签文件第一列为节点编号，其余列为标签 id（至少 1 列）。

## 预处理产物

预处理脚本需输出以下张量文件（Torch `pt` 或 NumPy `npz`）：

- `X`: 节点特征矩阵，形状为 `(num_nodes, num_features)`。
- `Y`: 节点标签或回归目标，形状 `(num_nodes,)` 或 `(num_nodes, target_dim)`。
- `H`: 入射矩阵稀疏表示，提供 `indices`, `values`, `shape`。
- `E`: 超边属性（可选）。
- `splits`: 训练/验证/测试划分索引。

## 质量检查

- 节点索引连续、起始为 0。
- 超边基数 ≥ 2，若存在重复超边需去重或合并权重。
- 时间戳缺失时默认填充为 0。
- 对原始与派生特征执行缺失值检测与异常裁剪（如按 0.5%–99.5% 分位截断）。

## 特征标准化

确定性特征在拼接前必须执行零均值单位方差标准化并保存统计量，确保线上/离线一致。

## 版本控制

- 原始数据：以 `data/raw/<dataset>/<version>` 形式存储，并在 README 中记录来源与校验和。
- 预处理产物：使用 DVC 或 MLflow Artifact 管理，保持可追溯性。

## 数据集补充说明

| 数据集 | 目录 | 关键文件 | 推荐任务 |
| --- | --- | --- | --- |
| email-Eu-full | `data/raw/email-Eu-full/` | `*-nverts.txt`, `*-simplices.txt`, `*-times.txt`, `*-labels.npy` | 节点分类、超边预测 |
| cat-edge-DAWN | `data/raw/cat-edge-DAWN/` | `cat-edge-DAWN-simplices.txt`, 可选 `cat-edge-DAWN-node-features.npy`, `cat-edge-DAWN-node-labels.npy`, `cat-edge-DAWN-times.txt` | 低标注率节点分类、时间感知预测 |
| coauth-DBLP-full | `data/raw/coauth-DBLP-full/` | `coauth-DBLP-full-simplices.txt`, 可选 `coauth-DBLP-full-node-features.npy`, `coauth-DBLP-full-node-labels.npy` | 跨域迁移、社区检测 |
| stackoverflow-answers | `data/raw/stackoverflow-answers/` | `hyperedges-stackoverflow-answers.txt`, `node-labels-stackoverflow-answers.txt`, `label-names-stackoverflow-answers.txt` | OOD 节点分类、超边预测 |

> 注意：仓库中以 Git LFS 指针形式存储的大文件需要在本地执行 `git lfs pull` 后方可由上述脚本读取；未提供的可选文件请在配置中将对应条目设为 `null`。
