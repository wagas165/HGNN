# 模型卡：DF-HGNN

## 概述

- **模型名称**：Deterministic-Feature–Augmented Hypergraph Neural Network (DF-HGNN)
- **版本**：0.1.0
- **任务类型**：节点分类、超边预测、风险回归（根据配置）

## 数据

- **主要数据集**：email-Eu-full（EU 研究机构内部邮件）。
- **数据来源**：https://snap.stanford.edu/data/email-Eu-core.html
- **处理流程**：`scripts/preprocess_email_eu_full.py`

## 性能指标

> 训练完成后请更新下列指标（mean±std）：

| 任务 | 指标 | DF-HGNN | 最佳基线 |
| ---- | ---- | ------- | -------- |
| 节点分类 | Macro-F1 | TBD | TBD |
| 节点分类 | ROC-AUC | TBD | TBD |

## 训练配置

- 优化器：Adam (前期) → L-BFGS (后期)
- 特征：原始节点特征（若有） + 结构/谱确定性特征
- 正则项：对齐损失、门控稀疏

## 伦理与偏差

- 邮件数据包含个人通信信息，若用于生产需确保匿名化及合规。
- 建议定期审查模型输出，防止对特定群体的不公平影响。

## 使用限制

- 仅限研究与内部评估，未获得外部部署授权。
- 不适合处理包含敏感个人信息的未脱敏数据。

## 维护者

- 项目负责人：待定
- 联系方式：research@example.com
