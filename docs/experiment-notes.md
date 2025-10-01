# 实验说明：确定性特征消融

本说明总结 DF-HGNN 在禁用确定性特征时的设置与默认实验的差异，便于复现实验与撰写报告。

## 配置入口

- **默认实验**：`configs/experiment/baseline.yaml`（继承 `configs/default.yaml`，其中 `features.deterministic.enabled` 默认为 `true`）。
- **确定性特征消融**：`configs/experiment/baseline_no_deterministic.yaml` 将 `features.deterministic.enabled` 显式设置为 `false`，用于评估模型在仅使用原始节点特征时的表现。

## 主要差异

| 组件 | 默认 DF-HGNN | 消融设定 |
| ---- | ------------- | -------- |
| 确定性特征管道 | 计算结构/谱/时间统计，并可复用缓存 | 完全跳过计算，传入空特征占位符 |
| 模型输入维度 (`det_dim`) | 由特征组合结果决定（通常 > 0） | 固定为 0，模型仅依赖原始节点特征 |
| 训练脚本行为 | 实例化 `DeterministicFeatureBank`，可能触发缓存加载 | 直接创建与节点特征 dtype 相同的零列张量，无额外 I/O |
| 报告记录 | 作为基准对照 | 标注为 “no deterministic features” 或同义描述 |

## 报告建议

- 运行两种配置时请在实验日志与报告标题中注明 `deterministic features: enabled/disabled`，便于对齐指标。
- 若数据集缺乏原始节点特征，则禁用确定性特征会导致模型输入维度为 0，训练脚本会报错提醒需至少一种特征来源。
- 缓存目录仍可复用，但消融模式不会生成或读取确定性特征缓存，避免误用过期结果。
