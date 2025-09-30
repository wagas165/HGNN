# DF-HGNN 系统架构

本文档概述 DF-HGNN 企业级实现的端到端架构组件，包括数据治理、特征工程、模型训练、推理服务与运维监控。详细流程可参考 `docs/data-spec.md` 与 `docs/model-card.md`。

## 架构分层

1. **数据治理层**：负责从原始超图数据集中抽取节点、超边与时间戳信息，完成数据质检与版本化管理。通过 `src/data/loaders` 中的 Loader 统一读取格式，并将处理后的张量缓存于数据湖或对象存储。
2. **特征工程层**：`src/features` 模块实现确定性结构/谱特征提取与标准化，支持特征缓存和在线实时计算。该层保证原始特征与确定性特征在维度和尺度上的一致性。
3. **模型训练层**：`src/models` 和 `src/training` 提供 DF-HGNN 主干网络、门控融合、两阶段优化器切换、评估指标与训练回调。Trainer 支持单机多卡、混合精度以及断点续训。
4. **评估与解释层**：`src/evaluation` 包含指标计算、可解释性分析、报告生成。通过统计门控权重和对齐损失，实现可追溯的模型行为分析。
5. **部署与服务层**：`src/serving` 提供 REST/gRPC 推理服务框架，并与 `src/pipelines` 的监控模块协同，实现上线后的性能监控与数据漂移预警。

## 运行流程

1. 通过 `scripts/preprocess_email_eu_full.py` 解析原始数据生成张量化缓存。
2. 使用 `scripts/train_df_hgnn.py` 指定配置（Hydra）进行训练，自动记录实验日志和模型权重。
3. 通过 `scripts/evaluate.py` 聚合多次实验结果并输出报告。
4. 若需要部署，使用 `scripts/export_onnx.py` 导出模型，再由 `src/serving/inference_service.py` 加载发布。

## 依赖与环境

- Python 3.9+
- PyTorch 2.1+
- 详见 `pyproject.toml` 依赖列表。

## 安全与合规

- 模型卡需在部署前更新，说明数据来源、性能、潜在偏差与使用限制。
- 所有实验需记录配置、版本、随机种子以满足审计要求。
