# DF-HGNN 项目任务与技术路线规划

本文档基于既有 DF-HGNN 企业级方案骨架，并结合当前已上传的四个超图数据集（已确认的 `email-Eu-full` 及待集成的另外三个数据包）进行任务定义、技术路径梳理与代码框架规划。后续在其余数据集解压完成后，可按本规划快速落地实验与部署工作。

## 1. 数据集概览与任务映射

| 数据集 | 数据特征 | 推荐核心任务 | 辅助任务/评估 | 备注 |
| --- | --- | --- | --- | --- |
| `email-Eu-full` | 时间戳超边（邮件会话），`*-nverts/simplices/times` 三文件格式 | 1. 节点动态分类（基于外部标签，如部门/角色）；<br>2. 时间敏感的超边补全/下一步交互预测 | - 超边形成时间的回归或排序；<br>- 异常交互检测 | 已解压，可直接用于流程验证；标签需与外部元数据对齐，若暂缺则可转为自监督或链接预测任务 |
| 数据集 #2 | （待解压）预期同样包含节点属性/超边列表 | 1. 节点分类（若含类别标签）；<br>2. 超边分类/标签预测 | - 低标注率实验；<br>- 特征消融 | 待确认字段后在 `configs/data/` 内新增配置 |
| 数据集 #3 | （待解压）可包含更大规模或不同领域（如问答、科研、消费行为） | 1. 节点/超边分类；<br>2. 链接/超边补全 | - OOD 划分；<br>- 时间窗口泛化 | 用于检验跨领域泛化能力 |
| 数据集 #4 | （待解压）可能包含数值标签或回归目标 | 1. 节点属性回归或风险评分预测 | - 对抗扰动/噪声鲁棒性；<br>- 解释性案例分析 | 若无标签，可通过自构造任务（如超边大小预测）补足 |

> ⚠️ 由于其余三个数据包尚未解压，表中任务为规划建议。待解析 `DATA-DESCRIPTION` 或元数据文件后，需在 `docs/data-spec.md` 中补充字段定义，并在 `configs/data/` 下新增对应 YAML 描述训练/评估划分。

### 1.1 任务优先级

1. **基础验证阶段**（Sprint 1）
   - 使用 `email-Eu-full` 构建端到端流程，完成时间感知超边补全与节点分类两个任务的 baseline。
   - 验证确定性特征管线（结构特征 + 谱编码）是否稳定；完成 Adam→L-BFGS 两阶段优化的实现与测试。

2. **多数据集扩展阶段**（Sprint 2）
   - 针对其余三个数据集，根据标签类型定制任务（分类/回归/生成式预测）。
   - 补充低标注率、跨域迁移与 OOD 场景实验。

3. **稳健性与可解释性阶段**（Sprint 3）
   - 门控权重分析、确定性特征重要性排序。
   - 噪声注入、超边 Drop/Mask、对抗扰动实验，形成稳定性报告。

## 2. 技术路线

整体流程依循 "数据治理 → 确定性特征注入 → DF-HGNN 建模 → 评估与部署" 四段：

1. **数据治理与预处理**
   - 在 `data/raw/` 下存放原始超图数据，通过 `scripts/preprocess_*.py` 将 `nverts/simplices/times` 转换为稀疏入射矩阵、节点/超边属性张量与任务标签。
   - 若缺失标签，优先考虑：a) 外部元数据匹配；b) 构造自监督任务（如超边大小预测）；c) 使用负采样生成链接预测数据。
   - 使用 DVC 或湖仓系统追踪数据版本；在 `docs/data-spec.md` 记录字段及质量检查。

2. **确定性特征构建**
   - `src/features/deterministic_bank.py` 提供结构与（可选）动力学特征计算：节点超度、超边基数统计、谱位置编码、Hodge 指标等。
   - 对于时间序列型任务（如 `email-Eu-full`），增量计算局部传播率、事件强度等时间特征。
   - 所有特征在拼接前执行零均值/单位方差标准化与分位裁剪，避免尺度偏差。

3. **模型训练与优化**
   - 核心模型 `DFHGNN` 结合门控融合 + 超图卷积（消息传递/谱滤波两种模式），通过对齐正则和门控稀疏惩罚提升可解释性。
   - 训练策略：先使用 Adam 进行快速收敛，再切换 L-BFGS 精细优化；必要时冻结 BatchNorm 统计，使用小批量。
   - 支持混合精度与多卡分布式训练（DDP），并通过回调记录指标、门控行为、特征贡献度。

4. **评估与部署**
   - `src/evaluation/` 汇总分类/回归/链接预测指标，提供 mean±std、置信区间、成对 t 检验。
   - 通过 `scripts/evaluate.py` 与 `src/evaluation/reporting.py` 输出实验报告、可解释性图表。
   - 模型导出为 TorchScript/ONNX，集成 FastAPI/Triton 推理服务；监控漂移与性能。

## 3. 代码框架落地

### 3.1 目录结构（增量）

```
df_hgnn/
├── configs/
│   ├── data/
│   │   ├── email_eu_full.yaml       # 已知数据集配置
│   │   ├── dataset2.yaml            # 待定，占位，解压后补充
│   │   ├── dataset3.yaml
│   │   └── dataset4.yaml
│   ├── experiment/
│   │   ├── node_classification.yaml
│   │   ├── hyperedge_completion.yaml
│   │   └── regression.yaml
│   └── default.yaml
├── data/
│   ├── raw/
│   │   └── email-Eu-full/...
│   └── processed/
├── docs/
│   ├── task_tech_plan.md            # 本文档
│   ├── data-spec.md                 # 待补充
│   └── architecture.md              # 企业级整体架构说明
├── scripts/
│   ├── preprocess_email_eu_full.py
│   ├── preprocess_dataset2.py
│   ├── train_df_hgnn.py
│   ├── evaluate.py
│   └── export_onnx.py
├── src/
│   ├── data/
│   │   ├── loaders/
│   │   │   ├── email_eu_full.py
│   │   │   └── datasetX.py
│   │   └── transforms/
│   ├── features/
│   │   ├── deterministic_bank.py
│   │   └── preprocessors.py
│   ├── models/
│   │   ├── df_hgnn.py
│   │   ├── layers/
│   │   └── registry.py
│   ├── training/
│   ├── evaluation/
│   ├── serving/
│   └── pipelines/
└── tests/
    ├── unit/
    ├── integration/
    └── regression/
```

### 3.2 关键开发里程碑

- **M1：数据 Loader 与配置**
  - 解析 `email-Eu-full`，实现 `HypergraphDataset` 基类与数据注册。
  - 模板化预处理脚本，后续数据集仅需提供字段映射。

- **M2：确定性特征组件**
  - 完成结构特征、谱特征实现，支持批量缓存。
  - 实现动力学特征计算接口（允许无时间数据的场景关闭）。

- **M3：模型与训练框架**
  - DF-HGNN 主干、门控融合、正则项与两阶段优化调度。
  - Trainer 支持多任务训练、消融实验自动化。

- **M4：评估与可解释性**
  - 指标计算、置信区间、t-test；特征重要性与门控可视化。
  - 生成实验报告及模型卡初稿。

- **M5：部署与监控（可选）**
  - 推理服务、批量离线任务、模型导出及版本管理。
  - 监控指标（性能、门控漂移、输入分布）。

## 4. 下一步行动项

1. 解压并登记另外三个数据集：
   - 将原始文件放入 `data/raw/dataset_name/`。
   - 更新 `docs/data-spec.md` 与 `configs/data/dataset_name.yaml`。

2. 在 `scripts/preprocess_email_eu_full.py` 中实现首个数据流水线，并产出小规模样例供 `tests/` 使用。

3. 初始化 `src/models/df_hgnn.py` 与 `src/features/deterministic_bank.py`，构建最小可运行 Demo，验证与规划一致。

4. 规划 CI 流程：`ruff`/`black`/`mypy` + `pytest`（利用样例数据），确保后续提交质量可控。

5. 结合业务需求，与产品/合规团队确认各任务输出的解释性要求与上线场景。

---

此文档将作为项目 Kick-off 的执行蓝图。随着数据与需求明确，可在对应章节追加细节，并在代码库中逐步落地所述模块。
