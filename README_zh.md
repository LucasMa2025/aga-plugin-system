# AGA 插件生态系统

<p align="center">
  <strong>冻结 LLM 的无损能力扩展</strong><br/>
  注意力治理 · 知识管理 · 可观测性
</p>

<p align="center">
  <img src="https://img.shields.io/badge/aga--core-v4.4.0-blue" alt="aga-core"/>
  <img src="https://img.shields.io/badge/aga--knowledge-v0.3.0-green" alt="aga-knowledge"/>
  <img src="https://img.shields.io/badge/aga--observability-v1.0.0-orange" alt="aga-observability"/>
  <img src="https://img.shields.io/badge/python-3.9+-brightgreen" alt="python"/>
  <img src="https://img.shields.io/badge/torch-2.0+-red" alt="torch"/>
  <img src="https://img.shields.io/badge/license-MIT-lightgrey" alt="license"/>
</p>

<p align="center">
  <a href="README.md">📖 English Version</a>
</p>

---

## 什么是 AGA？

**AGA（Auxiliary Governed Attention，辅助注意力治理）** 是面向冻结大语言模型的**运行时注意力治理插件**。当 LLM 在推理过程中遇到知识空白（表现为高熵/不确定性）时，AGA 自动将外部知识注入到 Transformer 的注意力层中 — **不修改任何模型参数**。

**AGA 不是 RAG，不是 LoRA，不是 Prompt Engineering。** 它在注意力层层面、推理过程中工作，由模型自身的熵信号驱动原子事实级注入。

```
Token → Transformer 层 → 自注意力 → [熵高？] → AGA 注入 → 融合输出
```

| 维度     | RAG                    | LoRA                 | AGA                             |
| -------- | ---------------------- | -------------------- | ------------------------------- |
| 介入时机 | 推理前（拼接 context） | 训练时（微调参数）   | 推理中（注意力层实时注入）      |
| 修改模型 | 否                     | 是（增加适配器权重） | 否（纯 hook，零参数修改）       |
| 知识粒度 | 文档/段落级            | 全局知识             | 原子事实级（10-50 tokens/slot） |
| 动态性   | 静态检索               | 需重新训练           | 实时增删，秒级生效              |
| 决策依据 | 用户查询相似度         | 无（始终生效）       | 模型内部熵信号（自适应）        |

**适用场景：**

- **垂直领域私有知识系统** — 医疗、法律、金融等领域的专业知识实时注入
- **动态知识更新场景** — 新闻、政策、产品信息等需要实时更新的知识
- **多租户知识隔离** — 不同用户/租户拥有独立的知识空间
- **模型知识补丁** — 快速修复模型的事实性错误，无需重新训练
- **流式生成场景** — 在 token-by-token 生成过程中持续注入知识

---

## 生态系统架构

AGA 采用**三包分离**架构，`aga-core` 是唯一必需包：

```
+-------------------------------------------------------------+
|                      AGA 生态系统                             |
|                                                              |
|  +---------------+                                           |
|  |   aga-core    | ← 必需                                    |
|  |   v4.4.0      |    pip install aga-core                   |
|  |               |    唯一依赖: torch>=2.0.0                  |
|  |                                                           |
|  |  • 注意力治理引擎                                          |
|  |  • 三段式熵门控                                            |
|  |  • 瓶颈 KV 注入                                           |
|  |  • GPU 常驻 KVStore                                       |
|  |  • BaseRetriever 标准协议                                  |
|  |  • 流式生成支持                                            |
|  |  • HuggingFace + vLLM 适配器                               |
|  +-------+-------+                                           |
|          |                                                   |
|  +-------v-------+  +----------------------+                 |
|  | aga-knowledge |  |  aga-observability   | ← 可选           |
|  |   v0.3.0      |  |     v1.0.0           |                 |
|  |               |  |                      |                 |
|  | • 知识管理    |  | • Prometheus 指标    |                 |
|  | • Portal API  |  | • Grafana 仪表盘    |                 |
|  | • 持久化存储  |  | • SLO/SLI 告警      |                 |
|  | • 混合检索    |  | • 结构化日志        |                 |
|  | • 文档分片    |  | • 审计持久化        |                 |
|  |               |  | • 健康检查          |                 |
|  +---------------+  +----------------------+                 |
+-------------------------------------------------------------+
```

---

## 快速开始

### 3 行集成（仅 aga-core）

```python
from aga import AGAPlugin, AGAConfig

plugin = AGAPlugin(AGAConfig(hidden_dim=4096))
plugin.attach(model)                    # 挂载到任意 HuggingFace 模型
output = model.generate(input_ids)      # AGA 自动工作
```

### 知识注册

```python
import torch

# 注册知识（pinned=True 保护核心知识不被淘汰）
plugin.register(
    id="fact_001",
    key=torch.randn(64),       # [bottleneck_dim] 检索键
    value=torch.randn(4096),   # [hidden_dim] 知识向量
    reliability=0.95,
    pinned=True,
    metadata={"source": "medical_kb", "namespace": "cardiology"}
)

# 批量注册
plugin.register_batch([
    {"id": "fact_002", "key": k2, "value": v2, "reliability": 0.9},
    {"id": "fact_003", "key": k3, "value": v3, "reliability": 0.85},
])
```

### 流式生成注入

```python
plugin = AGAPlugin(AGAConfig(hidden_dim=4096))
plugin.attach(model)

streamer = plugin.create_streaming_session()
for token_output in model_generate_stream(input_ids):
    diag = streamer.get_step_diagnostics()
    if diag["aga_applied"]:
        print(f"Token {diag['step']}: AGA 注入, gate={diag['gate_mean']:.4f}")

summary = streamer.get_session_summary()
print(f"总 token 数: {summary['total_steps']}, 注入率: {summary['injection_rate']:.2%}")
```

### 外部召回器集成

```python
from aga import AGAPlugin, AGAConfig
from aga.retriever.base import BaseRetriever, RetrievalQuery, RetrievalResult

# 实现自定义召回器（如基于 Chroma、Milvus 等）
class MyRetriever(BaseRetriever):
    def retrieve(self, query: RetrievalQuery) -> list:
        return [RetrievalResult(id="doc_1", key=k, value=v, score=0.95)]

plugin = AGAPlugin(AGAConfig(hidden_dim=4096), retriever=MyRetriever())
plugin.attach(model)
# AGA 在高熵时自动调用召回器获取知识
```

### 全栈集成（aga-core + aga-knowledge + aga-observability）

```python
from aga import AGAPlugin, AGAConfig
from aga_knowledge import KnowledgeManager, AGACoreAlignment
from aga_knowledge.config import PortalConfig
from aga_knowledge.encoder import create_encoder, EncoderConfig
from aga_knowledge.retriever import KnowledgeRetriever

# 1. 对齐配置（aga-core 与 aga-knowledge 的桥梁）
alignment = AGACoreAlignment(
    hidden_dim=4096, bottleneck_dim=64,
    key_norm_target=5.0, value_norm_target=3.0,
)

# 2. 知识管理
manager = KnowledgeManager(PortalConfig.for_development())
await manager.start()

# 3. 编码器 + 混合召回器
encoder = create_encoder(EncoderConfig.from_alignment(alignment))
retriever = KnowledgeRetriever(
    manager=manager, encoder=encoder,
    alignment=alignment, namespace="default",
    index_backend="hnsw", bm25_enabled=True,
)

# 4. 带可观测性的插件
config = AGAConfig(
    hidden_dim=4096, bottleneck_dim=64,
    observability_enabled=True,  # 自动检测 aga-observability
    prometheus_enabled=True,
    prometheus_port=9090,
)
plugin = AGAPlugin(config, retriever=retriever)
plugin.attach(model)
```

---

## 安装

### 从源码安装（单仓库）

```bash
cd AGAPlugin

# 仅安装 aga-core（唯一依赖: torch）
pip install -e .

# 安装 aga-knowledge（含所有可选依赖）
pip install -e ./aga_knowledge[all]

# 安装 aga-observability（含 Prometheus 支持）
pip install -e ./aga_observability[full]

# 安装全部
pip install -e .[all]
pip install -e ./aga_knowledge[all]
pip install -e ./aga_observability[full]
```

### 从 PyPI 安装（发布后）

```bash
pip install aga-core                           # 仅核心
pip install aga-core[yaml]                     # 核心 + YAML 配置支持
pip install aga-knowledge[all]                 # 知识管理
pip install aga-observability[full]            # 可观测性
pip install aga-core[knowledge,observability]  # 全栈
```

### 系统要求

- Python >= 3.9
- PyTorch >= 2.0.0
- CUDA（推荐，CPU 也可运行但性能较低）

---

## 各包核心特性

### aga-core v4.4.0 — 注意力治理引擎

> 📖 [详细文档 (中文)](aga/README_zh.md) · [Detailed README (English)](aga/README_en.md)

| 类别             | 特性                                                                        |
| ---------------- | --------------------------------------------------------------------------- |
| **集成**         | 3 行集成：`AGAPlugin(config).attach(model)`                                |
| **集成**         | `from_config()` — YAML/Dict 配置驱动创建                                   |
| **熵门控**       | 三段式门控：Gate-0（命名空间）→ Gate-1（熵）→ Gate-2（置信度）              |
| **熵门控**       | 低熵 token 的 Early Exit 优化                                               |
| **注入**         | 瓶颈注意力：Query 投影 → Top-K 路由 → Value 投影                           |
| **注入**         | 注入延迟 < 0.1ms / forward pass                                            |
| **KVStore**      | GPU 预分配常驻内存，256 slots ≈ 2MB VRAM                                   |
| **KVStore**      | LRU 淘汰 + 知识锁定（`pin`/`unpin`）+ 命名空间隔离                         |
| **流式生成**     | `create_streaming_session()` — 逐 token 诊断                               |
| **流式生成**     | 动态知识热更新 `update_knowledge()`                                         |
| **召回器**       | `BaseRetriever` 标准协议 — 可插拔外部检索                                   |
| **召回器**       | 内置 `NullRetriever` 和 `KVStoreRetriever`                                 |
| **Slot 治理**    | 预算控制、语义去重、冷却期、稳定性检测                                      |
| **适配器**       | HuggingFace（LLaMA/Qwen/Mistral/GPT-2/Phi/Gemma/Falcon）                   |
| **适配器**       | vLLM（无需 fork）+ IBM vLLM-Hook 兼容                                       |
| **分布式**       | `TPManager` — 张量并行 KVStore 广播                                         |
| **安全**         | Fail-Open — 异常永不阻断推理                                               |
| **埋点**         | EventBus + ForwardMetrics（P50/P95/P99）+ AuditLog                          |

### aga-knowledge v0.3.0 — 知识管理系统

> 📖 [详细文档 (中文)](aga_knowledge/README_zh.md) · [Detailed README (English)](aga_knowledge/README_en.md)

| 类别             | 特性                                                                        |
| ---------------- | --------------------------------------------------------------------------- |
| **知识注册**     | Portal REST API（FastAPI）— 完整 CRUD + 图片资产服务                        |
| **知识注册**     | 明文 `condition/decision` 对 — 人类可读的知识格式                           |
| **知识注册**     | 命名空间隔离、生命周期管理、信任层级                                        |
| **持久化**       | 4 种后端：内存、SQLite、PostgreSQL（asyncpg）、Redis（aioredis）            |
| **持久化**       | 所有 CRUD 操作均有审计记录                                                  |
| **同步**         | Redis Pub/Sub 跨实例实时知识同步                                            |
| **同步**         | 按需全量同步、心跳检测                                                      |
| **编码**         | `SentenceTransformerEncoder` — 语义嵌入 + 投影层                            |
| **编码**         | `AGACoreAlignment` — 与 aga-core 的维度/范数强制对齐                        |
| **检索**         | HNSW 稠密检索（hnswlib ANN）+ BM25 稀疏检索                                |
| **检索**         | RRF（互惠排序融合）混合结果                                                 |
| **检索**         | 增量索引更新、自动刷新、线程安全、Fail-Open                                 |
| **文档分片**     | 5 种策略：FixedSize、Sentence、Semantic、SlidingWindow、Document            |
| **文档分片**     | `DocumentChunker`（Markdown 感知）+ `ConditionGenerator` + `ImageHandler`   |
| **版本控制**     | 完整版本历史、回滚、差异比较、变更审计                                      |
| **压缩**         | zlib / LZ4 / Zstd，带 LRU 解压缓存                                         |

### aga-observability v1.0.0 — 生产级可观测性

> 📖 [详细文档 (中文)](aga_observability/README_zh.md) · [Detailed README (English)](aga_observability/README_en.md)

| 类别             | 特性                                                                        |
| ---------------- | --------------------------------------------------------------------------- |
| **Prometheus**   | 15+ 指标：计数器、直方图、仪表盘（forward、retrieval、audit 等）            |
| **Prometheus**   | HTTP 端点 `:9090` 供 Prometheus 抓取                                        |
| **Grafana**      | 自动生成 5 组面板 JSON（概览、forward、门控、召回、审计）                    |
| **告警**         | SLO/SLI 规则：延迟 P99、利用率、Slot 抖动                                  |
| **告警**         | 通道：日志输出、Webhook（HTTP POST）、自定义回调                            |
| **日志**         | 结构化 JSON/Text 格式，支持文件轮转                                         |
| **审计**         | 持久化审计追踪 — JSONL 或 SQLite，支持保留策略                              |
| **健康检查**     | HTTP 端点 `GET /health`，支持 Kubernetes 存活/就绪探针                      |
| **设计原则**     | 零侵入 — EventBus 订阅，不修改 aga-core 源码                               |
| **设计原则**     | 自动集成 — `pip install` 后自动激活                                         |
| **设计原则**     | Fail-Open — 可观测性故障不影响 LLM 推理                                    |

---

## 单仓库结构

```
AGAPlugin/
├── aga/                    ← aga-core（必需）
│   ├── plugin.py           # AGAPlugin — 3 行集成入口
│   ├── config.py           # AGAConfig — 完整外部化配置
│   ├── kv_store.py         # GPU 常驻 KV 存储（LRU + 锁定）
│   ├── streaming.py        # StreamingSession — 逐 token 诊断
│   ├── distributed.py      # TPManager — 张量并行
│   ├── gate/               # 三段式熵门控 + 衰减
│   ├── operator/           # 瓶颈注入算子
│   ├── retriever/          # BaseRetriever 协议 + 内置实现
│   ├── adapter/            # HuggingFace / vLLM 适配器
│   └── instrumentation/    # EventBus、ForwardMetrics、AuditLog
│
├── aga_knowledge/          ← aga-knowledge（可选）
│   ├── portal/             # FastAPI REST API + 资产服务
│   ├── persistence/        # 内存 / SQLite / PostgreSQL / Redis
│   ├── encoder/            # 文本→向量（SentenceTransformer）
│   ├── retriever/          # HNSW + BM25 + RRF 混合检索
│   ├── chunker/            # 文档 → 知识片段
│   ├── alignment.py        # AGACoreAlignment
│   ├── sync/               # Redis Pub/Sub 同步
│   └── config_adapter/     # aga-core ↔ aga-knowledge 配置桥接
│
├── aga_observability/      ← aga-observability（可选）
│   ├── prometheus_exporter.py  # Prometheus 指标导出
│   ├── grafana_dashboard.py    # 自动生成 Grafana 仪表盘
│   ├── alert_manager.py        # SLO/SLI 告警引擎
│   ├── log_exporter.py         # 结构化日志导出
│   ├── audit_storage.py        # 持久化审计追踪
│   ├── health.py               # 健康检查 HTTP 端点
│   └── stack.py                # ObservabilityStack 编排器
│
├── configs/                # 示例配置文件
├── tests/                  # 所有单元测试
└── pyproject.toml          # 根包（aga-core）
```

---

## 配置

示例配置文件位于 `configs/` 目录：

| 文件                                                         | 用途                                           | 使用方                 |
| ------------------------------------------------------------ | ---------------------------------------------- | ---------------------- |
| [`configs/runtime_config.yaml`](configs/runtime_config.yaml) | AGA 运行时：熵门控、衰减、设备、召回器         | `aga-core` AGAPlugin   |
| [`configs/portal_config.yaml`](configs/portal_config.yaml)   | 知识 Portal：持久化、消息队列、治理             | `aga-knowledge` Portal |

```python
# aga-core：从 YAML 加载
plugin = AGAPlugin.from_config("configs/runtime_config.yaml")

# aga-knowledge：加载 Portal 配置
from aga_knowledge.config import PortalConfig
config = PortalConfig.from_yaml("configs/portal_config.yaml")
```

---

## 文档

### aga-core

| 文档                                                          | 语言    |
| ------------------------------------------------------------- | ------- |
| [README (English)](aga/README_en.md)                          | English |
| [README (中文)](aga/README_zh.md)                             | 中文    |
| [Product Documentation (English)](aga/docs/product_doc_en.md) | English |
| [产品说明书 (中文)](aga/docs/product_doc_zh.md)               | 中文    |
| [User Manual (English)](aga/docs/user_manual_en.md)           | English |
| [用户手册 (中文)](aga/docs/user_manual_zh.md)                 | 中文    |

### aga-knowledge

| 文档                                                                    | 语言    |
| ----------------------------------------------------------------------- | ------- |
| [README (English)](aga_knowledge/README_en.md)                          | English |
| [README (中文)](aga_knowledge/README_zh.md)                             | 中文    |
| [Product Documentation (English)](aga_knowledge/docs/product_doc_en.md) | English |
| [产品说明书 (中文)](aga_knowledge/docs/product_doc_zh.md)               | 中文    |
| [User Manual (English)](aga_knowledge/docs/user_manual_en.md)           | English |
| [用户手册 (中文)](aga_knowledge/docs/user_manual_zh.md)                 | 中文    |

### aga-observability

| 文档                                                              | 语言    |
| ----------------------------------------------------------------- | ------- |
| [README (English)](aga_observability/README_en.md)                | English |
| [README (中文)](aga_observability/README_zh.md)                   | 中文    |
| [User Manual (English)](aga_observability/docs/user_manual_en.md) | English |
| [用户手册 (中文)](aga_observability/docs/user_manual_zh.md)       | 中文    |

---

## 测试

```bash
# 全部测试
python -m pytest tests/ -v

# aga-core 测试
python -m pytest tests/test_aga/ -v

# aga-knowledge 测试
python -m pytest tests/test_knowledge/ -v

# aga-observability 测试
python -m pytest tests/test_observability/ -v
```

---

## 路线图

| 包                    | 当前版本                                                   | 下一个里程碑                                    |
| --------------------- | ---------------------------------------------------------- | ----------------------------------------------- |
| **aga-core**          | v4.4.0 — 召回器协议、Slot 治理、流式生成                   | v5.0 — 分层知识、INT8 KVStore、自适应瓶颈       |
| **aga-knowledge**     | v0.3.0 — HNSW+BM25+RRF、DocumentChunker、AGACoreAlignment | v0.4.x — 对比学习微调、分布式编码器、Prometheus |
| **aga-observability** | v1.0.0 — Prometheus、Grafana、告警、审计、健康检查         | v1.1.0 — OpenTelemetry 链路追踪、分布式聚合     |

---

## 许可证

MIT License

Copyright (c) 2024-2026 AGA Team

---

<p align="center">
  <strong>AGA — 让每一次推理都充满知识的力量</strong>
</p>
