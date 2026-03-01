# AGA Plugin Ecosystem

<p align="center">
  <strong>Lossless Capability Extension for Frozen LLMs</strong><br/>
  Attention Governance · Knowledge Management · Observability
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
  <a href="README_zh.md">📖 中文版</a>
</p>

---

## What is AGA?

**AGA (Auxiliary Governed Attention)** is a **runtime attention governance plugin** for frozen Large Language Models. When an LLM encounters knowledge gaps during inference (manifested as high entropy / uncertainty), AGA automatically injects external knowledge into the Transformer's attention layers — **without modifying any model parameters**.

**AGA is not RAG, not LoRA, not Prompt Engineering.** It operates at the attention layer level, mid-inference, providing atomic-fact-level injection driven by the model's own entropy signals.

```
Token → Transformer Layer → Self-Attention → [Entropy High?] → AGA Injection → Fused Output
```

| Dimension       | RAG                        | LoRA                       | AGA                                         |
| --------------- | -------------------------- | -------------------------- | -------------------------------------------- |
| Intervention    | Pre-inference (concat ctx) | Training (fine-tune)       | Mid-inference (attention layer injection)    |
| Modifies Model  | No                         | Yes (adapter weights)      | No (pure hooks, zero param modification)     |
| Knowledge Grain | Document / paragraph       | Global knowledge           | Atomic facts (10–50 tokens/slot)             |
| Dynamism        | Static retrieval           | Requires retraining        | Real-time add/remove, seconds to take effect |
| Decision Basis  | Query similarity           | None (always active)       | Model internal entropy signal (adaptive)     |

**Use Cases:**

- **Vertical domain private knowledge** — Medical, legal, financial real-time knowledge injection
- **Dynamic knowledge updates** — News, policies, product info requiring real-time updates
- **Multi-tenant knowledge isolation** — Independent knowledge spaces per user/tenant
- **Model knowledge patching** — Quickly fix factual errors without retraining
- **Streaming generation** — Continuous knowledge injection during token-by-token generation

---

## Ecosystem Architecture

AGA adopts a **three-package separation** architecture. Only `aga-core` is required:

```
+-------------------------------------------------------------+
|                      AGA Ecosystem                           |
|                                                              |
|  +---------------+                                           |
|  |   aga-core    | ← Required                                |
|  |   v4.4.0      |    pip install aga-core                   |
|  |               |    Only dependency: torch>=2.0.0          |
|  |  • Attention governance engine                            |
|  |  • 3-stage entropy gating                                 |
|  |  • Bottleneck KV injection                                |
|  |  • GPU-resident KVStore                                   |
|  |  • BaseRetriever protocol                                 |
|  |  • Streaming generation                                   |
|  |  • HuggingFace + vLLM adapters                            |
|  +-------+-------+                                           |
|          |                                                   |
|  +-------v-------+  +----------------------+                 |
|  | aga-knowledge |  |  aga-observability   | ← Optional      |
|  |   v0.3.0      |  |     v1.0.0           |                 |
|  |               |  |                      |                 |
|  | • Knowledge   |  | • Prometheus metrics |                 |
|  |   management  |  | • Grafana dashboards |                 |
|  | • Portal API  |  | • SLO/SLI alerting  |                 |
|  | • Persistence |  | • Structured logging |                 |
|  | • Hybrid      |  | • Audit persistence |                 |
|  |   retrieval   |  | • Health checking   |                 |
|  | • Doc chunking|  |                      |                 |
|  +---------------+  +----------------------+                 |
+-------------------------------------------------------------+
```

---

## Quick Start

### 3-Line Integration (aga-core only)

```python
from aga import AGAPlugin, AGAConfig

plugin = AGAPlugin(AGAConfig(hidden_dim=4096))
plugin.attach(model)                    # Hook into any HuggingFace model
output = model.generate(input_ids)      # AGA works automatically
```

### Knowledge Registration

```python
import torch

# Register knowledge (pinned=True protects core knowledge from eviction)
plugin.register(
    id="fact_001",
    key=torch.randn(64),       # [bottleneck_dim] retrieval key
    value=torch.randn(4096),   # [hidden_dim] knowledge vector
    reliability=0.95,
    pinned=True,
    metadata={"source": "medical_kb", "namespace": "cardiology"}
)

# Batch register
plugin.register_batch([
    {"id": "fact_002", "key": k2, "value": v2, "reliability": 0.9},
    {"id": "fact_003", "key": k3, "value": v3, "reliability": 0.85},
])
```

### Streaming Generation

```python
plugin = AGAPlugin(AGAConfig(hidden_dim=4096))
plugin.attach(model)

streamer = plugin.create_streaming_session()
for token_output in model_generate_stream(input_ids):
    diag = streamer.get_step_diagnostics()
    if diag["aga_applied"]:
        print(f"Token {diag['step']}: AGA injected, gate={diag['gate_mean']:.4f}")

summary = streamer.get_session_summary()
print(f"Total tokens: {summary['total_steps']}, Injection rate: {summary['injection_rate']:.2%}")
```

### External Retriever Integration

```python
from aga import AGAPlugin, AGAConfig
from aga.retriever.base import BaseRetriever, RetrievalQuery, RetrievalResult

# Implement custom retriever (e.g., backed by Chroma, Milvus, etc.)
class MyRetriever(BaseRetriever):
    def retrieve(self, query: RetrievalQuery) -> list:
        return [RetrievalResult(id="doc_1", key=k, value=v, score=0.95)]

plugin = AGAPlugin(AGAConfig(hidden_dim=4096), retriever=MyRetriever())
plugin.attach(model)
# AGA automatically calls retriever on high entropy
```

### Full Stack (aga-core + aga-knowledge + aga-observability)

```python
from aga import AGAPlugin, AGAConfig
from aga_knowledge import KnowledgeManager, AGACoreAlignment
from aga_knowledge.config import PortalConfig
from aga_knowledge.encoder import create_encoder, EncoderConfig
from aga_knowledge.retriever import KnowledgeRetriever

# 1. Alignment (bridge between aga-core and aga-knowledge)
alignment = AGACoreAlignment(
    hidden_dim=4096, bottleneck_dim=64,
    key_norm_target=5.0, value_norm_target=3.0,
)

# 2. Knowledge Management
manager = KnowledgeManager(PortalConfig.for_development())
await manager.start()

# 3. Encoder + Hybrid Retriever
encoder = create_encoder(EncoderConfig.from_alignment(alignment))
retriever = KnowledgeRetriever(
    manager=manager, encoder=encoder,
    alignment=alignment, namespace="default",
    index_backend="hnsw", bm25_enabled=True,
)

# 4. Plugin with observability
config = AGAConfig(
    hidden_dim=4096, bottleneck_dim=64,
    observability_enabled=True,  # auto-detects aga-observability
    prometheus_enabled=True,
    prometheus_port=9090,
)
plugin = AGAPlugin(config, retriever=retriever)
plugin.attach(model)
```

---

## Installation

### From Source (Mono-Repo)

```bash
cd AGAPlugin

# Install aga-core only (only dependency: torch)
pip install -e .

# Install aga-knowledge (with all optional deps)
pip install -e ./aga_knowledge[all]

# Install aga-observability (with Prometheus support)
pip install -e ./aga_observability[full]

# Install everything
pip install -e .[all]
pip install -e ./aga_knowledge[all]
pip install -e ./aga_observability[full]
```

### From PyPI (when published)

```bash
pip install aga-core                           # Core only
pip install aga-core[yaml]                     # Core + YAML config support
pip install aga-knowledge[all]                 # Knowledge management
pip install aga-observability[full]            # Observability
pip install aga-core[knowledge,observability]  # Full stack
```

### System Requirements

- Python >= 3.9
- PyTorch >= 2.0.0
- CUDA (recommended; CPU also works but with lower performance)

---

## Key Features by Package

### aga-core v4.4.0 — Attention Governance Engine

> 📖 [Detailed README (English)](aga/README_en.md) · [详细文档 (中文)](aga/README_zh.md)

| Category                  | Feature                                                                     |
| ------------------------- | --------------------------------------------------------------------------- |
| **Integration**           | 3-line integration: `AGAPlugin(config).attach(model)`                       |
| **Integration**           | `from_config()` — YAML/Dict configuration-driven creation                  |
| **Entropy Gating**        | 3-stage gating: Gate-0 (namespace) → Gate-1 (entropy) → Gate-2 (confidence)|
| **Entropy Gating**        | Early Exit optimization for low-entropy tokens                              |
| **Injection**             | Bottleneck Attention: Query projection → Top-K routing → Value projection   |
| **Injection**             | Injection latency < 0.1ms per forward pass                                 |
| **KVStore**               | GPU-resident pre-allocated memory, 256 slots ≈ 2MB VRAM                    |
| **KVStore**               | LRU eviction + knowledge pinning (`pin`/`unpin`) + namespace isolation      |
| **Streaming**             | `create_streaming_session()` — per-token diagnostics during generation      |
| **Streaming**             | Dynamic knowledge hot-update via `update_knowledge()`                       |
| **Retriever**             | `BaseRetriever` standard protocol — pluggable external retrieval            |
| **Retriever**             | Built-in `NullRetriever` and `KVStoreRetriever`                             |
| **Slot Governance**       | Budget control, semantic dedup, cooldown, stability detection               |
| **Adapters**              | HuggingFace (LLaMA/Qwen/Mistral/GPT-2/Phi/Gemma/Falcon)                   |
| **Adapters**              | vLLM (no fork required) + IBM vLLM-Hook compatibility                       |
| **Distributed**           | `TPManager` — Tensor Parallelism KVStore broadcast                          |
| **Safety**                | Fail-Open — exceptions never block inference                                |
| **Instrumentation**       | EventBus + ForwardMetrics (P50/P95/P99) + AuditLog                          |

### aga-knowledge v0.3.0 — Knowledge Management System

> 📖 [Detailed README (English)](aga_knowledge/README_en.md) · [详细文档 (中文)](aga_knowledge/README_zh.md)

| Category                  | Feature                                                                     |
| ------------------------- | --------------------------------------------------------------------------- |
| **Knowledge Registration**| Portal REST API (FastAPI) — full CRUD + image asset serving                 |
| **Knowledge Registration**| Plaintext `condition/decision` pairs — human-readable knowledge format      |
| **Knowledge Registration**| Namespace isolation, lifecycle management, trust tiers                      |
| **Persistence**           | 4 backends: Memory, SQLite, PostgreSQL (asyncpg), Redis (aioredis)          |
| **Persistence**           | Audit logging for all CRUD operations                                       |
| **Synchronization**       | Redis Pub/Sub cross-instance real-time knowledge sync                       |
| **Synchronization**       | Full sync on demand, heartbeat detection                                    |
| **Encoding**              | `SentenceTransformerEncoder` — semantic embedding + projection layer        |
| **Encoding**              | `AGACoreAlignment` — mandatory dimension/norm alignment with aga-core       |
| **Retrieval**             | HNSW dense retrieval (hnswlib ANN) + BM25 sparse retrieval                  |
| **Retrieval**             | RRF (Reciprocal Rank Fusion) for hybrid results                             |
| **Retrieval**             | Incremental index update, auto-refresh, thread-safe, Fail-Open              |
| **Document Chunking**     | 5 strategies: FixedSize, Sentence, Semantic, SlidingWindow, Document        |
| **Document Chunking**     | `DocumentChunker` (Markdown-aware) + `ConditionGenerator` + `ImageHandler`  |
| **Versioning**            | Full version history, rollback, diff comparison, change audit               |
| **Compression**           | zlib / LZ4 / Zstd with LRU decompression cache                             |

### aga-observability v1.0.0 — Production Observability

> 📖 [Detailed README (English)](aga_observability/README_en.md) · [详细文档 (中文)](aga_observability/README_zh.md)

| Category                  | Feature                                                                     |
| ------------------------- | --------------------------------------------------------------------------- |
| **Prometheus**            | 15+ metrics: counters, histograms, gauges (forward, retrieval, audit, etc.) |
| **Prometheus**            | HTTP endpoint `:9090` for Prometheus scraping                               |
| **Grafana**               | Auto-generated 5-group dashboard JSON (overview, forward, gating, retrieval, audit) |
| **Alerting**              | SLO/SLI rules: latency P99, utilization, slot thrashing                     |
| **Alerting**              | Channels: log output, Webhook (HTTP POST), custom callbacks                 |
| **Logging**               | Structured JSON/Text format with file rotation                              |
| **Audit**                 | Persistent audit trail — JSONL or SQLite with retention policies            |
| **Health**                | HTTP endpoint `GET /health` for Kubernetes liveness/readiness probes        |
| **Design**                | Zero intrusion — EventBus subscription, no aga-core source modification     |
| **Design**                | Auto-integration — `pip install` and it activates automatically             |
| **Design**                | Fail-Open — observability failures never affect LLM inference               |

---

## Mono-Repo Structure

```
AGAPlugin/
├── aga/                    ← aga-core (required)
│   ├── plugin.py           # AGAPlugin — 3-line integration entry
│   ├── config.py           # AGAConfig — full externalization
│   ├── kv_store.py         # GPU-resident KV storage (LRU + pinning)
│   ├── streaming.py        # StreamingSession — per-token diagnostics
│   ├── distributed.py      # TPManager — Tensor Parallelism
│   ├── gate/               # 3-stage entropy gating + decay
│   ├── operator/           # Bottleneck injection operator
│   ├── retriever/          # BaseRetriever protocol + built-in impls
│   ├── adapter/            # HuggingFace / vLLM adapters
│   └── instrumentation/    # EventBus, ForwardMetrics, AuditLog
│
├── aga_knowledge/          ← aga-knowledge (optional)
│   ├── portal/             # FastAPI REST API + asset serving
│   ├── persistence/        # Memory / SQLite / PostgreSQL / Redis
│   ├── encoder/            # Text→Vector (SentenceTransformer)
│   ├── retriever/          # HNSW + BM25 + RRF hybrid search
│   ├── chunker/            # Document → Knowledge fragments
│   ├── alignment.py        # AGACoreAlignment
│   ├── sync/               # Redis Pub/Sub synchronization
│   └── config_adapter/     # aga-core ↔ aga-knowledge config bridge
│
├── aga_observability/      ← aga-observability (optional)
│   ├── prometheus_exporter.py  # Prometheus metrics export
│   ├── grafana_dashboard.py    # Auto-generated Grafana dashboards
│   ├── alert_manager.py        # SLO/SLI alerting engine
│   ├── log_exporter.py         # Structured log export
│   ├── audit_storage.py        # Persistent audit trail
│   ├── health.py               # Health check HTTP endpoint
│   └── stack.py                # ObservabilityStack orchestrator
│
├── configs/                # Example configuration files
├── tests/                  # All unit tests
└── pyproject.toml          # Root package (aga-core)
```

---

## Configuration

Example configuration files are provided in the `configs/` directory:

| File                                                         | Purpose                                                | Used By                |
| ------------------------------------------------------------ | ------------------------------------------------------ | ---------------------- |
| [`configs/runtime_config.yaml`](configs/runtime_config.yaml) | AGA runtime: entropy gating, decay, device, retriever  | `aga-core` AGAPlugin   |
| [`configs/portal_config.yaml`](configs/portal_config.yaml)   | Knowledge Portal: persistence, messaging, governance   | `aga-knowledge` Portal |

```python
# aga-core: load from YAML
plugin = AGAPlugin.from_config("configs/runtime_config.yaml")

# aga-knowledge: load Portal config
from aga_knowledge.config import PortalConfig
config = PortalConfig.from_yaml("configs/portal_config.yaml")
```

---

## Documentation

### aga-core

| Document                                                      | Language |
| ------------------------------------------------------------- | -------- |
| [README (English)](aga/README_en.md)                          | English  |
| [README (中文)](aga/README_zh.md)                             | 中文     |
| [Product Documentation (English)](aga/docs/product_doc_en.md) | English  |
| [产品说明书 (中文)](aga/docs/product_doc_zh.md)               | 中文     |
| [User Manual (English)](aga/docs/user_manual_en.md)           | English  |
| [用户手册 (中文)](aga/docs/user_manual_zh.md)                 | 中文     |

### aga-knowledge

| Document                                                                | Language |
| ----------------------------------------------------------------------- | -------- |
| [README (English)](aga_knowledge/README_en.md)                          | English  |
| [README (中文)](aga_knowledge/README_zh.md)                             | 中文     |
| [Product Documentation (English)](aga_knowledge/docs/product_doc_en.md) | English  |
| [产品说明书 (中文)](aga_knowledge/docs/product_doc_zh.md)               | 中文     |
| [User Manual (English)](aga_knowledge/docs/user_manual_en.md)           | English  |
| [用户手册 (中文)](aga_knowledge/docs/user_manual_zh.md)                 | 中文     |

### aga-observability

| Document                                                          | Language |
| ----------------------------------------------------------------- | -------- |
| [README (English)](aga_observability/README_en.md)                | English  |
| [README (中文)](aga_observability/README_zh.md)                   | 中文     |
| [User Manual (English)](aga_observability/docs/user_manual_en.md) | English  |
| [用户手册 (中文)](aga_observability/docs/user_manual_zh.md)       | 中文     |

---

## Testing

```bash
# All tests
python -m pytest tests/ -v

# aga-core tests
python -m pytest tests/test_aga/ -v

# aga-knowledge tests
python -m pytest tests/test_knowledge/ -v

# aga-observability tests
python -m pytest tests/test_observability/ -v
```

---

## Roadmap

| Package               | Current                                                    | Next Milestone                                                    |
| --------------------- | ---------------------------------------------------------- | ----------------------------------------------------------------- |
| **aga-core**          | v4.4.0 — Retriever protocol, slot governance, streaming    | v5.0 — Per-layer knowledge, INT8 KVStore, adaptive bottleneck     |
| **aga-knowledge**     | v0.3.0 — HNSW+BM25+RRF, DocumentChunker, AGACoreAlignment | v0.4.x — Contrastive fine-tuning, distributed encoder, Prometheus |
| **aga-observability** | v1.0.0 — Prometheus, Grafana, alerting, audit, health      | v1.1.0 — OpenTelemetry traces, distributed aggregation            |

---

## License

MIT License

Copyright (c) 2024-2026 AGA Team

---

<p align="center">
  <strong>AGA — Empowering every inference with knowledge</strong>
</p>
