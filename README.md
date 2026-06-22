# 🏯 旅游景点 RAG 知识库项目

> 基于 LangChain + Chroma + BGE 的旅游景点智能检索问答系统
> 后续将接入 [LangGraph](https://github.com/langchain-ai/langgraph) 构建旅游智能体

## 📁 项目结构

```
travel_scenic_rag/
├── config/                      # 🔧 配置模块 — 全局参数统一管理
│   ├── __init__.py
│   └── settings.py              #   所有可调参数集中定义，支持 .env 覆盖
│
├── core/                        # 🧠 核心模块 — RAG 检索管线
│   ├── __init__.py
│   ├── embeddings/              #   向量化：BGE 模型加载与文本嵌入
│   │   ├── __init__.py
│   │   └── embedding_loader.py  #     SentenceTransformer 封装
│   ├── vectorstore/             #   向量库：Chroma 连接与集合管理
│   │   ├── __init__.py
│   │   └── chroma_store.py      #     Chroma PersistentClient 封装
│   ├── retrieval/               #   检索：混合检索、重排序、过滤
│   │   └── __init__.py          #     (预留) 语义/BM25/混合检索实现
│   └── llm/                     #   LLM：大模型调用与提示词模板
│       ├── __init__.py
│       └── llm_client.py        #     LangChain ChatOpenAI 封装
│
├── agents/                      # 🤖 智能体模块 — LangGraph 旅游智能体(预留)
│   └── __init__.py
│
├── api/                         # 🌐 API 模块 — FastAPI 对外服务
│   ├── __init__.py
│   └── routes/                  #   路由：按资源拆分接口
│       └── __init__.py
│
├── data/                        # 📦 数据目录
│   ├── raw/                     #   原始数据：抓取的景点文本/JSON
│   ├── processed/               #   处理后数据：清洗分块后的文档
│   └── vector_db/               #   向量库持久化：Chroma 本地存储
│
├── utils/                       # 🛠 工具模块 — 日志、文本处理等通用函数
│   ├── __init__.py
│   └── logger.py                #   统一日志配置
│
├── tests/                       # 🧪 测试模块 — 单元测试与集成测试
│   └── __init__.py
│
├── scripts/                     # 📜 脚本目录 — 数据导入、索引构建等
│
├── .env.example                 # 环境变量模板
├── requirements.txt             # 依赖清单
└── README.md                    # 项目说明
```

## 🚀 快速启动

### 1. 环境准备

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate   # Linux/Mac
# venv\Scripts\activate    # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 按需编辑 .env 文件，配置 API Key 等参数
```

### 3. 验证安装

```python
from config import Settings
from core.embeddings import EmbeddingLoader
from core.vectorstore import ChromaStore

# 测试配置加载
settings = Settings()
settings.ensure_directories()
print("✅ 配置加载成功")

# 测试 Embedding 加载
loader = EmbeddingLoader()
loader.load()
print("✅ Embedding 模型加载成功")

# 测试 Chroma 连接
store = ChromaStore()
store.connect()
print("✅ Chroma 连接成功")
```

## ⚙️ 核心配置参数

| 分类 | 参数 | 默认值 | 说明 |
|------|------|--------|------|
| Embedding | `EMBEDDING_MODEL_NAME` | `BAAI/bge-large-zh-v1.5` | BGE 中文向量模型 |
| 向量库 | `CHROMA_COLLECTION_NAME` | `travel_scenic_spots` | Chroma 集合名称 |
| 检索 | `SEMANTIC_TOP_K` | `10` | 语义检索候选数 |
| 检索 | `HYBRID_TOP_K` | `5` | 混合检索返回数 |
| 混合权重 | `HYBRID_ALPHA` | `0.7` | 语义检索权重 (BM25=1-α) |
| 重排序 | `RERANK_ENABLED` | `false` | 是否启用 ReRank |
| 分块 | `CHUNK_SIZE` | `512` | 文本分块大小 |

> 完整参数列表见 [config/settings.py](config/settings.py) 或 [.env.example](.env.example)

## 🧩 技术栈

- **LLM 框架**: LangChain
- **向量数据库**: Chroma
- **Embedding**: BGE (BAAI/bge-large-zh-v1.5)
- **关键词检索**: BM25 (rank_bm25)
- **API 服务**: FastAPI + Uvicorn
- **智能体**: LangGraph (后续接入)
- **数据处理**: Pandas + Pydantic

## 📝 开发计划

- [ ] 语义检索实现
- [ ] BM25 关键词检索实现
- [ ] 混合检索 + RRF 融合
- [ ] BGE ReRanker 重排序
- [ ] FastAPI 检索接口
- [ ] LangGraph 旅游智能体
- [ ] 景点数据导入脚本
- [ ] 单元测试用例
