# 旅途小智 - 旅游 AI Agent

> 基于 LangGraph + LangChain + Chroma + BGE 的旅游智能体，支持多工具调用、持久记忆、流式对话

## 特性

- **6 个智能工具**：景点检索 / 天气查询 / 美食推荐 / 行程规划 / 汇率换算 / 酒店查询
- **混合检索**：BM25 关键词 + BGE 语义向量加权融合
- **Agent 自主决策**：LLM 判断是否调工具、调哪个、传什么参数，零硬编码路由
- **流式对话**：SSE 逐字输出 + 工具调用可视化
- **持久记忆**：SQLite 对话存储，重启不丢
- **多对话管理**：Web UI 支持新建/切换/删除对话
- **双入口**：Web 聊天页面 + 命令行终端
- **51 条景点 + 30 道美食**：内置真实数据，开箱即用

## 快速启动

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 API Key
cp .env.example .env
# 编辑 .env，填入 LLM_API_KEY

# 3. 导入数据（景点 + 美食）
python -c "from modules.data_processor import ScenicDataProcessor; ScenicDataProcessor().process_and_import('scenic_city.json')"
python -c "from modules.food_tool import FoodTool; FoodTool()"

# 4. 启动 Web 服务
python main_api.py
# 访问 http://localhost:8000/chat
```

## 项目结构

```
travel_scenic_rag/
├── config/                        # 全局配置
│   └── settings.py                #   所有参数集中管理，.env 覆盖
│
├── modules/                       # 功能模块
│   ├── embedding.py               #   BGE 向量化（单例）
│   ├── vector_store.py            #   Chroma 向量库封装
│   ├── retriever.py               #   多策略检索（filter/hybrid/rerank）
│   ├── data_processor.py          #   数据加载、清洗、入库管线
│   ├── scenic_tool.py             #   景点检索工具（LangChain Tool）
│   ├── weather_tool.py            #   天气查询工具（Open-Meteo API）
│   ├── food_tool.py               #   美食推荐工具
│   ├── hotel_tool.py              #   酒店查询工具（参考价 + 预订链接）
│   ├── itinerary_tool.py          #   行程规划工具
│   └── exchange_tool.py           #   汇率换算工具
│
├── core/                          # Agent 核心
│   ├── agent_builder.py           #   Agent 构建（LLM + 工具 + 提示词）
│   └── conversation_store.py      #   SQLite 对话持久化
│
├── api/                           # API 服务
│   └── routes/agent.py            #   SSE 流式 + 对话管理接口
│
├── static/
│   └── index.html                 #   Web 聊天 UI
│
├── data/                          # 数据
│   ├── raw/                       #   原始 JSON（景点/美食）
│   ├── vector_db/                 #   Chroma 向量库
│   └── conversations.db           #   对话历史 SQLite
│
├── tests/                         # 单元测试（62+ 条断言）
├── main_api.py                    # Web 服务入口
├── main_rag_chat.py               # CLI 对话入口
└── demo_tool.py                   # 工具演示脚本
```

## API 接口

| 方法       | 路径                                 | 说明         |
| ---------- | ------------------------------------ | ------------ |
| `POST`   | `/api/agent/chat`                  | SSE 流式对话 |
| `GET`    | `/api/agent/threads`               | 对话列表     |
| `GET`    | `/api/agent/threads/{id}/messages` | 历史消息     |
| `DELETE` | `/api/agent/threads/{id}`          | 删除对话     |
| `POST`   | `/api/scenic/search`               | 景点检索     |
| `GET`    | `/api/health`                      | 健康检查     |
| `GET`    | `/docs`                            | Swagger 文档 |

## Agent 工具箱

| 工具                   | 数据源                     | 能力                 |
| ---------------------- | -------------------------- | -------------------- |
| `scenic_spot_search` | Chroma 向量库（51 条景点） | 按城市/票价/标签过滤 |
| `get_weather`        | Open-Meteo（免费，全球）   | 7 天预报             |
| `search_food`        | Chroma 向量库（30 道美食） | 按城市/菜系过滤      |
| `search_hotels`    | 内置城市价格区间表        | 参考价 + Trip.com/携程直达链接 |
| `plan_itinerary`     | Agent 编排                 | 按城市分组、分配天数 |
| `exchange_rate`      | 内置汇率表                 | 20+ 货币换算         |

## 技术栈

| 层级       | 技术                                |
| ---------- | ----------------------------------- |
| Agent 框架 | LangGraph + LangChain 1.3+          |
| LLM        | DeepSeek / OpenAI（ChatOpenAI）     |
| 向量库     | Chroma（HNSW 索引 + SQLite 持久化） |
| Embedding  | BAAI/bge-small-zh-v1.5（512 维）    |
| 混合检索   | BGE 语义 + BM25（jieba 分词）       |
| 对话记忆   | AsyncSqliteSaver                    |
| API 服务   | FastAPI + SSE 流式                  |
| 前端       | 原生 HTML/CSS/JS（零框架依赖）      |

## 配置参数

| 参数                     | 默认值                     | 说明                        |
| ------------------------ | -------------------------- | --------------------------- |
| `LLM_API_KEY`          | —                         | **必填**，LLM API Key |
| `LLM_API_BASE`         | —                         | DeepSeek/OpenAI 地址        |
| `EMBEDDING_MODEL_NAME` | `BAAI/bge-small-zh-v1.5` | Embedding 模型              |
| `HYBRID_ALPHA`         | `0.7`                    | 语义检索权重                |
| `SIMILARITY_THRESHOLD` | `0.0`                    | 相似度过滤阈值              |
| `RERANK_ENABLED`       | `false`                  | ReRank 精排开关             |

> 完整配置见 [.env.example](.env.example)


