# 大模型工具调用链路详解

本文档完整阐述「旅途小智」旅游 RAG 系统中，**大模型如何感知工具 → 决策调用 → 执行工具 → 整合响应的全过程**。

---

## 一、整体架构概览

```
用户输入 ──→ ① LangGraph Agent (LLM + 工具列表)
                  │
                  ├←─ ② LLM 推理：决定调用哪个工具 + 填充参数
                  │
                  ├──→ ③ 工具执行（检索/API/计算）
                  │        │
                  │        ├─ scenic_spot_search  → Chroma 向量库 + BM25 + Reranker
                  │        ├─ get_weather          → Open-Meteo API
                  │        ├─ search_food           → Chroma 美食集合
                  │        ├─ plan_itinerary        → 纯本地逻辑规划
                  │        ├─ search_hotels         → 硬编码参考价 + OTA 链接
                  │        └─ exchange_rate         → 硬编码汇率计算
                  │
                  ├──→ ④ 工具结果返回给 LLM
                  │
                  └──→ ⑤ LLM 整合结果 → 自然语言回答
                           │
                       用户看到最终回答
```

---

## 二、工具的定义方式

每个工具都继承自 `langchain_core.tools.BaseTool`，需要声明 4 个要素：

| 要素 | 说明 | 示例（scenic_tool.py） |
|------|------|----------------------|
| `name` | 工具名称，LLM 通过这个名字引用工具 | `"scenic_spot_search"` |
| `description` | **自然语言描述**，LLM 据此判断何时调用 | `"从旅游景点知识库中检索匹配的景点信息..."` |
| `args_schema` | Pydantic 模型，声明每个参数名、类型、描述 | `ScenicSpotSearchInput` |
| `_run()` | 工具的实际执行逻辑 | 调用 `retriever.hybrid_search()` |

关键点：**LLM 不是通过代码逻辑感知工具的，而是通过 `description` 字段的语义理解**。工具的描述写得越清晰，LLM 决策越准确。

### 工具清单

| 工具名称 | 触发场景 | 数据来源 |
|----------|----------|----------|
| `scenic_spot_search` | 用户问景点推荐、门票、等级 | Chroma 向量知识库 |
| `get_weather` | 用户问天气 | Open-Meteo 免费 API |
| `search_food` | 用户问美食、餐厅 | Chroma 美食集合 |
| `plan_itinerary` | 用户要求规划行程 | 纯本地逻辑（不自感知调用 LLM） |
| `search_hotels` | 用户问酒店、住宿 | 硬编码参考价 + 携程链接 |
| `exchange_rate` | 用户问汇率换算 | 硬编码汇率表 |

---

## 三、LLM 决策工具调用的完整流程

### 第 1 步：Agent 初始化（main_rag_chat.py / core/agent_builder.py）

```python
# core/agent_builder.py:57-66
tools = [
    ScenicSpotRetrieveTool(retriever=ScenicRetriever()),
    WeatherTool(), FoodTool(), ItineraryTool(), ExchangeTool(), HotelTool(),
]
agent = create_agent(
    model=llm,           # ChatOpenAI（兼容 DeepSeek / OpenAI）
    tools=tools,         # 工具列表 → 自动转为 LLM 的 tool 定义
    system_prompt=TRAVEL_SYSTEM_PROMPT,  # 系统指令
    checkpointer=checkpointer,           # MemorySaver 对话记忆
)
```

`create_agent` 是 LangChain 1.3+ 的高阶 API（底层是 LangGraph `create_react_agent`），它做了 3 件事：

1. 将每个 `BaseTool` 转为 LLM 的 **tool schema**（OpenAI 工具调用格式）
2. 将 `system_prompt` 设为 Agent 的系统消息
3. 用 `MemorySaver` 包裹，实现多轮对话记忆

### 第 2 步：用户输入触发推理（chat_loop → astream_events）

```python
# main_rag_chat.py:302-304
async for event in agent.astream_events(
    {"messages": [HumanMessage(content=user_input)]},
    config=config,
    version="v2",
):
```

当用户输入 `"北京有哪些免费5A景点？"` 时：

1. **Agent 打包上下文**：系统提示词 + 历史消息 + 当前用户消息 → 发送给 LLM
2. **LLM 推理**：LLM 看到工具列表的 `description` + `args_schema`，判断应该调用 `scenic_spot_search`
3. **LLM 输出 tool_call**：LLM 返回的不是自然语言，而是结构化的 **tool_call 指令**：

```json
{
  "tool_calls": [{
    "name": "scenic_spot_search",
    "args": {
      "query": "北京免费5A景点",
      "city": "北京",
      "tags": ["免费", "5A"],
      "top_k": 5
    }
  }]
}
```

这个过程不涉及任何代码分支判断——**完全由 LLM 根据描述语义自主决策**。

### 第 3 步：工具执行

LangGraph 运行时检测到 LLM 输出的 `tool_calls`，自动路由到对应的工具执行：

```
on_tool_start 事件触发
    ↓
调用 ScenicSpotRetrieveTool._run(query="北京免费5A景点", city="北京", tags=["免费", "5A"], top_k=5)
    ↓
ScenicSpotRetrieveTool._run() 内部：
    ├── 输入校验
    ├── 构建过滤条件（_build_filter_clause）
    ├── 调用 retriever.hybrid_search()  ← 混合检索核心
    │       ├── 语义检索（BGE Embedding → Chroma 相似度搜索）
    │       ├── BM25 关键词检索（jieba 分词 + rank_bm25）
    │       └── 加权融合（alpha=0.7 语义 + 0.3 BM25）
    ├── Python 层后置过滤（票价、标签兜底）
    └── 格式化为自然语言文本（_format_results）
    ↓
on_tool_end 事件触发 → 工具结果返回给 LLM
```

**混合检索细节**（ScenicRetriever.hybrid_search）：

```
用户 query: "北京免费5A景点"
            │
            ├──→ BGE Embedding (bge-small-zh-v1.5) → 512维向量
            │       └──→ Chroma cosine similarity search → top 15 候选
            │
            ├──→ jieba 分词 → ["北京", "免费", "5A", "景点"]
            │       └──→ BM25Okapi 关键词匹配 → 分数归一化
            │
            └──→ 融合：final_score = 0.7 × 语义分 + 0.3 × BM25分
                    └──→ 排序 → top 5 → 返回
```

### 第 4 步：LLM 接收工具结果并生成回答

工具执行完毕后，工具的输出（格式化文本）作为 **ToolMessage** 返回给 LLM。LLM 现在手里有：

```
系统消息: "你是一个专业的旅游景点咨询助手..."
用户消息: "北京有哪些免费5A景点？"
工具结果: "[搜索] 为您找到 3 个匹配的旅游景点：
           1. 【故宫博物院】🏛 古迹 · 博物馆
              城市：北京 | 票价：免费 | 等级：5A
              ...
           2. 【天坛公园】..."
```

LLM 将此工具结果整合为自然语言回答，流式输出给用户。

---

## 四、系统提示词的作用

系统提示词（[agent_builder.py:22-30](core/agent_builder.py#L22-L30)）直接约束 LLM 的工具调用决策：

```
## 核心规则
1. 只能基于 scenic_spot_search 工具检索到的景点信息回答用户问题。
2. 如果检索结果为空，如实告知用户。
3. 回答时整理：景点名称、所在城市、票价、景区等级、特色标签、简介。
4. 查询天气时，根据用户需求选择天数：问"周末"传 days=7，问"明天"传 days=2，
   问"今天"传 days=1，一次调够，不要重复调用。
5. 对于不在知识库范围内的问题，可以用自己的常识简短回答。
```

第 4 条规则说明了一个重要模式：**系统提示词可以指导 LLM 如何填充工具参数**，弥补纯 Pydantic schema 描述无法表达的"业务规则"。

---

## 五、多工具串联场景

用户提问："杭州这周天气怎么样，有什么好玩的景点？"

LLM 的推理链路：

```
第 1 轮推理:
  LLM 输出: tool_call → get_weather(city="杭州", days=7)
  工具执行 → 返回天气数据 → LLM 收到 ToolMessage

第 2 轮推理:
  LLM 输出: tool_call → scenic_spot_search(query="杭州热门景点", city="杭州", top_k=5)
  工具执行 → 返回景点列表 → LLM 收到 ToolMessage

第 3 轮推理:
  LLM 整合天气 + 景点结果 → 输出综合回答
```

这个过程不是提前编排的，而是 **LLM 看到工具结果后，自主决定是否还需要另一个工具**。

---

## 六、记忆机制：多轮对话如何工作

```python
# core/agent_builder.py:56
checkpointer = MemorySaver()
```

- `MemorySaver` 是 LangGraph 的检查点机制，保存每轮对话的完整状态
- 每轮对话消息自动累积：`[sys_msg, user_msg, tool_call, tool_result, ai_answer, user_msg2, ...]`
- `thread_id` 标识不同对话会话（`config = {"configurable": {"thread_id": "travel_chat_001"}}`）
- API 模式下使用 SQLite 持久化（[ConversationStore](core/conversation_store.py)）

这允许用户在后续轮次中追问 "第一个景点门票多少？" —— LLM 能从历史消息中提取上下文。

---

## 七、流式响应事件流

[main_rag_chat.py](main_rag_chat.py) 中通过 `astream_events` 监听 4 种事件：

| 事件 | 含义 | UI 表现 |
|------|------|---------|
| `on_chat_model_stream` | LLM 逐 token 生成 | 逐个字符打印回答 |
| `on_tool_start` | 工具开始执行 | 日志打印 `[Agent] 调用工具: scenic_spot_search({...})` |
| `on_tool_end` | 工具执行完毕 | 日志打印检索结果预览 |
| `on_chain_end` | 一轮完整推理结束 | 日志收尾 |

API 模式下（[agent.py](api/routes/agent.py#L48-L78)）通过 **SSE (Server-Sent Events)** 推送：

```
event: tool_start
data: {"name": "scenic_spot_search", "input": {"query": "北京免费5A景点", ...}}

event: tool_end
data: {"observation": "[搜索] 为您找到 3 个匹配的旅游景点：..."}

event: token
data: {"text": "为您推荐以下"}   ← 逐 token

event: token
data: {"text": "北京免费5A景点"}   ← 逐 token

event: done
data: {"text": "", "thread_id": "travel_chat_xxx"}
```

---

## 八、完整链路图（含数据流）

```
┌─────────────────────────────────────────────────────────────────┐
│ 用户: "北京有哪些免费5A景点？"                                    │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│ [LangGraph Agent]  create_react_agent                           │
│                                                                 │
│   系统提示词 (约束规则)                                           │
│   + 历史消息 (对话记忆)                                          │
│   + 用户消息                                                    │
│   + 工具描述 (6个工具的 name + description)                      │
│                       │                                         │
└───────────────────────┼─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│ [LLM 推理]  如 DeepSeek / OpenAI                                │
│                                                                 │
│   决策: 这个查询需要调用 scenic_spot_search                      │
│   填充参数: query="北京免费5A景点", city="北京", tags=[免费,5A]   │
│                                                                 │
│   输出: tool_call(name="scenic_spot_search", args={...})        │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│ [工具路由]  LangGraph 按 name 匹配并执行                         │
│                                                                 │
│   scenic_spot_search._run(query, city, tags, top_k=5)           │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│ [ScenicRetriever.hybrid_search]  混合检索                        │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐                           │
│  │ BGE Embedding │    │ BM25 关键词  │                          │
│  │ bge-small-zh  │    │ jieba +      │                          │
│  │ → 512维向量   │    │ BM25Okapi    │                          │
│  │ → Chroma 检索  │    │ → 分数归一化 │                          │
│  └──────┬───────┘    └──────┬───────┘                           │
│         │                    │                                   │
│         └────────┬──────────┘                                   │
│                  ▼                                               │
│         加权融合 (α=0.7)                                        │
│                  │                                               │
│                  ▼                                               │
│         排序 + 截断 → top 5                                     │
│                  │                                               │
│                  ▼                                               │
│   [格式化] 转为自然语言文本                                      │
│   "【故宫博物院】... 票价：免费 | 等级：5A..."                    │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│ [LLM 推理]  收到 ToolMessage                                    │
│                                                                 │
│   输入: 系统提示词 + 用户消息 + 工具结果                         │
│   输出: "为您推荐3个北京免费的5A景点:                            │
│           1. 故宫博物院——位于北京东城区，免费开放，5A级景区...   │
│           2. 天坛公园——..."                                     │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│ [流式输出]  astream_events → on_chat_model_stream               │
│                                                                 │
│   "为" → "您" → "推" → "荐" → ... → 完整回答打印                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 九、关键代码引用

| 功能 | 文件 | 行号 |
|------|------|------|
| 工具定义基类 | [modules/scenic_tool.py](modules/scenic_tool.py) | 83-370 |
| Agent 组装 | [core/agent_builder.py](core/agent_builder.py) | 52-68 |
| 系统提示词 | [core/agent_builder.py](core/agent_builder.py) | 22-30 |
| LLM 初始化 | [core/agent_builder.py](core/agent_builder.py) | 32-50 |
| 对话循环 + 流式 | [main_rag_chat.py](main_rag_chat.py) | 263-362 |
| 工具调用事件提取 | [main_rag_chat.py](main_rag_chat.py) | 238-260 |
| 混合检索核心 | [modules/retriever.py](modules/retriever.py) | 338-450 |
| Reranker 精排 | [modules/retriever.py](modules/retriever.py) | 455-608 |
| 向量库检索 | [modules/vector_store.py](modules/vector_store.py) | 393-509 |
| Embedding 模型 | [modules/embedding.py](modules/embedding.py) | 55-355 |
| API SSE 流式 | [api/routes/agent.py](api/routes/agent.py) | 48-78 |
| 对话持久化 | [core/conversation_store.py](core/conversation_store.py) | 23-159 |
| 全局配置 | [config/settings.py](config/settings.py) | 29-192 |
| 数据预处理管线 | [modules/data_processor.py](modules/data_processor.py) | 73-623 |

---

## 十、总结

| 阶段 | 谁在做 | 本质 |
|------|--------|------|
| **工具定义** | 开发者（`BaseTool` 子类） | 把功能包装成 LLM 可理解的 schema |
| **调用决策** | **LLM**（不是代码） | 根据 `description` 语义自主判断 |
| **参数填充** | **LLM**（不是代码） | 从用户问题中提取参数 |
| **工具执行** | Python 代码（`_run()`） | 检索 / API 调用 / 计算 |
| **结果整合** | **LLM** | 把结构化数据转为自然语言回答 |
| **多轮串联** | LangGraph MemorySaver | 自动累积消息历史 |

**核心设计思想**：LLM 充当"决策大脑"，工具是"手脚"——大脑决定用什么工具、怎么用，手脚执行具体操作，然后把结果汇报给大脑，大脑再组织语言回答用户。这与 Function Calling / Tool Using 的范式完全一致。