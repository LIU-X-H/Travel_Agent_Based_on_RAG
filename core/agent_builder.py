"""Agent 构建模块 - 供 CLI 和 API 共享使用。"""
import sys, os, logging
from typing import Optional, Any, Dict
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from config.settings import settings
from modules.vector_store import ScenicVectorStore
from modules.retriever import ScenicRetriever
from modules.scenic_tool import ScenicSpotRetrieveTool
from modules.weather_tool import WeatherTool
from modules.food_tool import FoodTool
from modules.itinerary_tool import ItineraryTool
from modules.exchange_tool import ExchangeTool
from core.conversation_store import ConversationStore

logger = logging.getLogger("agent_builder")

TRAVEL_SYSTEM_PROMPT = """\
你是一个专业的旅游景点咨询助手，名叫"旅途小智"。
## 核心规则
1. 只能基于 scenic_spot_search 工具检索到的景点信息回答用户问题。
2. 如果检索结果为空，如实告知用户。
3. 回答时整理：景点名称、所在城市、票价、景区等级、特色标签、简介。
4. 查询天气时，根据用户需求选择天数：问"周末"传 days=7，问"明天"传 days=2，问"今天"传 days=1，一次调够，不要重复调用。
5. 对于不在知识库范围内的问题，可以用自己的常识简短回答。
"""

def init_llm() -> ChatOpenAI:
    api_key = settings.LLM_API_KEY
    if not api_key:
        raise RuntimeError("LLM_API_KEY 未配置。请在 .env 文件中设置。")
    kwargs = {
        "model": settings.LLM_MODEL_NAME, "api_key": api_key,
        "temperature": settings.LLM_TEMPERATURE,
        "max_tokens": settings.LLM_MAX_TOKENS, "streaming": True,
    }
    if settings.LLM_API_BASE:
        kwargs["base_url"] = settings.LLM_API_BASE
    logger.info(f"LLM 模型: {settings.LLM_MODEL_NAME}")
    try:
        tk = dict(kwargs, streaming=False)
        r = ChatOpenAI(**tk).invoke("Hi")
        logger.info(f"LLM 连接验证成功: {r.content[:50]}")
    except Exception as e:
        raise RuntimeError(f"LLM API 连接失败: {e}") from e
    return ChatOpenAI(**kwargs)

def init_agent(llm: Optional[ChatOpenAI] = None, checkpointer=None) -> Any:
    if llm is None:
        llm = init_llm()
    if checkpointer is None:
        checkpointer = MemorySaver()
    tools = [
        ScenicSpotRetrieveTool(retriever=ScenicRetriever()),
        WeatherTool(), FoodTool(), ItineraryTool(), ExchangeTool(),
    ]
    logger.info(f"Agent 工具: {[t.name for t in tools]}")
    agent = create_agent(
        model=llm, tools=tools,
        system_prompt=TRAVEL_SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )
    logger.info("Agent 创建完成")
    return agent

def check_vector_store() -> int:
    c = ScenicVectorStore().count()
    if c == 0:
        print("[WARN] 向量库为空！请先导入景点数据。")
    else:
        print(f"[OK] 向量库 {c} 条景点数据")
    return c
