"""
旅游景点 RAG 对话程序
=====================
基于 LangChain 1.3+ create_agent + 已完成全栈模块的交互式旅游问答系统。

特性：
- 加载 .env 配置 LLM（兼容 DeepSeek / OpenAI / 任何 OpenAI-API 格式服务）
- 自动初始化向量库、检索器、ScenicSpotRetrieveTool
- Agent 工具调用：景点查询自动检索知识库，无关问题直接模型回答
- LangGraph MemorySaver 对话记忆，支持多轮连续咨询
- 分层日志：工具检索内容 + 模型最终回答
- 全局异常兜底：API 密钥缺失、网络超时、检索异常、空知识库

启动方式：
    python main_rag_chat.py

前置条件：
    1. 向量库已有景点数据（运行 data_processor 导入）
    2. .env 中配置 LLM_API_KEY、LLM_API_BASE、LLM_MODEL_NAME
"""

import sys
import os
import logging
from typing import Optional, List, Dict, Any

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent          # LangChain 1.3+ 新 API
from langgraph.checkpoint.memory import MemorySaver  # 对话记忆
from langchain_core.messages import HumanMessage, AIMessage

from config.settings import settings
from modules.vector_store import ScenicVectorStore
from modules.retriever import ScenicRetriever
from modules.data_processor import ScenicDataProcessor
from modules.scenic_tool import ScenicSpotRetrieveTool


# ============================================================
# 日志配置
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rag_chat")


# ============================================================
# 旅游问答系统提示词
# ============================================================
TRAVEL_SYSTEM_PROMPT = """\
你是一个专业的旅游景点咨询助手，名叫"旅途小智"。你的知识来源于一个旅游景点知识库。

## 核心规则
1. 你只能基于 scenic_spot_search 工具检索到的景点信息回答用户问题。
2. 如果检索结果为空或不相关，请如实告知用户"抱歉，知识库中暂无相关信息"，绝不能编造景点信息。
3. 回答时请分点整理：景点名称、所在城市、票价、景区等级、特色标签、简介要点。
4. 如果用户询问多个景点或做对比，逐一列出，方便用户比较选择。
5. 对于拍照建议、穿搭推荐、交通路线、美食推荐等不在知识库范围内的问题，
   可以用你自己的常识简短回答，但需说明"此建议来自通用常识，非景点知识库信息"。

## 回答格式
- 先概述检索结果，再分点详述
- 票价分"免费"和具体金额
- 特色标签用 #古迹 #自然风光 #免费 等展示
- 语言通俗易懂，适合游客阅读
- 必要时给出游玩建议（如"建议游玩3-4小时""建议春秋季节前往"等通用建议）

## 禁止行为
- 禁止编造景点名称、地址、票价、等级等具体信息
- 禁止声称信息来自知识库而实际未检索到
- 禁止对知识库范围外的问题强制调用工具
"""


# ============================================================
# 初始化组件
# ============================================================
def init_llm() -> ChatOpenAI:
    """
    初始化 LLM 客户端（兼容 DeepSeek / OpenAI）。

    从 settings 读取 LLM_API_KEY, LLM_API_BASE, LLM_MODEL_NAME。
    若 API_KEY 缺失则抛出明确错误。

    返回:
        ChatOpenAI 实例
    """
    api_key: Optional[str] = settings.LLM_API_KEY
    api_base: Optional[str] = settings.LLM_API_BASE
    model_name: str = settings.LLM_MODEL_NAME

    if not api_key:
        raise RuntimeError(
            "LLM_API_KEY 未配置。请在 .env 文件中设置:\n"
            "  LLM_API_KEY=your-api-key-here\n"
            "  LLM_API_BASE=https://api.deepseek.com/v1  (DeepSeek必须带/v1后缀)\n"
            "  LLM_MODEL_NAME=deepseek-chat              (模型名称)"
        )

    kwargs: Dict[str, Any] = {
        "model": model_name,
        "api_key": api_key,
        "temperature": settings.LLM_TEMPERATURE,
        "max_tokens": settings.LLM_MAX_TOKENS,
        "streaming": False,
    }
    if api_base:
        kwargs["base_url"] = api_base

    print(f"[LLM] 模型: {model_name}, base_url: {api_base or '(OpenAI默认)'}")

    # 快速验证连接（发送一条极短消息，提前暴露错误）
    try:
        llm = ChatOpenAI(**kwargs)
        test_resp = llm.invoke("Hi")
        print(f"[LLM] 连接验证成功: {test_resp.content[:50]}...")
    except Exception as e:
        raise RuntimeError(
            f"LLM API 连接失败。请检查 .env 中的 LLM_API_KEY 和 LLM_API_BASE 配置。\n"
            f"原始错误: {type(e).__name__}: {e}"
        ) from e

    return llm


def init_agent(llm: ChatOpenAI) -> Any:
    """
    创建带记忆的 LangChain 1.3+ Agent。

    使用 langchain.agents.create_agent + LangGraph MemorySaver
    实现工具调用 + 多轮对话记忆。

    参数:
        llm: 已初始化的 ChatOpenAI 实例

    返回:
        Agent 实例，调用 agent.invoke({"messages": [...]})
    """
    # ---- 检索工具 ----
    retriever = ScenicRetriever()
    tool = ScenicSpotRetrieveTool(retriever=retriever)
    print(f"[Agent] 已加载检索工具: {tool.name}")

    # ---- 对话记忆（LangGraph MemorySaver） ----
    checkpointer = MemorySaver()
    print("[Agent] 对话记忆: LangGraph MemorySaver")

    # ---- 创建 Agent (LangChain 1.3+ API) ----
    agent = create_agent(
        model=llm,
        tools=[tool],
        system_prompt=TRAVEL_SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )

    print(f"[Agent] Agent 创建完成: tool={tool.name}")
    return agent


def check_vector_store() -> int:
    """
    检查向量库是否有数据。

    返回:
        向量库文档总数
    """
    store = ScenicVectorStore()
    count = store.count()
    if count == 0:
        print("[WARN] 向量库为空！请先导入景点数据。")
        print("  方式1: python -c \"from modules.data_processor import ScenicDataProcessor; "
              "ScenicDataProcessor().process_and_import()\"")
        print("  方式2: 将景点 JSON 文件放入 data/raw/ 目录后运行上述命令")
    else:
        print(f"[OK] 向量库: {count} 条景点数据")
    return count


# ============================================================
# 交互式对话循环
# ============================================================
WELCOME_BANNER = """
+----------------------------------------------------------------------+
|  旅途小智 - 旅游景点 RAG 智能问答系统                                  |
|  DeepSeek/OpenAI LLM + LangChain Agent + Chroma 向量知识库            |
|                                                                      |
|  输入问题开始对话，例如:                                               |
|    "北京有哪些5A级景点？"                                             |
|    "杭州免费景点推荐"                                                 |
|    "故宫和长城哪个更值得去？"                                          |
|    "成都看熊猫的地方"                                                 |
|                                                                      |
|  输入 exit / quit 退出   输入 clear 清除对话记忆                       |
+----------------------------------------------------------------------+
"""


def is_exit_command(text: str) -> bool:
    """判断是否为退出命令。"""
    return text.lower() in ("exit", "quit", "q", "退出")


def is_clear_command(text: str) -> bool:
    """判断是否为清空记忆命令。"""
    return text.lower() in ("clear", "cls", "清空", "重置")


def extract_tool_calls(messages: list) -> list:
    """
    从 LangChain 1.3 Agent 响应消息中提取工具调用详情。

    参数:
        messages: Agent 返回的消息列表

    返回:
        [{"name": str, "input": dict, "observation": str}, ...]
    """
    calls = []
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                name = tc.get("name", "unknown")
                args = tc.get("args", {})
                calls.append({"name": name, "input": args, "observation": ""})
        if hasattr(msg, "content") and hasattr(msg, "name") and msg.name:
            for call in calls:
                if call["name"] == msg.name and not call["observation"]:
                    call["observation"] = str(msg.content)[:200].replace("\n", " ")
                    break
    return calls


def chat_loop(agent: Any) -> None:
    """
    交互式命令行对话循环。

    参数:
        agent: create_agent 返回的 Agent 实例
    """
    print(WELCOME_BANNER)

    config = {"configurable": {"thread_id": "travel_chat_001"}}

    while True:
        try:
            user_input = input("\n[你] >>> ").strip()

            if not user_input:
                continue

            if is_exit_command(user_input):
                print("\n[旅途小智] 感谢使用，祝旅途愉快！再见~")
                break

            if is_clear_command(user_input):
                import uuid
                config["configurable"]["thread_id"] = (
                    f"travel_chat_{uuid.uuid4().hex[:8]}"
                )
                print("[旅途小智] 对话记忆已清除，我们可以重新开始~")
                continue

            # ---- 调用 Agent ----
            print()
            result = agent.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
            )

            # ---- 日志：工具调用 ----
            all_messages = result.get("messages", [])
            tool_calls = extract_tool_calls(all_messages)
            if tool_calls:
                logger.info(f"[Agent] 工具调用 {len(tool_calls)} 次:")
                for i, call in enumerate(tool_calls, 1):
                    logger.info(f"  [{i}] {call['name']}: {call['input']}")
                    if call.get("observation"):
                        logger.info(f"      结果预览: {call['observation']}...")
            else:
                logger.info("[Agent] 未调用检索工具，直接模型回答")

            # ---- 提取最终回答 ----
            final_answer = ""
            for msg in reversed(all_messages):
                if isinstance(msg, AIMessage) and msg.content:
                    final_answer = msg.content
                    break

            if final_answer:
                print(f"[旅途小智]\n{final_answer}")
            else:
                print("[旅途小智] 抱歉，我暂时无法回答这个问题，请稍后重试。")

        except KeyboardInterrupt:
            print("\n\n[旅途小智] 检测到中断，再见~")
            break

        except Exception as e:
            error_msg = str(e)
            logger.error(f"对话异常: {type(e).__name__}: {error_msg}")

            if "api_key" in error_msg.lower() or "auth" in error_msg.lower():
                print("[旅途小智] LLM API 密钥无效。请检查 .env 中的 LLM_API_KEY。")
            elif "timeout" in error_msg.lower() or "connect" in error_msg.lower():
                print("[旅途小智] 连接 LLM 服务超时，请检查网络或稍后重试。")
            elif "rate" in error_msg.lower() or "quota" in error_msg.lower():
                print("[旅途小智] API 调用频率过高或配额不足，请稍后重试。")
            else:
                print("[旅途小智] 抱歉，处理您的问题时出现了一个错误，请稍后重试。")
                logger.error(f"详细错误: {error_msg[:300]}")


# ============================================================
# 主入口
# ============================================================
def main() -> None:
    """主函数：初始化全部组件并启动对话循环。"""
    print("=" * 60)
    print("  旅途小智 - 旅游景点 RAG 对话系统 启动中...")
    print("=" * 60)

    env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(env_file):
        print("[WARN] 未找到 .env 文件。建议: cp .env.example .env 并填入 LLM_API_KEY")
    else:
        print(f"[OK] .env 已加载: {env_file}")

    check_vector_store()

    try:
        llm = init_llm()
    except RuntimeError as e:
        print(f"\n[ERROR] {e}")
        print("\n配置示例 (.env):")
        print("  LLM_API_KEY=sk-your-key-here")
        print("  LLM_API_BASE=https://api.deepseek.com/v1")
        print("  LLM_MODEL_NAME=deepseek-chat")
        sys.exit(1)

    try:
        agent = init_agent(llm)
    except Exception as e:
        print(f"\n[ERROR] Agent 初始化失败: {e}")
        sys.exit(1)

    try:
        chat_loop(agent)
    except Exception as e:
        print(f"\n[ERROR] 对话循环异常退出: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
