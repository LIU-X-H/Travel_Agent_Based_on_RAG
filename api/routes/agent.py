"""Agent 聊天 API 路由 - SSE 流式响应。"""
import json
import uuid
import asyncio
import logging
from typing import Optional, AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage

from core.agent_builder import init_agent, init_llm, check_vector_store
from core.conversation_store import ConversationStore

logger = logging.getLogger("agent_router")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    thread_id: Optional[str] = Field(default=None)


class ChatError(BaseModel):
    error: str = Field(...)
    message: str = Field(...)


_agent = None
_agent_lock = asyncio.Lock()


async def get_agent():
    global _agent
    if _agent is not None:
        return _agent
    async with _agent_lock:
        if _agent is not None:
            return _agent
        check_vector_store()
        loop = asyncio.get_event_loop()
        llm = await loop.run_in_executor(None, init_llm)
        checkpointer = await ConversationStore.init_checkpointer()
        _agent = await loop.run_in_executor(None, init_agent, llm, checkpointer)
        return _agent


async def event_stream(agent, user_message: str, thread_id: str) -> AsyncGenerator[str, None]:
    config = {"configurable": {"thread_id": thread_id}}
    full_answer: list[str] = []
    try:
        async for ev in agent.astream_events(
            {"messages": [HumanMessage(content=user_message)]},
            config=config, version="v2",
        ):
            k = ev["event"]
            if k == "on_tool_start":
                d = {"name": ev.get("name",""), "input": ev.get("data",{}).get("input",{})}
                yield "event: tool_start\ndata: " + json.dumps(d, ensure_ascii=False) + "\n\n"
            elif k == "on_tool_end":
                out = str(ev.get("data",{}).get("output",""))[:200]
                yield "event: tool_end\ndata: " + json.dumps({"observation": out}, ensure_ascii=False) + "\n\n"
            elif k == "on_chat_model_stream":
                chunk = ev.get("data",{}).get("chunk")
                if not chunk:
                    continue
                c = chunk.content if hasattr(chunk, "content") else ""
                if isinstance(c, str) and c.strip():
                    full_answer.append(c)
                    yield "event: token\ndata: " + json.dumps({"text": c}, ensure_ascii=False) + "\n\n"
        # 保存 AI 回答到持久化存储
        ai_text = "".join(full_answer)
        if ai_text.strip():
            ConversationStore.save_message(thread_id, "ai", ai_text)
        yield "event: done\ndata: " + json.dumps({"text": "", "thread_id": thread_id}) + "\n\n"
    except Exception as e:
        logger.error(f"event_stream error: {e}")
        yield "event: error\ndata: " + json.dumps({"error": str(e)}, ensure_ascii=False) + "\n\n"
        yield "event: done\ndata: " + json.dumps({"text": "", "thread_id": thread_id}) + "\n\n"


router = APIRouter(prefix="/api/agent", tags=["agent"])


@router.post("/chat", response_class=StreamingResponse)
async def chat(request: ChatRequest):
    try:
        agent = await get_agent()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    tid = request.thread_id or f"travel_chat_{uuid.uuid4().hex[:8]}"
    # 保存对话标题 + 用户消息
    title = request.message[:30].replace("\n", " ")
    if len(request.message) > 30:
        title += "…"
    ConversationStore.upsert_thread(tid, title)
    ConversationStore.save_message(tid, "user", request.message)
    return StreamingResponse(
        event_stream(agent, request.message, tid),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache", "Connection": "keep-alive",
            "X-Thread-Id": tid, "Access-Control-Allow-Origin": "*",
        },
    )


@router.get("/threads")
async def list_threads():
    """获取所有对话列表。"""
    return {"threads": ConversationStore.list_threads()}


@router.delete("/threads/{thread_id}")
async def delete_thread(thread_id: str):
    """删除指定对话及其检查点。"""
    ConversationStore.delete_thread(thread_id)
    return {"success": True, "thread_id": thread_id}


@router.get("/threads/{thread_id}/messages")
async def get_thread_messages(thread_id: str):
    """获取指定对话的历史消息。"""
    return {"thread_id": thread_id, "messages": ConversationStore.get_messages(thread_id)}
