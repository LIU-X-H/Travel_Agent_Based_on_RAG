"""
对话持久化存储
==============
用 SQLite 持久化 LangGraph 对话检查点（异步兼容），支持：
- 服务重启后对话记忆不丢失
- 列出所有对话（thread_id + 标题 + 时间）
- 删除指定对话
"""

import os
import sqlite3
import threading
import time
from typing import Optional, List, Dict

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
import aiosqlite

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "data", "conversations.db")


class ConversationStore:
    """
    对话存储单例。
    - 底层: LangGraph AsyncSqliteSaver 存 Agent 检查点（兼容 astream_events）
    - 辅助: SQLite 表存对话元数据 (thread_id, title, updated_at)
    """

    _instance: Optional["ConversationStore"] = None
    _lock = threading.Lock()
    _async_lock = None  # asyncio.Lock, 在事件循环中初始化
    _checkpointer: Optional[AsyncSqliteSaver] = None
    _conn_path: str = DB_PATH

    def __init__(self):
        raise RuntimeError("请使用 ConversationStore.get_instance()")

    @classmethod
    def get_checkpointer(cls) -> AsyncSqliteSaver:
        inst = cls._get_instance()
        return inst._checkpointer

    # ---- 核心 ----
    @classmethod
    def _get_instance(cls) -> "ConversationStore":
        if cls._instance is not None:
            return cls._instance
        with cls._lock:
            if cls._instance is not None:
                return cls._instance
            inst = cls._instance = super().__new__(cls)
            os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
            # 创建元数据表（同步 SQLite）
            conn = sqlite3.connect(DB_PATH, check_same_thread=False)
            conn.execute(
                "CREATE TABLE IF NOT EXISTS threads "
                "(thread_id TEXT PRIMARY KEY, title TEXT, updated_at REAL)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS messages "
                "(id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "thread_id TEXT NOT NULL, role TEXT NOT NULL, "
                "content TEXT NOT NULL, created_at REAL NOT NULL)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_msg_thread ON messages(thread_id)"
            )
            conn.commit()
            conn.close()
            print(f"[ConversationStore] SQLite 对话存储已就绪: {DB_PATH}")
            return inst

    @classmethod
    async def init_checkpointer(cls) -> AsyncSqliteSaver:
        """异步初始化 AsyncSqliteSaver（必须在事件循环中调用）。"""
        import asyncio as _asyncio
        inst = cls._get_instance()
        if inst._checkpointer is not None:
            return inst._checkpointer
        if cls._async_lock is None:
            cls._async_lock = _asyncio.Lock()
        async with cls._async_lock:
            if inst._checkpointer is not None:
                return inst._checkpointer
            conn = await aiosqlite.connect(DB_PATH)
            inst._checkpointer = AsyncSqliteSaver(conn)
            await inst._checkpointer.setup()
            print(f"[ConversationStore] AsyncSqliteSaver 已初始化")
            return inst._checkpointer

    # ---- 对话元数据 ----
    @classmethod
    def _get_conn(cls) -> sqlite3.Connection:
        return sqlite3.connect(DB_PATH, check_same_thread=False)

    @classmethod
    def upsert_thread(cls, thread_id: str, title: str = "") -> None:
        conn = cls._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO threads(thread_id, title, updated_at) VALUES(?,?,?)",
            (thread_id, title, time.time()),
        )
        conn.commit(); conn.close()

    @classmethod
    def delete_thread(cls, thread_id: str) -> None:
        conn = cls._get_conn()
        conn.execute("DELETE FROM threads WHERE thread_id=?", (thread_id,))
        conn.execute("DELETE FROM messages WHERE thread_id=?", (thread_id,))
        conn.commit()
        try:
            conn.execute("DELETE FROM checkpoints WHERE thread_id=?", (thread_id,))
            conn.execute("DELETE FROM checkpoint_writes WHERE thread_id=?", (thread_id,))
            conn.commit()
        except Exception:
            pass
        conn.close()

    # ---- 消息 CRUD ----
    @classmethod
    def save_message(cls, thread_id: str, role: str, content: str) -> None:
        conn = cls._get_conn()
        conn.execute(
            "INSERT INTO messages(thread_id, role, content, created_at) VALUES(?,?,?,?)",
            (thread_id, role, content, time.time()),
        )
        conn.commit(); conn.close()

    @classmethod
    def get_messages(cls, thread_id: str) -> List[Dict]:
        conn = cls._get_conn()
        try:
            cur = conn.execute(
                "SELECT role, content, created_at FROM messages "
                "WHERE thread_id=? ORDER BY id ASC",
                (thread_id,),
            )
            return [{"role": r[0], "content": r[1], "created_at": r[2]}
                    for r in cur.fetchall()]
        finally:
            conn.close()

    @classmethod
    def delete_messages(cls, thread_id: str) -> None:
        conn = cls._get_conn()
        conn.execute("DELETE FROM messages WHERE thread_id=?", (thread_id,))
        conn.commit(); conn.close()

    @classmethod
    def list_threads(cls) -> List[Dict]:
        conn = cls._get_conn()
        cur = conn.execute(
            "SELECT thread_id, title, updated_at FROM threads ORDER BY updated_at DESC"
        )
        result = [{"thread_id": r[0], "title": r[1] or "(空对话)", "updated_at": r[2]}
                  for r in cur.fetchall()]
        conn.close()
        return result
