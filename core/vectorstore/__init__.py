"""
向量库模块 (VectorStore)
------------------------
负责 Chroma 向量数据库的连接、集合管理与持久化。
"""

from .chroma_store import ChromaStore

__all__ = ["ChromaStore"]
