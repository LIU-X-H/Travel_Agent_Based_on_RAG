"""
Chroma 向量库封装
-----------------
提供 Chroma 客户端的初始化、集合管理与基础 CRUD 操作。
所有连接参数统一从 config.settings 读取。
"""

import os
from typing import List, Optional, Dict, Any
from pathlib import Path

from config.settings import settings


class ChromaStore:
    """
    Chroma 向量数据库封装
    ---------------------
    管理向量库连接与集合，提供统一的 CRUD 接口。

    使用示例：
        store = ChromaStore()
        collection = store.get_collection()
    """

    def __init__(self) -> None:
        """初始化连接参数，不立即建立连接（延迟连接）。"""
        self._client = None
        self._collection = None
        self._persist_path: Path = settings.VECTOR_DB_PATH
        self._collection_name: str = settings.CHROMA_COLLECTION_NAME
        self._distance_metric: str = settings.CHROMA_DISTANCE_METRIC

    def connect(self) -> None:
        """
        建立 Chroma 客户端连接。
        若已连接则跳过。
        """
        if self._client is not None:
            return

        try:
            import chromadb

            # 确保持久化目录存在
            self._persist_path.mkdir(parents=True, exist_ok=True)

            self._client = chromadb.PersistentClient(
                path=str(self._persist_path),
            )
            print(
                f"[ChromaStore] 客户端已连接，持久化路径: {self._persist_path}"
            )

        except ImportError as e:
            raise ImportError(
                "请安装 chromadb: pip install chromadb"
            ) from e
        except Exception as e:
            raise RuntimeError(f"连接 Chroma 失败: {e}") from e

    def get_collection(self):
        """
        获取或创建 Chroma 集合。
        首次调用会自动创建集合（若不存在）。

        返回：
            Chroma Collection 对象
        """
        if self._client is None:
            self.connect()

        if self._collection is not None:
            return self._collection

        try:
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={
                    "description": "旅游景点知识库",
                    "distance_metric": self._distance_metric,
                    "embedding_dim": settings.EMBEDDING_DIM,
                },
            )
            print(
                f"[ChromaStore] 集合已就绪: {self._collection_name}"
            )
            return self._collection

        except Exception as e:
            raise RuntimeError(f"获取/创建 Chroma 集合失败: {e}") from e

    def count(self) -> int:
        """返回集合中的文档总数。"""
        try:
            collection = self.get_collection()
            return collection.count()
        except Exception as e:
            print(f"[ChromaStore] 获取文档计数失败: {e}")
            return 0

    def reset(self) -> None:
        """
        删除并重建集合（清空所有数据）。
        危险操作，确认后调用。
        """
        try:
            if self._client is None:
                self.connect()
            self._client.delete_collection(name=self._collection_name)
            self._collection = None
            print(f"[ChromaStore] 集合已删除: {self._collection_name}")
        except Exception as e:
            raise RuntimeError(f"重置 Chroma 集合失败: {e}") from e

    @property
    def client(self):
        """返回原生 Chroma 客户端（供高级操作使用）。"""
        return self._client

    @property
    def is_connected(self) -> bool:
        """检查是否已连接到 Chroma。"""
        return self._client is not None
