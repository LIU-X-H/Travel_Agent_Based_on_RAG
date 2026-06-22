"""
向量化模块 (Embeddings)
-----------------------
负责加载 BGE Embedding 模型，将文本转换为稠密向量。
支持：
- HuggingFace SentenceTransformer 模型加载
- 批量文本向量化
- 设备自动选择（GPU / CPU）
"""

from .embedding_loader import EmbeddingLoader

__all__ = ["EmbeddingLoader"]
