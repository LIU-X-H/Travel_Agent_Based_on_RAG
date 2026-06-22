"""
核心模块
-------
RAG 检索管线的核心组件：
- embeddings/  : 文本向量化（BGE 模型加载与调用）
- vectorstore/ : 向量数据库（Chroma 连接与集合管理）
- retrieval/   : 检索策略（语义检索、BM25 关键词检索、混合检索、重排序）
- llm/         : 大语言模型（API 调用、提示词模板）
"""

__all__ = [
    "embeddings",
    "vectorstore",
    "retrieval",
    "llm",
]
