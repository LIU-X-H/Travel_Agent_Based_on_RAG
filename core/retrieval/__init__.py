"""
检索模块 (Retrieval)
--------------------
实现多种检索策略：
- 语义检索（基于向量相似度）
- BM25 关键词检索
- 混合检索（语义 + BM25 加权融合）
- 重排序（ReRank 模型精排）
"""

__all__ = [
    "semantic_search",
    "bm25_search",
    "hybrid_search",
    "reranker",
]
