"""
配置模块
-------
统一管理项目所有可调参数，包括：
- 向量库路径与连接参数
- Embedding 模型名称与设备
- 检索 top_k、相似度阈值
- 混合检索权重（BM25 + 语义）
- ReRank 重排序开关与模型
"""

from .settings import Settings

__all__ = ["Settings"]
