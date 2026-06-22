"""
大模型模块 (LLM)
----------------
负责大语言模型的连接与调用：
- OpenAI API 兼容接口封装
- 提示词模板管理
- 流式/非流式生成
- 后续可扩展 LangGraph 智能体
"""

from .llm_client import LLMClient

__all__ = ["LLMClient"]
