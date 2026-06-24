"""
路由模块
--------
按资源拆分 API 路由定义。
"""

__all__ = []
from .agent import router as agent_router

__all__ = ["agent_router"]
