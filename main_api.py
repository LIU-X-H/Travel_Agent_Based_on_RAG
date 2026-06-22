"""
旅游景点知识库 RAG 检索服务（FastAPI）
======================================
基于 FastAPI + 已完成模块搭建的 RESTful 检索 API。

特性：
- POST /api/scenic/search  景点混合检索接口
- 自动加载向量库、检索器，启动即用
- Pydantic 请求/响应模型，自动生成 /docs 文档
- 全局异常捕获，统一标准错误码
- CORS 跨域支持，便于前端调用

启动方式：
    python main_api.py                     # 默认 0.0.0.0:8000
    python main_api.py --port 8080         # 自定义端口
    uvicorn main_api:app --reload          # 开发模式热重载

API 文档：
    启动后访问 http://localhost:8000/docs  查看 Swagger UI
    启动后访问 http://localhost:8000/redoc 查看 ReDoc
"""

import sys
import os
from typing import List, Optional

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from config.settings import settings
from modules.retriever import ScenicRetriever
from modules.vector_store import ScenicVectorStore

# ============================================================
# 应用初始化
# ============================================================
app = FastAPI(
    title="旅游景点知识库 RAG 检索服务",
    description=(
        "基于 LangChain + Chroma + BGE 的旅游景点智能检索 API。"
        "支持语义检索、关键词检索、城市/票价/标签组合过滤。"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS 跨域（允许前端调用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# 全局服务状态（启动时初始化，全局复用）
# ============================================================
_retriever: Optional[ScenicRetriever] = None
_vector_store: Optional[ScenicVectorStore] = None


def get_retriever() -> ScenicRetriever:
    """获取全局检索器实例（懒加载）。"""
    global _retriever
    if _retriever is None:
        _retriever = ScenicRetriever()
    return _retriever


def get_vector_store() -> ScenicVectorStore:
    """获取全局向量库实例（懒加载）。"""
    global _vector_store
    if _vector_store is None:
        _vector_store = ScenicVectorStore()
    return _vector_store


# ============================================================
# Pydantic 模型
# ============================================================
class SearchRequest(BaseModel):
    """
    景点检索请求体。

    所有字段均为可选（除 query），未传的筛选项不做限制。
    """
    query: str = Field(
        ...,
        description="检索查询文本，必填。例如：'北京故宫'、'杭州免费5A景点'",
        min_length=1,
        max_length=500,
    )
    city: Optional[str] = Field(
        default=None,
        description="限定城市名称。例如：'北京'、'杭州'",
    )
    min_ticket: Optional[float] = Field(
        default=None,
        ge=0,
        description="最低票价（元），含此值。0 表示包含免费景点",
    )
    max_ticket: Optional[float] = Field(
        default=None,
        ge=0,
        description="最高票价（元），含此值",
    )
    tags: Optional[List[str]] = Field(
        default=None,
        description="景点标签列表，至少匹配一个。例如：['5A', '古迹', '免费']",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="返回结果条数，默认 5，最大 50",
    )


class ScenicItem(BaseModel):
    """单条景点检索结果。"""
    name: str = Field(description="景点名称")
    city: str = Field(description="所在城市")
    ticket: float = Field(description="票价（元），0 表示免费")
    level: str = Field(description="景区等级，如 5A/4A")
    tags: List[str] = Field(description="景点标签列表")
    description: str = Field(description="景点简介文本")
    score: float = Field(description="匹配分数，0.0~1.0，越高越相关")


class SearchResponse(BaseModel):
    """检索接口标准响应。"""
    success: bool = Field(description="是否成功")
    total: int = Field(description="返回结果总数")
    query: str = Field(description="用户原始查询")
    filters: dict = Field(description="实际应用的过滤条件")
    results: List[ScenicItem] = Field(description="检索结果列表")
    message: str = Field(default="", description="附加提示信息")


class ErrorResponse(BaseModel):
    """标准错误响应。"""
    success: bool = Field(default=False)
    error: str = Field(description="错误类型")
    message: str = Field(description="错误描述")


# ============================================================
# 全局异常处理
# ============================================================
@app.exception_handler(400)
async def bad_request_handler(request: Request, exc: Exception) -> JSONResponse:
    """捕获 FastAPI 自动生成的 400 校验错误。"""
    return JSONResponse(
        status_code=400,
        content={
            "success": False,
            "error": "ValidationError",
            "message": str(exc),
        },
    )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """捕获未预期的内部错误。"""
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "InternalServerError",
            "message": f"服务器内部错误: {exc}",
        },
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """兜底全局异常捕获。"""
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": type(exc).__name__,
            "message": str(exc),
        },
    )


# ============================================================
# 根路径 & 健康检查
# ============================================================
@app.get("/", tags=["系统"], response_model=dict)
async def root():
    """
    服务根路径，返回基本信息与可用接口链接。
    """
    store = get_vector_store()
    doc_count: int = store.count() if store else 0
    return {
        "service": "旅游景点知识库 RAG 检索服务",
        "version": "1.0.0",
        "status": "running",
        "vector_store_docs": doc_count,
        "docs": "/docs",
        "redoc": "/redoc",
        "api_search": "/api/scenic/search",
    }


@app.get("/api/health", tags=["系统"], response_model=dict)
async def health_check():
    """
    健康检查接口，返回向量库状态。
    """
    try:
        store = get_vector_store()
        return {
            "status": "healthy",
            "vector_store_docs": store.count() if store else 0,
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)},
        )


# ============================================================
# 核心检索接口
# ============================================================
@app.post(
    "/api/scenic/search",
    tags=["检索"],
    response_model=SearchResponse,
    responses={
        400: {"model": ErrorResponse, "description": "参数校验失败"},
        500: {"model": ErrorResponse, "description": "服务器内部错误"},
    },
)
async def scenic_search(request: SearchRequest):
    """
    旅游景点混合检索接口。

    内部调用 BM25 + 向量语义混合检索，支持按城市、票价、标签组合过滤。

    请求示例：
    ```json
    {
        "query": "北京故宫",
        "city": "北京",
        "min_ticket": 0,
        "max_ticket": 100,
        "tags": ["5A", "世界文化遗产"],
        "top_k": 5
    }
    ```

    返回示例：
    ```json
    {
        "success": true,
        "total": 2,
        "query": "北京故宫",
        "filters": {"city": "北京", "min_ticket": 0, "max_ticket": 100, "tags": ["5A", "世界文化遗产"]},
        "results": [
            {
                "name": "故宫博物院",
                "city": "北京",
                "ticket": 60.0,
                "level": "5A",
                "tags": ["世界文化遗产", "古迹", "博物馆"],
                "description": "故宫是明清两代的皇家宫殿...",
                "score": 0.8523
            }
        ],
        "message": ""
    }
    ```
    """
    retriever = get_retriever()

    # ---- 1) 构建过滤条件 ----
    filters: dict = {
        "city": request.city,
        "min_ticket": request.min_ticket,
        "max_ticket": request.max_ticket,
        "tags": request.tags,
    }

    # ---- 2) 执行混合检索 ----
    try:
        raw_results = retriever.hybrid_search(
            query=request.query.strip(),
            top_k=request.top_k,
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "RetrievalError",
                "message": (
                    f"检索执行失败。请确认向量库已导入数据。"
                    f"原始错误: {type(e).__name__}: {e}"
                ),
            },
        )

    # ---- 3) 后置过滤（票价 / 标签） ----
    results = _apply_post_filters(
        raw_results,
        min_ticket=request.min_ticket,
        max_ticket=request.max_ticket,
        tags=request.tags,
    )

    # ---- 4) 无结果处理 ----
    message: str = ""
    if not results:
        message = (
            f"未找到匹配的景点。建议放宽筛选条件。"
            f"原始检索命中 {len(raw_results)} 条，经过滤后剩余 0 条。"
        )

    # ---- 5) 构建响应 ----
    scenic_items: List[ScenicItem] = []
    for r in results[: request.top_k]:
        meta = r.get("metadata", {})
        scenic_items.append(
            ScenicItem(
                name=meta.get("name", ""),
                city=meta.get("city", ""),
                ticket=meta.get("ticket", 0.0),
                level=meta.get("level", ""),
                tags=meta.get("tags", []),
                description=r.get("content", ""),
                score=r.get("score", 0.0),
            )
        )

    return SearchResponse(
        success=True,
        total=len(scenic_items),
        query=request.query,
        filters=filters,
        results=scenic_items,
        message=message,
    )


# ============================================================
# 后置过滤（复刻 retriever 逻辑，确保 API 层独立可控）
# ============================================================
def _apply_post_filters(
    results: list,
    min_ticket: Optional[float] = None,
    max_ticket: Optional[float] = None,
    tags: Optional[List[str]] = None,
) -> list:
    """
    对混合检索结果做票价、标签后置过滤。

    参数：
        results:     hybrid_search 返回的原始结果列表
        min_ticket:  最低票价
        max_ticket:  最高票价
        tags:        标签列表

    返回：
        过滤后的结果列表
    """
    filtered = list(results)

    # 票价过滤
    if min_ticket is not None:
        filtered = [
            r for r in filtered
            if r.get("metadata", {}).get("ticket", 0) >= min_ticket
        ]
    if max_ticket is not None:
        filtered = [
            r for r in filtered
            if r.get("metadata", {}).get("ticket", 0) <= max_ticket
        ]

    # 标签过滤
    if tags:
        tag_set = set(tags)
        filtered = [
            r for r in filtered
            if tag_set & set(r.get("metadata", {}).get("tags", []))
        ]

    return filtered


# ============================================================
# 启动入口
# ============================================================
if __name__ == "__main__":
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser(
        description="旅游景点知识库 RAG 检索服务"
    )
    parser.add_argument(
        "--host",
        type=str,
        default=settings.API_HOST,
        help=f"监听地址，默认 {settings.API_HOST}",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=settings.API_PORT,
        help=f"监听端口，默认 {settings.API_PORT}",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        default=False,
        help="开启热重载（开发模式）",
    )
    args = parser.parse_args()

    # 启动日志
    print("=" * 60)
    print("  旅游景点知识库 RAG 检索服务")
    print("=" * 60)
    print(f"  监听地址: http://{args.host}:{args.port}")
    print(f"  API 文档: http://{args.host}:{args.port}/docs")
    print(f"  健康检查: http://{args.host}:{args.port}/api/health")
    print("=" * 60)

    # 预热：检查向量库数据
    try:
        store = get_vector_store()
        doc_count = store.count()
        if doc_count == 0:
            print(
                "[WARN] 向量库为空！请先运行 data_processor 导入景点数据。"
                "示例: python -c \"from modules.data_processor import ScenicDataProcessor; "
                "ScenicDataProcessor().process_and_import()\""
            )
        else:
            print(f"[OK] 向量库已加载: {doc_count} 条景点数据")
            # 预热检索器
            get_retriever()
            print("[OK] 检索器已就绪")
    except Exception as e:
        print(f"[WARN] 向量库预热失败: {e}")

    uvicorn.run(
        "main_api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
