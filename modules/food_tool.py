"""
美食推荐工具
============
基于 langchain_core.tools.BaseTool 封装，供 Agent 自动调用。

数据：data/raw/food_data.json（30 道城市代表性美食）
存储：Chroma 向量库 collection=food_recommendations
检索：BGE 语义检索 + 城市/菜系过滤

使用示例：
    from modules.food_tool import FoodTool
    tool = FoodTool()
    print(tool._run(query="辣的火锅", city="成都"))
"""

import json
import os
from typing import Optional, List, Type, Dict, Any

from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

from config.settings import settings
from modules.vector_store import ScenicVectorStore
from modules.embedding import EmbeddingModel
from langchain_core.documents import Document


# ============================================================
# Pydantic 输入参数 Schema
# ============================================================
class FoodSearchInput(BaseModel):
    """美食检索入参。"""
    query: str = Field(
        description="检索查询，自然语言描述。例如 '成都火锅'、'清淡的杭州菜'、'特色小吃'"
    )
    city: Optional[str] = Field(
        default=None,
        description="限定城市名称。例如 '北京'、'成都'。不传则不限城市。"
    )
    cuisine_type: Optional[str] = Field(
        default=None,
        description="菜系类型。例如 '川菜'、'粤菜'、'清真'。不传则不限菜系。"
    )
    top_k: int = Field(
        default=5, ge=1, le=10,
        description="返回结果条数，默认 5"
    )


# ============================================================
# 美食检索工具
# ============================================================
class FoodTool(BaseTool):
    """
    城市美食推荐工具。

    从美食知识库中检索匹配的菜品，支持城市和菜系过滤。
    首次使用时自动从 JSON 导入数据到 Chroma。
    """

    name: str = "search_food"
    description: str = (
        "从美食知识库中检索城市特色美食和餐厅推荐。"
        "支持按城市名称、菜系类型（川菜/粤菜/京菜/清真等）过滤。"
        "返回菜品名称、人均价格、推荐餐厅、特色介绍等信息。"
        "适用于：用户询问'某地有什么好吃的''推荐当地美食''哪里吃正宗某菜'等场景。"
        "参数：query（检索文本）、city（可选城市）、cuisine_type（可选菜系）、top_k（返回数）。"
    )
    args_schema: Type[BaseModel] = FoodSearchInput

    # 注意：不要在类级别声明带下划线的私有属性（带类型注解），
    # 否则 Pydantic v2 会创建 ModelPrivateAttr 描述符，
    # 导致 LangChain 处理工具时引发 "argument of type 'ModelPrivateAttr' is not iterable" 错误。
    # 所有内部属性改在 __init__ 中通过 self.xxx 直接设置（无类型注解）。

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # 以下为实例属性（非 Pydantic 字段/私有属性），避免 ModelPrivateAttr 问题
        self._store: Optional[ScenicVectorStore] = None
        self._food_collection: str = "food_recommendations"
        self._imported: bool = False
        self._ensure_food_data()

    # ============================================================
    # 数据导入
    # ============================================================
    def _ensure_food_data(self) -> None:
        """确保美食数据已导入 Chroma。"""
        store = ScenicVectorStore(collection_name=self._food_collection)
        if store.count() > 0:
            self._store = store
            self._imported = True
            return

        # ---- 从 JSON 加载 ----
        json_path = os.path.join(
            settings.RAW_DATA_DIR, "food_data.json"
        )
        if not os.path.exists(json_path):
            print(f"[FoodTool] 未找到 {json_path}，美食工具将返回空结果")
            self._store = store
            return

        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # ---- 转 Document ----
        docs: List[Document] = []
        for item in raw:
            name = item.get("name", "")
            city = item.get("city", "")
            cuisine = item.get("cuisine", "")
            price = item.get("price", "")
            desc = item.get("description", "")

            page_content = (
                f"{name}是{city}的{cuisine}名菜。{desc}人均{price}。"
            )

            docs.append(Document(
                page_content=page_content,
                metadata={
                    "name": name,
                    "city": city,
                    "cuisine": cuisine,
                    "price": price,
                    "food_type": item.get("type", ""),
                    "recommend": item.get("recommend", ""),
                    "address": item.get("address", ""),
                },
            ))

        # ---- 入库 ----
        store.add_scenic_docs(docs, batch_size=16)
        self._store = store
        self._imported = True
        print(f"[FoodTool] 已导入 {len(docs)} 道美食")

    # ============================================================
    # 检索
    # ============================================================
    def _run(
        self,
        query: str,
        city: Optional[str] = None,
        cuisine_type: Optional[str] = None,
        top_k: int = 5,
    ) -> str:
        """
        检索美食并返回格式化文本。

        参数与 ScenicSpotRetrieveTool 风格一致。
        """
        if not query.strip():
            return "[ERROR] 请提供检索关键词，例如 '火锅'、'特色小吃'。"

        store = self._store or ScenicVectorStore(
            collection_name=self._food_collection
        )

        # ---- 检索 ----
        results = store.base_similarity_search(
            query=query.strip(), top_k=top_k * 2  # 多取一些做后置过滤
        )

        if not results:
            return (
                f"未找到与 '{query}' 匹配的美食。"
                f"建议尝试其他关键词或扩大搜索范围。"
            )

        # ---- 后置过滤：城市 / 菜系 ----
        if city and city.strip():
            results = [
                r for r in results
                if r["metadata"].get("city") == city.strip()
            ]
        if cuisine_type and cuisine_type.strip():
            results = [
                r for r in results
                if r["metadata"].get("cuisine") == cuisine_type.strip()
            ]

        if not results:
            hint = f"{city}的" if city else ""
            hint2 = f"{cuisine_type}菜" if cuisine_type else ""
            return (
                f"未找到{hint}{hint2}相关美食。建议尝试其他筛选条件。"
            )

        # ---- 格式化 ----
        results = results[:top_k]
        lines = [f"为您找到 {len(results)} 道匹配美食："]

        for i, r in enumerate(results, 1):
            meta = r["metadata"]
            name = meta.get("name", "")
            fc = meta.get("cuisine", "")
            price = meta.get("price", "")
            rec = meta.get("recommend", "")
            c = meta.get("city", "")
            addr = meta.get("address", "")

            line = f"\n{i}. {name}（{fc}）"
            if c:
                line += f" | {c}"
            line += f" | {price}"
            if rec:
                line += f"\n   推荐：{rec}"
            if addr:
                line += f"\n   地址：{addr}"
            line += f"\n   {r['content'][:100]}..."
            lines.append(line)

        lines.append("\n以上信息来自美食知识库，推荐店铺可能排队建议提前预约。")
        return "\n".join(lines)
