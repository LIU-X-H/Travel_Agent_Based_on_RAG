"""
旅游景点检索工具（LangChain Agent 标准工具）
============================================
基于 langchain_core.tools.BaseTool 封装，供 LangGraph / LangChain Agent 自动调用。

特性：
- 符合 LangChain 工具标准：定义 args_schema (Pydantic)、description、_run 方法
- 内部复用 ScenicRetriever 混合检索 + 元数据过滤，零重复逻辑
- 检索结果自动格式化为自然语言文本，大模型可直接总结输出
- 完善异常捕获：无结果、参数非法、向量库未初始化均返回友好提示

使用示例：
    from modules.scenic_tool import ScenicSpotRetrieveTool

    tool = ScenicSpotRetrieveTool()
    result = tool._run(
        query="故宫门票多少钱",
        city="北京",
        tags=["世界文化遗产"],
        top_k=3,
    )
    print(result)  # 自然语言格式化文本

Agent 集成示例：
    from langgraph.prebuilt import create_react_agent
    agent = create_react_agent(llm, [tool])
    agent.invoke({"messages": [{"role": "user", "content": "北京有哪些免费5A景点？"}]})
"""

from typing import List, Optional, Type, Dict, Any

from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

from modules.retriever import ScenicRetriever


# ============================================================
# Pydantic 输入参数 Schema
# ============================================================
class ScenicSpotSearchInput(BaseModel):
    """
    旅游景点检索工具入参。

    字段说明：
        query:      用户检索需求，自然语言描述，例如 "推荐北京的免费5A景点"
        city:       限定城市名称，可选。例如 "北京"、"杭州"
        min_ticket: 最低票价（含），可选。0 表示包含免费景点
        max_ticket: 最高票价（含），可选。例如 100
        tags:       景点标签过滤，可选。例如 ["免费", "5A", "古迹"]
        top_k:      返回结果条数，默认 5，范围 1~20
    """
    query: str = Field(
        description="用户检索需求，自然语言描述。例如：'北京故宫门票多少钱'、'杭州有哪些免费5A景点'"
    )
    city: Optional[str] = Field(
        default=None,
        description="限定城市名称。例如 '北京'、'杭州'。不传则不限城市。",
    )
    min_ticket: Optional[int] = Field(
        default=None,
        description="最低票价（元），含此值。0 表示包含免费景点。不传则不设下限。",
    )
    max_ticket: Optional[int] = Field(
        default=None,
        description="最高票价（元），含此值。不传则不设上限。",
    )
    tags: Optional[List[str]] = Field(
        default=None,
        description="景点标签列表，需至少匹配一个。例如 ['5A', '古迹', '免费']。不传则不限标签。",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="返回结果条数，默认 5，最大 20。",
    )


# ============================================================
# 旅游景点检索工具
# ============================================================
class ScenicSpotRetrieveTool(BaseTool):
    """
    旅游景点知识库检索工具

    用途：
        - 供 LangChain / LangGraph Agent 调用
        - 根据用户需求从景点向量库中检索匹配的景点信息
        - 支持城市、票价、标签组合过滤
        - 返回自然语言格式化的结果文本，适合大模型直接总结

    Agent 调用示例：
        tool = ScenicSpotRetrieveTool()
        # Agent 自动填充 args_schema 定义的参数
        tool.invoke({
            "query": "北京有哪些免费5A景点",
            "city": "北京",
            "tags": ["免费", "5A"],
            "top_k": 5,
        })
    """

    name: str = "scenic_spot_search"
    description: str = (
        "从旅游景点知识库中检索匹配的景点信息。"
        "支持按城市名称、票价区间、景点标签（如5A、免费、古迹、自然风光等）组合筛选。"
        "返回自然语言格式的景点描述文本，包含景点名称、城市、票价、等级、标签等关键信息。"
        "适用于：用户询问某城市的景点推荐、特定类型景点查询、票价范围内的景点筛选等场景。"
    )
    args_schema: Type[BaseModel] = ScenicSpotSearchInput

    # ---- 内部依赖（在 __init__ 中设置，避免 Pydantic v2 私有属性问题） ----
    # 不要在类级别声明带下划线的私有属性，否则 Pydantic v2 会创建 ModelPrivateAttr 描述符
    # 导致 LangChain ToolNode 处理时出错: "argument of type 'ModelPrivateAttr' is not iterable"

    def __init__(self, retriever: ScenicRetriever | None = None, **kwargs: Any) -> None:
        """
        初始化检索工具。

        参数：
            retriever: ScenicRetriever 实例。为 None 时自动创建默认实例。
        """
        super().__init__(**kwargs)
        self._retriever = retriever or ScenicRetriever()

    # ============================================================
    # 核心执行逻辑
    # ============================================================
    def _run(
        self,
        query: str,
        city: Optional[str] = None,
        min_ticket: Optional[int] = None,
        max_ticket: Optional[int] = None,
        tags: Optional[List[str]] = None,
        top_k: int = 5,
    ) -> str:
        """
        执行景点检索并返回格式化的自然语言结果。

        参数：
            query:      检索查询文本（必填）
            city:       限定城市（可选）
            min_ticket: 最低票价（可选）
            max_ticket: 最高票价（可选）
            tags:       标签列表（可选）
            top_k:      返回条数

        返回：
            自然语言格式的检索结果字符串，无结果时返回友好提示
        """
        # ---- 1) 输入校验 ----
        if not query or not query.strip():
            return (
                "[ERROR] 检索失败：查询文本(query)不能为空。"
                "请提供一个自然语言查询，例如 '北京有哪些5A景点'。"
            )

        if top_k < 1:
            return f"[ERROR] 检索失败：top_k 必须 >= 1，实际传入 {top_k}。"

        # ---- 2) 构建元数据过滤条件 ----
        try:
            where_clause = self._build_filter_clause(
                city=city,
                min_ticket=min_ticket,
                max_ticket=max_ticket,
            )
        except ValueError as e:
            return f"[ERROR] 过滤参数非法：{e}"

        # ---- 3) 执行混合检索 ----
        try:
            results = self._retriever.hybrid_search(
                query=query.strip(),
                top_k=top_k,
            )
        except Exception as e:
            return (
                f"[ERROR] 检索执行失败：向量库可能未初始化或数据为空。"
                f"请先通过 ScenicDataProcessor 导入景点数据。"
                f"原始错误: {type(e).__name__}: {e}"
            )

        if not results:
            return (
                f"[INFO] 未找到匹配的旅游景点。"
                f"查询条件: query='{query}'"
                f"{', city=' + city if city else ''}"
                f"{', 票价' + str(min_ticket) + '~' + str(max_ticket) + '元' if min_ticket or max_ticket else ''}"
                f"{', tags=' + str(tags) if tags else ''}"
                f"。建议尝试放宽筛选条件或更换关键词。"
            )

        # ---- 4) Python 层过滤（票价、标签由 filter_search 处理，此处做兜底） ----
        if min_ticket is not None or max_ticket is not None:
            results = self._filter_by_ticket(results, min_ticket, max_ticket)

        if tags:
            results = self._filter_by_tags(results, tags)

        if not results:
            return (
                f"[INFO] 经城市/票价/标签筛选后无匹配景点。"
                f"原始混合检索命中 {len(results) + (0 if not results else 0)} 条，"
                f"但都不满足附加筛选条件。"
            )

        # ---- 5) 格式化为自然语言 ----
        return self._format_results(results, query)


    # ============================================================
    # 内部：过滤 & 格式化
    # ============================================================
    @staticmethod
    def _build_filter_clause(
        city: Optional[str] = None,
        min_ticket: Optional[int] = None,
        max_ticket: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        构建 Chroma 元数据过滤条件（供调试/日志使用）。

        返回:
            where 字典，无条件时为 None
        """
        conditions = []
        if city and city.strip():
            conditions.append({"city": city.strip()})
        if min_ticket is not None and max_ticket is not None:
            if min_ticket > max_ticket:
                raise ValueError(
                    f"min_ticket ({min_ticket}) 不能大于 max_ticket ({max_ticket})"
                )
        if min_ticket is not None:
            conditions.append({"ticket": {"$gte": min_ticket}})
        if max_ticket is not None:
            conditions.append({"ticket": {"$lte": max_ticket}})
        if not conditions:
            return None
        elif len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    @staticmethod
    def _filter_by_ticket(
        results: List[Dict[str, Any]],
        min_ticket: Optional[int],
        max_ticket: Optional[int],
    ) -> List[Dict[str, Any]]:
        """Python 层票价后置过滤（兜底）。"""
        filtered = []
        for r in results:
            ticket = r.get("metadata", {}).get("ticket", 0)
            if min_ticket is not None and ticket < min_ticket:
                continue
            if max_ticket is not None and ticket > max_ticket:
                continue
            filtered.append(r)
        return filtered

    @staticmethod
    def _filter_by_tags(
        results: List[Dict[str, Any]],
        tags: List[str],
    ) -> List[Dict[str, Any]]:
        """Python 层标签后置过滤（文档需至少包含一个指定标签）。"""
        if not tags:
            return results
        tag_set = set(tags)
        filtered = []
        for r in results:
            doc_tags = set(r.get("metadata", {}).get("tags", []))
            if doc_tags & tag_set:
                filtered.append(r)
        return filtered

    @staticmethod
    def _format_results(
        results: List[Dict[str, Any]],
        query: str,
    ) -> str:
        """
        将检索结果格式化为自然语言文本。

        输出格式示例：

        [搜索] 为您找到 3 个匹配的旅游景点：

        1. 【故宫博物院】🏛 古迹 · 博物馆
           城市：北京 | 票价：60元 | 等级：5A
           简介：故宫位于北京市中心，是明清两代的皇家宫殿...

        2. 【八达岭长城】🏯 古迹 · 登山
           ...

        [提示] 以上信息基于景点知识库检索，如需更详细信息可继续提问。
        """
        if not results:
            return "未找到匹配的景点。"

        lines: List[str] = []
        lines.append(f"[搜索] 为您找到 {len(results)} 个匹配的旅游景点：")
        lines.append("")

        # 标签对应的 emoji 图标
        tag_icon: Dict[str, str] = {
            "古迹": "#", "博物馆": "#", "寺庙": "#", "石窟": "#",
            "自然风光": "~", "湖泊": "~", "皇家园林": "~",
            "登山": "^", "动物园": "@", "亲子": "@",
            "世界文化遗产": "*", "免费": "[免费]", "熊猫": "@",
            "佛教": "#", "历史遗迹": "#", "水利工程": "~",
        }

        for i, r in enumerate(results, 1):
            meta: Dict[str, Any] = r.get("metadata", {})
            name: str = meta.get("name", "未知景点")
            city: str = meta.get("city", "")
            ticket: float = meta.get("ticket", 0.0)
            level: str = meta.get("level", "")
            doc_tags: List[str] = meta.get("tags", [])

            # 选择 2-3 个标签展示
            display_tags: List[str] = doc_tags[:3]
            tag_parts = []
            for t in display_tags:
                icon = tag_icon.get(t, "")
                tag_parts.append(f"{icon}{t}")
            tag_str: str = " ".join(tag_parts)

            # 票价格式化
            if ticket == 0:
                ticket_str = "免费"
            elif ticket == int(ticket):
                ticket_str = f"{int(ticket)}元"
            else:
                ticket_str = f"{ticket}元"

            # 等级
            level_str: str = f" | {level}级景区" if level else ""

            # 拼接
            lines.append(f"{i}. 【{name}】{tag_str}")
            lines.append(f"   城市：{city} | 票价：{ticket_str}{level_str}")

            # 简介取前 120 字
            content: str = r.get("content", "")
            brief: str = content[:120].replace("\n", " ")
            if len(content) > 120:
                brief += "..."
            lines.append(f"   简介：{brief}")
            lines.append(f"   匹配度：{r.get('score', 0):.2%}")
            lines.append("")

        lines.append("[提示] 以上信息基于景点知识库检索，如需更详细信息可继续提问。")
        return "\n".join(lines)

    # ============================================================
    # 异步（可选）
    # ============================================================
    async def _arun(self, **kwargs: Any) -> str:
        """
        异步检索（当前直接调用同步 _run，后续可替换为异步向量库查询）。
        """
        return self._run(**kwargs)
