"""
旅游景点检索工具演示脚本
========================
演示 ScenicSpotRetrieveTool 的多场景调用。

前置条件：
    python demo_tool.py

    首次运行会自动检测向量库是否已存在数据，若不存在则自动导入示例数据。
"""

import sys
import os

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.scenic_tool import ScenicSpotRetrieveTool


# ============================================================
# 辅助：确保向量库中有数据
# ============================================================
def ensure_data_ready():
    """检查向量库是否有数据，若无则自动导入示例数据。"""
    from modules.vector_store import ScenicVectorStore

    store = ScenicVectorStore()
    if store.count() > 0:
        print(f"[OK] 向量库已有 {store.count()} 条数据，跳过导入\n")
        return

    print("[WARN] 向量库为空，正在自动导入示例数据...\n")

    from modules.data_processor import ScenicDataProcessor
    processor = ScenicDataProcessor()
    try:
        count = processor.process_and_import()
        print(f"[OK] 已导入 {count} 条示例景点数据\n")
    except Exception as e:
        print(f"[WARN] 自动导入失败: {e}")
        print("请手动放入 data/raw/*.json 文件后重试")
        sys.exit(1)


# ============================================================
# 演示
# ============================================================
def demo():
    """运行多场景检索演示。"""
    print("=" * 60)
    print("  [旅游景点检索工具 ScenicSpotRetrieveTool 演示]")
    print("=" * 60)

    ensure_data_ready()

    tool = ScenicSpotRetrieveTool()

    # ---- 场景 1: 基础语义检索 ----
    print("\n" + "─" * 60)
    print("场景 1: 基础语义检索：'北京故宫'")
    print("─" * 60)
    result = tool._run(query="北京故宫")
    print(result)

    # ---- 场景 2: 指定城市 ----
    print("\n" + "─" * 60)
    print("场景 2:  指定城市检索：'杭州', query='西湖有什么好玩的'")
    print("─" * 60)
    result = tool._run(query="西湖有什么好玩的", city="杭州", top_k=3)
    print(result)

    # ---- 场景 3: 免费景点筛选 ----
    print("\n" + "─" * 60)
    print("场景 3:  免费景点筛选：query='景点', tags=['免费']")
    print("─" * 60)
    result = tool._run(query="景点", tags=["免费"], top_k=5)
    print(result)

    # ---- 场景 4: 票价区间筛选 ----
    print("\n" + "─" * 60)
    print("场景 4:  票价区间筛选：query='古迹', min_ticket=50, max_ticket=100")
    print("─" * 60)
    result = tool._run(query="古迹", min_ticket=50, max_ticket=100, top_k=5)
    print(result)

    # ---- 场景 5: 组合筛选 ----
    print("\n" + "─" * 60)
    print("场景 5:  组合筛选：query='自然', city='杭州', tags=['免费', '自然风光']")
    print("─" * 60)
    result = tool._run(
        query="自然风光",
        city="杭州",
        tags=["免费", "自然风光"],
        top_k=5,
    )
    print(result)

    # ---- 场景 6: 无结果 ----
    print("\n" + "─" * 60)
    print("场景 6:  无结果处理：query='火星基地', city='火星'")
    print("─" * 60)
    result = tool._run(query="火星基地", city="火星", top_k=5)
    print(result)

    # ---- 场景 7: Agent 调用模拟 ----
    print("\n" + "─" * 60)
    print("场景 7:  Agent 调用模拟（invoke 方式）")
    print("─" * 60)
    agent_input = {
        "query": "北京有哪些免费5A景点",
        "city": "北京",
        "tags": ["免费", "5A"],
        "top_k": 3,
    }
    print(f"Agent 输入: {agent_input}")
    result = tool.invoke(agent_input)
    print(result)

    print("\n" + "=" * 60)
    print("  [OK] 演示完成")
    print("=" * 60)


if __name__ == "__main__":
    demo()
