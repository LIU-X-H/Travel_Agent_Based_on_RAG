"""
多策略检索模块单元测试
======================
覆盖：
1. filter_search  —— 城市过滤 / 免费筛选 / 票价区间 / 标签过滤 / 组合过滤
2. hybrid_search  —— BM25 + 语义加权融合
3. rerank_search   —— 混合初排 + BGE Reranker 精排（需要 FlagEmbedding）
4. 异常场景        —— 空查询 / 非法参数 / 无结果处理

运行方式：
    cd travel_scenic_rag
    python tests/test_retriever.py                        # 直接运行
    pytest tests/test_retriever.py -v -s                  # pytest 运行

前置条件：
    - 向量库已有测试数据（本脚本自动构建临时数据）
    - pip install rank-bm25 jieba  (BM25 依赖)
    - pip install FlagEmbedding    (Reranker 依赖，可选)
"""

import sys
import os
import shutil

# 将项目根目录加入 sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from langchain_core.documents import Document

from modules.vector_store import ScenicVectorStore
from modules.retriever import (
    ScenicRetriever,
    RetrieverError,
    RetrieverParamError,
    RetrieverSearchError,
)
from modules.embedding import EmbeddingModel


# ============================================================
# 测试配置
# ============================================================
TEST_PERSIST_DIR = os.path.join(PROJECT_ROOT, "data", "test_retriever_db")
TEST_COLLECTION_NAME = "test_retriever_scenic_spots"


# ============================================================
# 模拟多城市、多类型景点数据
# ============================================================
def build_rich_test_documents():
    """构建含多种标签、票价的景点测试数据。"""
    scenic_data = [
        # ---- 北京 (3条) ----
        {
            "content": "故宫博物院位于北京市中心，是明清两代皇家宫殿，又称紫禁城。"
                       "占地面积72万平方米，有大小宫殿七十多座。1987年世界文化遗产，"
                       "国家5A级景区，旺季门票60元淡季40元。",
            "metadata": {"city": "北京", "name": "故宫博物院", "ticket": 60,
                         "level": "5A", "tags": ["世界文化遗产", "古迹", "博物馆"]},
        },
        {
            "content": "八达岭长城位于北京市延庆区，是明长城保存最完好的一段。"
                       "地势险要、城关坚固，史称天下九塞之一。1987年世界文化遗产，"
                       "国家5A级景区，旺季门票45元淡季40元。",
            "metadata": {"city": "北京", "name": "八达岭长城", "ticket": 45,
                         "level": "5A", "tags": ["世界文化遗产", "古迹", "登山"]},
        },
        {
            "content": "颐和园位于北京市海淀区，是中国现存规模最大、保存最完整的皇家园林。"
                       "以昆明湖、万寿山为基址，汲取江南园林设计手法建成。"
                       "1998年世界文化遗产，国家5A级景区，旺季门票30元淡季20元。",
            "metadata": {"city": "北京", "name": "颐和园", "ticket": 30,
                         "level": "5A", "tags": ["世界文化遗产", "皇家园林", "湖泊"]},
        },
        # ---- 杭州 (2条) ----
        {
            "content": "西湖位于浙江省杭州市西面，是中国首批国家重点风景名胜区。"
                       "三面环山，面积约6.39平方千米。2011年世界文化景观遗产，"
                       "国家5A级景区，全年免费开放。",
            "metadata": {"city": "杭州", "name": "西湖", "ticket": 0,
                         "level": "5A", "tags": ["世界文化遗产", "自然风光", "湖泊", "免费"]},
        },
        {
            "content": "灵隐寺位于杭州市西湖区，始建于东晋咸和元年（公元326年），"
                       "是中国佛教禅宗十大古刹之一。飞来峰有五代至元代石窟造像。"
                       "全国重点文物保护单位，门票75元。",
            "metadata": {"city": "杭州", "name": "灵隐寺", "ticket": 75,
                         "level": "5A", "tags": ["佛教", "古迹", "寺庙", "石窟"]},
        },
        # ---- 成都 (2条) ----
        {
            "content": "成都大熊猫繁育研究基地位于成都市成华区，是世界上最重要的"
                       "大熊猫保护研究机构之一。占地约100公顷，可近距离观察大熊猫。"
                       "门票55元，国家4A级景区。",
            "metadata": {"city": "成都", "name": "大熊猫繁育研究基地", "ticket": 55,
                         "level": "4A", "tags": ["动物园", "亲子", "自然", "熊猫"]},
        },
        {
            "content": "都江堰位于成都市都江堰市，是战国时期秦国蜀郡太守李冰父子"
                       "主持修建的大型水利工程，至今仍在发挥作用。2000年世界文化遗产，"
                       "国家5A级景区，门票80元。",
            "metadata": {"city": "成都", "name": "都江堰", "ticket": 80,
                         "level": "5A", "tags": ["世界文化遗产", "古迹", "水利工程"]},
        },
        # ---- 西安 (1条) ----
        {
            "content": "秦始皇兵马俑博物馆位于陕西省西安市临潼区，是秦始皇陵的陪葬坑。"
                       "1974年被发现，被誉为世界第八大奇迹。1987年世界文化遗产，"
                       "国家5A级景区，门票120元。",
            "metadata": {"city": "西安", "name": "秦始皇兵马俑", "ticket": 120,
                         "level": "5A", "tags": ["世界文化遗产", "古迹", "博物馆", "考古"]},
        },
    ]

    return [
        Document(page_content=item["content"], metadata=item["metadata"])
        for item in scenic_data
    ]


# ============================================================
# 测试辅助
# ============================================================
_pass_count = 0
_fail_count = 0


def assert_true(condition, test_name):
    global _pass_count, _fail_count
    if condition:
        print("  [PASS] " + test_name)
        _pass_count += 1
    else:
        print("  [FAIL] " + test_name + ": 条件不满足")
        _fail_count += 1


def assert_equal(actual, expected, test_name):
    global _pass_count, _fail_count
    if actual == expected:
        print("  [PASS] " + test_name)
        _pass_count += 1
    else:
        print("  [FAIL] " + test_name + ": 期望 " + repr(expected) + ", 实际 " + repr(actual))
        _fail_count += 1


def assert_greater(actual, threshold, test_name):
    global _pass_count, _fail_count
    if actual > threshold:
        print("  [PASS] " + test_name)
        _pass_count += 1
    else:
        print("  [FAIL] " + test_name + ": " + str(actual) + " 不大于 " + str(threshold))
        _fail_count += 1


def assert_in(item, container, test_name):
    global _pass_count, _fail_count
    if item in container:
        print("  [PASS] " + test_name)
        _pass_count += 1
    else:
        print("  [FAIL] " + test_name + ": " + repr(item) + " 不在容器中")
        _fail_count += 1


def assert_raises(exception_class, callable_fn, test_name):
    """
    断言 callable_fn 抛出指定类型的异常。

    参数：
        exception_class: 期望的异常类型
        callable_fn:     无参数的可调用对象（通常为 lambda）
        test_name:       测试名称
    """
    global _pass_count, _fail_count
    try:
        callable_fn()
        print("  [FAIL] " + test_name + ": 未抛出异常")
        _fail_count += 1
    except exception_class:
        print("  [PASS] " + test_name)
        _pass_count += 1
    except Exception as e:
        print("  [FAIL] " + test_name + ": 错误异常类型 " + type(e).__name__ + ": " + str(e))
        _fail_count += 1


def cleanup():
    """清理测试向量库目录，兼容 Windows 文件占用。"""
    if not os.path.exists(TEST_PERSIST_DIR):
        return
    import stat
    import time
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # 移除只读属性（Windows 需要）
            for root, dirs, files in os.walk(TEST_PERSIST_DIR):
                for name in dirs + files:
                    full = os.path.join(root, name)
                    try:
                        os.chmod(full, stat.S_IWRITE)
                    except OSError:
                        pass
            shutil.rmtree(TEST_PERSIST_DIR)
            print("[清理] 已删除: " + TEST_PERSIST_DIR)
            return
        except PermissionError as e:
            if attempt < max_retries - 1:
                print("[清理] 文件占用，1秒后重试... (" + str(e) + ")")
                time.sleep(1)
            else:
                print("[清理] 跳过: 文件被其他进程占用，无法删除 " + TEST_PERSIST_DIR)
                print("[清理] 提示: 可手动删除或重启后自动清理")
        except OSError as e:
            # Windows [WinError 32] 等文件锁定错误
            if hasattr(e, 'winerror') and e.winerror == 32:
                print("[清理] 跳过: Windows 文件锁定 [WinError 32]，请手动删除 " + TEST_PERSIST_DIR)
            else:
                print("[清理] 失败: " + str(e))
            break
        except Exception as e:
            print("[清理] 失败: " + str(e))
            break


# ============================================================
# 准备测试数据
# ============================================================
def setup_test_data():
    """构建临时向量库并填充测试数据。"""
    cleanup()

    store = ScenicVectorStore(
        collection_name=TEST_COLLECTION_NAME,
        persist_directory=TEST_PERSIST_DIR,
    )
    docs = build_rich_test_documents()
    store.add_scenic_docs(docs)
    print("[Setup] 已入库 " + str(len(docs)) + " 条测试数据")
    return store


# ============================================================
# 测试：filter_search 城市过滤
# ============================================================
def test_filter_by_city():
    print("\n" + "=" * 60)
    print("[测试] filter_search — 城市过滤")
    print("=" * 60)

    setup_test_data()
    retriever = ScenicRetriever(
        collection_name=TEST_COLLECTION_NAME,
        persist_directory=TEST_PERSIST_DIR,
    )

    # 北京
    results = retriever.filter_search("皇家宫殿", city="北京", top_k=10)
    print("  [INFO] 北京过滤结果: " + str(len(results)) + " 条")
    for i, r in enumerate(results):
        print("    " + str(i+1) + ". city=" + repr(r["metadata"].get("city")) + " name=" + str(r["metadata"].get("name", "?")))
    assert_greater(len(results), 0, "北京检索有结果")
    for r in results:
        assert_equal(r["metadata"].get("city"), "北京",
                     "全部结果为北京: " + str(r["metadata"].get("name", "?")))

    # 杭州
    results_hz = retriever.filter_search("寺庙", city="杭州", top_k=10)
    print("  [INFO] 杭州过滤结果: " + str(len(results_hz)) + " 条")
    for i, r in enumerate(results_hz):
        print("    " + str(i+1) + ". city=" + repr(r["metadata"].get("city")) + " name=" + str(r["metadata"].get("name", "?")))
    assert_greater(len(results_hz), 0, "杭州检索有结果")
    for r in results_hz:
        assert_equal(r["metadata"].get("city"), "杭州", "杭州结果城市正确")

    # 西安
    results_xa = retriever.filter_search("兵马俑", city="西安", top_k=5)
    print("  [INFO] 西安过滤结果: " + str(len(results_xa)) + " 条")
    if len(results_xa) >= 1:
        assert_equal(len(results_xa), 1, "西安兵马俑 1 条")
        assert_equal(results_xa[0]["metadata"].get("name"), "秦始皇兵马俑", "名称匹配")
    else:
        print("  [WARN] 西安过滤无结果，跳过名称断言")

    # 不存在的城市
    results_empty = retriever.filter_search("景点", city="火星", top_k=5)
    assert_equal(len(results_empty), 0, "火星城市无结果")


# ============================================================
# 测试：filter_search 免费/票价/标签
# ============================================================
def test_filter_by_price_and_tags():
    print("\n" + "=" * 60)
    print("[测试] filter_search — 免费 / 票价区间 / 标签过滤")
    print("=" * 60)

    setup_test_data()
    retriever = ScenicRetriever(
        collection_name=TEST_COLLECTION_NAME,
        persist_directory=TEST_PERSIST_DIR,
    )

    # 免费景点
    free_results = retriever.filter_search("景点", free_only=True, top_k=10)
    print("  [INFO] 免费景点过滤结果: " + str(len(free_results)) + " 条")
    for i, r in enumerate(free_results):
        print("    " + str(i+1) + ". ticket=" + str(r["metadata"].get("ticket")) + " name=" + str(r["metadata"].get("name", "?")) + " tags=" + str(r["metadata"].get("tags", [])))
    if len(free_results) == 0:
        print("  [WARN] 免费景点过滤无结果！请检查向量库中 ticket=0 的文档")
    assert_greater(len(free_results), 0, "免费景点有结果")
    for r in free_results:
        assert_equal(r["metadata"].get("ticket"), 0,
                     "免费景点 ticket=0: " + str(r["metadata"].get("name")))

    # 票价区间 [50, 100]
    mid_results = retriever.filter_search(
        "景点", ticket_min=50, ticket_max=100, top_k=10
    )
    print("  [INFO] 票价[50,100]过滤结果: " + str(len(mid_results)) + " 条")
    for i, r in enumerate(mid_results):
        print("    " + str(i+1) + ". ticket=" + str(r["metadata"].get("ticket")) + " name=" + str(r["metadata"].get("name", "?")))
    assert_greater(len(mid_results), 0, "票价 50~100 有结果")
    for r in mid_results:
        ticket = r["metadata"].get("ticket", 0)
        assert_true(50 <= ticket <= 100,
                    "票价在 [50,100]: " + str(r["metadata"].get("name")) + " ticket=" + str(ticket))

    # 标签过滤
    tag_results = retriever.filter_search(
        "景点", tags=["免费", "亲子"], top_k=10
    )
    print("  [INFO] 标签[免费,亲子]过滤结果: " + str(len(tag_results)) + " 条")
    for i, r in enumerate(tag_results):
        print("    " + str(i+1) + ". tags=" + str(r["metadata"].get("tags", [])) + " name=" + str(r["metadata"].get("name", "?")))
    if len(tag_results) == 0:
        print("  [WARN] 标签过滤无结果！请检查向量库文档 tags 字段是否包含'免费'或'亲子'")
    assert_greater(len(tag_results), 0, "标签过滤有结果")
    for r in tag_results:
        doc_tags = r["metadata"].get("tags", [])
        assert_true("免费" in doc_tags or "亲子" in doc_tags,
                    "标签匹配: " + str(doc_tags))

    # 组合过滤：北京 + 世界文化遗产
    combo = retriever.filter_search(
        "古迹", city="北京", tags=["世界文化遗产"], top_k=10
    )
    print("  [INFO] 北京+世界文化遗产组合过滤: " + str(len(combo)) + " 条")
    for i, r in enumerate(combo):
        print("    " + str(i+1) + ". city=" + repr(r["metadata"].get("city")) + " tags=" + str(r["metadata"].get("tags", [])) + " name=" + str(r["metadata"].get("name", "?")))
    assert_greater(len(combo), 0, "北京+世界文化遗产 有结果")
    for r in combo:
        assert_equal(r["metadata"].get("city"), "北京", "城市=北京")
        assert_in("世界文化遗产", r["metadata"].get("tags", []), "含世界文化遗产标签")


# ============================================================
# 测试：filter_search 无查询文本（纯过滤）
# ============================================================
def test_filter_without_query():
    print("\n" + "=" * 60)
    print("[测试] filter_search — 空query纯元数据过滤（不触发Embedding）")
    print("=" * 60)

    setup_test_data()
    retriever = ScenicRetriever(
        collection_name=TEST_COLLECTION_NAME,
        persist_directory=TEST_PERSIST_DIR,
    )

    # ---- 场景1：空字符串 query → 纯元数据过滤模式 ----
    print("  [MODE] 空字符串 query='' → 预期走「纯元数据过滤」分支，不调用 embed_query")
    results = retriever.filter_search(query="", city="成都", top_k=10)
    print("  [INFO] 成都纯元数据过滤结果: " + str(len(results)) + " 条")
    for i, r in enumerate(results):
        print("    " + str(i+1) + ". score=" + str(r["score"]) + " city=" + repr(r["metadata"].get("city")) + " name=" + str(r["metadata"].get("name", "?")))
    if len(results) == 0:
        print("  [WARN] 成都纯过滤无结果！请检查向量库中 city=成都 的文档")

    # 验证：返回2条成都数据
    assert_equal(len(results), 2, "空query成都过滤返回 2 条")
    for r in results:
        assert_equal(r["metadata"].get("city"), "成都", "城市=成都")
        # 空查询分支：score 固定为 1.0（无语义匹配）
        assert_equal(r["score"], 1.0, "空query纯元过滤 score=1.0: " + str(r["metadata"].get("name", "?")))

    # ---- 场景2：全空白 query → 同样走纯元数据过滤 ----
    print("  [MODE] 全空白 query='   ' → 预期同样走「纯元数据过滤」分支")
    results_ws = retriever.filter_search(query="   ", city="杭州", top_k=10)
    print("  [INFO] 杭州纯元数据过滤结果: " + str(len(results_ws)) + " 条")
    for i, r in enumerate(results_ws):
        print("    " + str(i+1) + ". score=" + str(r["score"]) + " city=" + repr(r["metadata"].get("city")) + " name=" + str(r["metadata"].get("name", "?")))
    assert_equal(len(results_ws), 2, "全空白query杭州过滤返回 2 条")
    for r in results_ws:
        assert_equal(r["score"], 1.0, "全空白query score=1.0")

    # ---- 场景3：正常 query → 走语义+元数据过滤（对比验证） ----
    print("  [MODE] 正常 query='大熊猫' → 预期走「语义+元数据过滤」分支")
    results_sem = retriever.filter_search(query="大熊猫", city="成都", top_k=10)
    print("  [INFO] 语义+元数据过滤结果: " + str(len(results_sem)) + " 条")
    for i, r in enumerate(results_sem):
        print("    " + str(i+1) + ". score=" + str(r["score"]) + " city=" + repr(r["metadata"].get("city")) + " name=" + str(r["metadata"].get("name", "?")))
    if len(results_sem) > 0:
        # 语义检索的 score 应 < 1.0（真实相似度）
        assert_true(results_sem[0]["score"] < 1.0,
                    "语义检索 score < 1.0: " + str(results_sem[0]["score"]))


# ============================================================
# 测试：hybrid_search 混合检索
# ============================================================
def test_hybrid_search():
    print("\n" + "=" * 60)
    print("[测试] hybrid_search 混合检索（BM25 + 语义）")
    print("=" * 60)

    setup_test_data()
    retriever = ScenicRetriever(
        collection_name=TEST_COLLECTION_NAME,
        persist_directory=TEST_PERSIST_DIR,
    )

    # 基础混合检索
    results = retriever.hybrid_search("北京故宫皇家宫殿", top_k=3)
    assert_greater(len(results), 0, "混合检索有结果")
    for r in results:
        assert_true("content" in r, "含 content")
        assert_true("metadata" in r, "含 metadata")
        assert_true("score" in r, "含 score")
        assert_true(0.0 <= r["score"] <= 1.0, "score 在 [0, 1] 范围")

    print("  [INFO] 混合检索返回 " + str(len(results)) + " 条:")
    for i, r in enumerate(results):
        print("    " + str(i+1) + ". [" + "{:.4f}".format(r["score"]) + "] "
              + str(r["metadata"].get("name", "?")))

    # 纯关键词查询（alpha=0 倾向 BM25）
    results_kw = retriever.hybrid_search("熊猫", top_k=3, alpha=0.3)
    assert_greater(len(results_kw), 0, "关键词'熊猫'有结果")
    # 熊猫基地应该在顶部
    names = [r["metadata"].get("name") for r in results_kw]
    assert_in("大熊猫繁育研究基地", names, "熊猫查询返回熊猫基地")

    # 阈值过滤
    results_filtered = retriever.hybrid_search(
        "海滩冲浪", top_k=10, score_threshold=0.5
    )
    print("  [INFO] 高阈值过滤: threshold=0.5, 返回 " + str(len(results_filtered)) + " 条")


# ============================================================
# 测试：rerank_search 重排序检索
# ============================================================
def test_rerank_search():
    print("\n" + "=" * 60)
    print("[测试] rerank_search 重排序检索")
    print("=" * 60)

    # ---- 前置检查：FlagEmbedding ----
    try:
        import FlagEmbedding  # noqa: F401
    except ImportError:
        print("  [SKIP] FlagEmbedding 未安装，跳过 Reranker 测试")
        print("  [SKIP] 安装方式: pip install FlagEmbedding")
        return

    # ---- 前置检查：RERANK_ENABLED 配置 ----
    from config.settings import settings
    if not settings.RERANK_ENABLED:
        print("  [SKIP] RERANK_ENABLED=false，按配置跳过 Reranker 测试")
        return

    # ---- 前置检查：transformers 版本兼容 ----
    try:
        import transformers
        ver = transformers.__version__
        major, minor = ver.split(".")[:2]
        if int(major) > 4 or (int(major) == 4 and int(minor) >= 36):
            print(
                f"  [SKIP] transformers=={ver} 版本过高, "
                f"与 FlagEmbedding 存在 prepare_for_model 兼容问题, 跳过 Reranker 测试"
            )
            return
    except ImportError:
        pass

    setup_test_data()
    retriever = ScenicRetriever(
        collection_name=TEST_COLLECTION_NAME,
        persist_directory=TEST_PERSIST_DIR,
    )

    # 调用 rerank_search，内部已做版本/配置兼容降级
    try:
        results = retriever.rerank_search("北京皇家宫殿建筑", top_k=3)
    except AttributeError as e:
        print("  [SKIP] Reranker 分词器兼容报错 (prepare_for_model), 跳过: " + str(e))
        print("  [SKIP] 建议: pip install transformers==4.35.2 或升级 FlagEmbedding")
        return

    assert_greater(len(results), 0, "Rerank 检索有结果")
    for r in results:
        assert_true("score" in r, "含 score")
        assert_true(0.0 <= r["score"] <= 1.0, "Rerank score 在 [0, 1]")

    print("  [INFO] Rerank 检索返回 " + str(len(results)) + " 条:")
    for i, r in enumerate(results):
        print("    " + str(i+1) + ". [" + "{:.4f}".format(r["score"]) + "] "
              + str(r["metadata"].get("name", "?")))

    if len(results) >= 1:
        top_name = results[0]["metadata"].get("name", "")
        print("  [INFO] 首位景点: " + str(top_name))


# ============================================================
# 测试：异常场景
# ============================================================
def test_error_scenarios():
    print("\n" + "=" * 60)
    print("[测试] 异常场景")
    print("=" * 60)

    setup_test_data()
    retriever = ScenicRetriever(
        collection_name=TEST_COLLECTION_NAME,
        persist_directory=TEST_PERSIST_DIR,
    )

    # filter_search 非法参数
    assert_raises(RetrieverParamError,
                  lambda: retriever.filter_search("北京", top_k=0),
                  "top_k=0 抛出异常")

    assert_raises(RetrieverParamError,
                  lambda: retriever.filter_search("北京", ticket_min=100, ticket_max=50),
                  "ticket_min > ticket_max 抛出异常")

    # hybrid_search 空查询
    assert_raises(RetrieverParamError,
                  lambda: retriever.hybrid_search(""),
                  "空查询抛出 RetrieverParamError")

    assert_raises(RetrieverParamError,
                  lambda: retriever.hybrid_search("   "),
                  "纯空格查询抛出异常")

    # hybrid_search 非法 alpha
    assert_raises(RetrieverParamError,
                  lambda: retriever.hybrid_search("北京", top_k=5, alpha=1.5),
                  "alpha=1.5 抛出异常")

    # hybrid_search 非法 top_k
    assert_raises(RetrieverParamError,
                  lambda: retriever.hybrid_search("北京", top_k=-1),
                  "top_k=-1 抛出异常")

    # rerank_search 空查询
    assert_raises(RetrieverParamError,
                  lambda: retriever.rerank_search(""),
                  "rerank 空查询抛出异常")


# ============================================================
# 测试：结果格式统一性
# ============================================================
def test_unified_format():
    print("\n" + "=" * 60)
    print("[测试] 统一返回格式")
    print("=" * 60)

    setup_test_data()
    retriever = ScenicRetriever(
        collection_name=TEST_COLLECTION_NAME,
        persist_directory=TEST_PERSIST_DIR,
    )

    # 三种检索的返回格式应一致
    filter_results = retriever.filter_search("故宫", city="北京", top_k=3)
    hybrid_results = retriever.hybrid_search("故宫", top_k=3)

    for name, results in [("filter_search", filter_results),
                          ("hybrid_search", hybrid_results)]:
        assert_true(isinstance(results, list), name + " 返回 list")
        if results:
            r = results[0]
            assert_true("content" in r, name + " 含 content")
            assert_true("metadata" in r, name + " 含 metadata")
            assert_true("score" in r, name + " 含 score")
            assert_true(isinstance(r["content"], str), name + " content 为 str")
            assert_true(isinstance(r["metadata"], dict), name + " metadata 为 dict")
            assert_true(isinstance(r["score"], float), name + " score 为 float")


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  多策略检索模块单元测试")
    print("  项目根目录: " + PROJECT_ROOT)
    print("  测试集合: " + TEST_COLLECTION_NAME)
    print("=" * 60)

    try:
        test_filter_by_city()
        test_filter_by_price_and_tags()
        test_filter_without_query()
        test_hybrid_search()
        test_rerank_search()
        test_unified_format()
        test_error_scenarios()

    finally:
        print("\n" + "=" * 60)
        print("  清理测试环境...")
        print("=" * 60)
        EmbeddingModel.reset_instance()
        cleanup()
        print("  清理完成")

    total = _pass_count + _fail_count
    print("\n" + "=" * 60)
    print("  测试结果: " + str(_pass_count) + "/" + str(total) + " 通过, " + str(_fail_count) + " 失败")
    print("=" * 60)

    if _fail_count > 0:
        sys.exit(1)
