"""
向量库模块单元测试
==================
覆盖：
1. 向量库初始化（自动创建/加载持久化）
2. add_scenic_docs   —— 批量入库 + 持久化验证
3. base_similarity_search —— 语义检索 + 分数过滤
4. get_retriever     —— LangChain Retriever 兼容性
5. delete_by_city    —— 按城市增量删除
6. 异常场景           —— 空文档、空查询、无效参数

运行方式（二选一）：
    cd travel_scenic_rag
    python tests/test_vector_store.py           # 直接运行
    pytest tests/test_vector_store.py -v -s     # pytest 运行

注意：
    - 测试使用临时向量库目录（data/test_vector_db），不会污染生产数据
    - 需要先安装依赖: pip install langchain-community chromadb sentence-transformers
"""

import sys
import os
import shutil

# 将项目根目录加入 sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from langchain_core.documents import Document

from modules.vector_store import (
    ScenicVectorStore,
    VectorStoreError,
    VectorStoreWriteError,
    VectorStoreQueryError,
    VectorStoreDeleteError,
)
from modules.embedding import EmbeddingModel


# ============================================================
# 测试配置
# ============================================================
# 使用临时目录隔离测试数据，避免污染正式向量库
TEST_PERSIST_DIR = os.path.join(PROJECT_ROOT, "data", "test_vector_db")
TEST_COLLECTION_NAME = "test_travel_scenic_spots"


# ============================================================
# 模拟景点测试数据
# ============================================================
def build_test_documents():
    """
    构建多城市、多类型景点测试 Document 数据。

    返回：
        list[Document] - 覆盖北京(2)、杭州(2)、成都(1) 三个城市
    """
    scenic_data = [
        # ---- 北京 ----
        {
            "content": (
                "故宫博物院位于北京市中心，是明清两代的皇家宫殿，"
                "又称紫禁城。故宫占地面积72万平方米，建筑面积约15万平方米，"
                "有大小宫殿七十多座，房屋九千余间。1987年被列为世界文化遗产，"
                "是国家5A级旅游景区。门票旺季60元，淡季40元。"
            ),
            "metadata": {
                "city": "北京",
                "name": "故宫博物院",
                "ticket": 60,
                "level": "5A",
                "tags": ["世界文化遗产", "古迹", "博物馆", "皇家园林"],
            },
        },
        {
            "content": (
                "八达岭长城位于北京市延庆区，是明长城中保存最完好、"
                "最具代表性的一段。长城全长约21,196公里，八达岭段因地势险要、"
                "城关坚固而闻名，史称天下九塞之一。1987年被列为世界文化遗产，"
                "是国家5A级旅游景区。门票旺季45元，淡季40元。"
            ),
            "metadata": {
                "city": "北京",
                "name": "八达岭长城",
                "ticket": 45,
                "level": "5A",
                "tags": ["世界文化遗产", "古迹", "登山", "历史遗迹"],
            },
        },
        # ---- 杭州 ----
        {
            "content": (
                "西湖位于浙江省杭州市西面，是中国大陆首批国家重点风景名胜区"
                "和中国十大风景名胜之一。西湖三面环山，面积约6.39平方千米，"
                "湖中被孤山、白堤、苏堤、杨公堤分隔。西湖以其湖光山色和"
                "深厚的人文底蕴吸引了历代文人墨客。2011年被列为世界文化景观遗产，"
                "是国家5A级旅游景区，免费开放。"
            ),
            "metadata": {
                "city": "杭州",
                "name": "西湖",
                "ticket": 0,
                "level": "5A",
                "tags": ["世界文化遗产", "自然风光", "湖泊", "免费"],
            },
        },
        {
            "content": (
                "灵隐寺位于浙江省杭州市西湖区，始建于东晋咸和元年（公元326年），"
                "是中国佛教禅宗十大古刹之一。寺内主要建筑有天王殿、大雄宝殿、"
                "药师殿等。灵隐寺飞来峰上雕刻有五代至元代的大量佛教石窟造像，"
                "是全国重点文物保护单位。门票75元（飞来峰景区45元+灵隐寺香花券30元）。"
            ),
            "metadata": {
                "city": "杭州",
                "name": "灵隐寺",
                "ticket": 75,
                "level": "5A",
                "tags": ["佛教", "古迹", "寺庙", "石窟"],
            },
        },
        # ---- 成都 ----
        {
            "content": (
                "成都大熊猫繁育研究基地位于四川省成都市成华区，"
                "是世界上最重要的大熊猫保护研究机构之一。基地占地面积约100公顷，"
                "饲养有大熊猫、小熊猫、黑颈鹤等珍稀动物。游客可以近距离观察"
                "大熊猫的日常生活，是亲子游和自然爱好者的热门目的地。"
                "门票55元，是国家4A级旅游景区。"
            ),
            "metadata": {
                "city": "成都",
                "name": "大熊猫繁育研究基地",
                "ticket": 55,
                "level": "4A",
                "tags": ["动物园", "亲子", "自然", "熊猫"],
            },
        },
    ]

    documents = [
        Document(page_content=item["content"], metadata=item["metadata"])
        for item in scenic_data
    ]
    return documents


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


def assert_raises(exception_class, callable_fn, test_name):
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


def cleanup_test_store():
    """清理测试向量库目录，兼容 Windows 文件占用。"""
    if not os.path.exists(TEST_PERSIST_DIR):
        return
    import stat
    import time
    max_retries = 3
    for attempt in range(max_retries):
        try:
            for root, dirs, files in os.walk(TEST_PERSIST_DIR):
                for name in dirs + files:
                    full = os.path.join(root, name)
                    try:
                        os.chmod(full, stat.S_IWRITE)
                    except OSError:
                        pass
            shutil.rmtree(TEST_PERSIST_DIR)
            print("[清理] 已删除测试目录: " + TEST_PERSIST_DIR)
            return
        except PermissionError as e:
            if attempt < max_retries - 1:
                print("[清理] 文件占用，1秒后重试... (" + str(e) + ")")
                time.sleep(1)
            else:
                print("[清理] 跳过: 文件被其他进程占用，无法删除 " + TEST_PERSIST_DIR)
                print("[清理] 提示: 可手动删除或重启后自动清理")
        except OSError as e:
            if hasattr(e, 'winerror') and e.winerror == 32:
                print("[清理] 跳过: Windows 文件锁定 [WinError 32]，请手动删除 " + TEST_PERSIST_DIR)
            else:
                print("[清理] 失败: " + str(e))
            break
        except Exception as e:
            print("[清理] 失败: " + str(e))
            break


# ============================================================
# 测试用例
# ============================================================
def test_init_and_persistence():
    print("\n" + "=" * 60)
    print("[测试] 向量库初始化与持久化")
    print("=" * 60)

    cleanup_test_store()

    store = ScenicVectorStore(
        collection_name=TEST_COLLECTION_NAME,
        persist_directory=TEST_PERSIST_DIR,
    )
    assert_true(store.store is not None, "底层 Chroma 实例不为 None")
    assert_equal(store.collection_name, TEST_COLLECTION_NAME, "集合名称正确")
    assert_equal(store.count(), 0, "新集合文档数为 0")

    docs = build_test_documents()
    store.add_scenic_docs(docs)
    doc_count_after_add = store.count()
    assert_equal(doc_count_after_add, len(docs), "入库后文档数 = " + str(len(docs)))

    store2 = ScenicVectorStore(
        collection_name=TEST_COLLECTION_NAME,
        persist_directory=TEST_PERSIST_DIR,
    )
    assert_equal(
        store2.count(), len(docs),
        "重新加载后文档数仍为 " + str(len(docs)) + "（持久化验证）"
    )
    print("  [INFO] 向量库路径: " + TEST_PERSIST_DIR)
    print("  [INFO] 集合名称: " + TEST_COLLECTION_NAME)


def test_add_scenic_docs():
    print("\n" + "=" * 60)
    print("[测试] add_scenic_docs 批量入库")
    print("=" * 60)

    cleanup_test_store()
    store = ScenicVectorStore(
        collection_name=TEST_COLLECTION_NAME,
        persist_directory=TEST_PERSIST_DIR,
    )

    docs = build_test_documents()
    count = store.add_scenic_docs(docs)

    assert_equal(count, 5, "入库返回 5 条")
    assert_equal(store.count(), 5, "向量库文档总数为 5")

    count2 = store.add_scenic_docs(docs[:1])
    assert_equal(count2, 1, "重复添加 1 条成功")
    print("  [INFO] 补充入库后文档总数: " + str(store.count()))

    cities = store.get_cities()
    assert_true("北京" in cities, "城市列表包含北京")
    assert_true("杭州" in cities, "城市列表包含杭州")
    assert_true("成都" in cities, "城市列表包含成都")
    print("  [INFO] 城市列表: " + str(cities))


def test_base_similarity_search():
    print("\n" + "=" * 60)
    print("[测试] base_similarity_search 语义检索")
    print("=" * 60)

    cleanup_test_store()
    store = ScenicVectorStore(
        collection_name=TEST_COLLECTION_NAME,
        persist_directory=TEST_PERSIST_DIR,
    )
    docs = build_test_documents()
    store.add_scenic_docs(docs)

    results = store.base_similarity_search("北京有哪些著名的古迹景点")
    assert_true(isinstance(results, list), "返回 list")
    assert_greater(len(results), 0, "至少返回 1 条结果")

    for r in results:
        assert_true("content" in r, "结果含 content 字段")
        assert_true("metadata" in r, "结果含 metadata 字段")
        assert_true("score" in r, "结果含 score 字段")
        assert_true(
            0.0 <= r["score"] <= 1.0,
            "score 在 [0, 1] 范围内"
        )

    print("  [INFO] 检索词='北京古迹'，返回 " + str(len(results)) + " 条:")
    for i, r in enumerate(results):
        print("    " + str(i+1) + ". [" + "{:.4f}".format(r["score"]) + "] "
              + str(r["metadata"].get("name", "?")))

    results_3 = store.base_similarity_search("杭州有什么好玩的", top_k=2)
    assert_equal(len(results_3), 2, "自定义 top_k=2 返回 2 条")

    results_filtered = store.base_similarity_search(
        "大熊猫", top_k=5, score_threshold=0.3
    )
    print("  [INFO] 阈值过滤: top_k=5, threshold=0.3, 实际返回 " + str(len(results_filtered)) + " 条")
    for r in results_filtered:
        assert_true(r["score"] >= 0.3, "过滤后 score >= 0.3")

    results_strict = store.base_similarity_search(
        "海滩冲浪潜水", top_k=10, score_threshold=0.9
    )
    print("  [INFO] 高阈值检索('海滩冲浪'): threshold=0.9, 返回 " + str(len(results_strict)) + " 条")


def test_get_retriever():
    print("\n" + "=" * 60)
    print("[测试] get_retriever LangChain 兼容性")
    print("=" * 60)

    cleanup_test_store()
    store = ScenicVectorStore(
        collection_name=TEST_COLLECTION_NAME,
        persist_directory=TEST_PERSIST_DIR,
    )
    docs = build_test_documents()
    store.add_scenic_docs(docs)

    retriever = store.get_retriever(top_k=3)
    assert_true(retriever is not None, "retriever 不为 None")

    retriever_docs = retriever.invoke("成都看熊猫")
    assert_true(isinstance(retriever_docs, list), "invoke 返回 list")
    assert_equal(len(retriever_docs), 3, "top_k=3 返回 3 条 Document")

    for doc in retriever_docs:
        assert_true(isinstance(doc, Document), "invoke 返回元素为 Document 类型")
        assert_true(len(doc.page_content) > 0, "Document page_content 非空")

    print("  [INFO] retriever.invoke('成都看熊猫') 返回 " + str(len(retriever_docs)) + " 条:")
    for i, doc in enumerate(retriever_docs):
        name = doc.metadata.get("name", "?")
        city = doc.metadata.get("city", "?")
        print("    " + str(i+1) + ". [" + str(city) + "] " + str(name))


def test_delete_by_city():
    print("\n" + "=" * 60)
    print("[测试] delete_by_city 按城市删除")
    print("=" * 60)

    cleanup_test_store()
    store = ScenicVectorStore(
        collection_name=TEST_COLLECTION_NAME,
        persist_directory=TEST_PERSIST_DIR,
    )
    docs = build_test_documents()
    store.add_scenic_docs(docs)

    total_before = store.count()
    print("  [INFO] 删除前文档总数: " + str(total_before))

    deleted = store.delete_by_city("北京")
    assert_equal(deleted, 2, "删除北京数据 2 条")

    total_after_beijing = store.count()
    assert_equal(total_after_beijing, total_before - 2, "删除北京后总数 = " + str(total_before - 2))

    city_results = store.base_similarity_search("故宫长城")
    for r in city_results:
        assert_true(
            r["metadata"].get("city") != "北京",
            "检索结果中不含北京景点"
        )

    deleted_zero = store.delete_by_city("火星")
    assert_equal(deleted_zero, 0, "删除不存在的城市返回 0")

    deleted_hz = store.delete_by_city("杭州")
    assert_equal(deleted_hz, 2, "删除杭州数据 2 条")

    deleted_cd = store.delete_by_city("成都")
    assert_equal(deleted_cd, 1, "删除成都数据 1 条")

    assert_equal(store.count(), 0, "所有城市删除后文档数为 0")


def test_error_scenarios():
    print("\n" + "=" * 60)
    print("[测试] 异常场景")
    print("=" * 60)

    cleanup_test_store()
    store = ScenicVectorStore(
        collection_name=TEST_COLLECTION_NAME,
        persist_directory=TEST_PERSIST_DIR,
    )

    assert_raises(VectorStoreWriteError, lambda: store.add_scenic_docs([]), "空列表抛出 VectorStoreWriteError")

    bad_doc = Document(page_content="", metadata={"city": "北京"})
    assert_raises(VectorStoreWriteError, lambda: store.add_scenic_docs([bad_doc]), "空 page_content 抛出 VectorStoreWriteError")

    whitespace_doc = Document(page_content="   ", metadata={"city": "北京"})
    assert_raises(VectorStoreWriteError, lambda: store.add_scenic_docs([whitespace_doc]), "纯空格 page_content 抛出 VectorStoreWriteError")

    assert_raises(VectorStoreWriteError, lambda: store.add_scenic_docs(["非Document"]), "非 Document 元素抛出 VectorStoreWriteError")

    assert_raises(VectorStoreQueryError, lambda: store.base_similarity_search(""), "空查询文本抛出 VectorStoreQueryError")

    assert_raises(VectorStoreQueryError, lambda: store.base_similarity_search("   "), "纯空格查询文本抛出 VectorStoreQueryError")

    assert_raises(VectorStoreQueryError, lambda: store.base_similarity_search("北京", top_k=0), "top_k=0 抛出 VectorStoreQueryError")

    assert_raises(VectorStoreQueryError, lambda: store.base_similarity_search("北京", top_k=5, score_threshold=1.5), "threshold=1.5 抛出 VectorStoreQueryError")

    assert_raises(VectorStoreError, lambda: store.delete_by_city(""), "delete_by_city('') 抛出 VectorStoreError")


def test_metadata_preservation():
    print("\n" + "=" * 60)
    print("[测试] 元数据完整性")
    print("=" * 60)

    cleanup_test_store()
    store = ScenicVectorStore(
        collection_name=TEST_COLLECTION_NAME,
        persist_directory=TEST_PERSIST_DIR,
    )
    docs = build_test_documents()
    store.add_scenic_docs(docs)

    results = store.base_similarity_search("成都熊猫基地", top_k=1)
    assert_equal(len(results), 1, "检索返回 1 条")

    result = results[0]
    meta = result["metadata"]

    assert_equal(meta.get("city"), "成都", "metadata.city 正确")
    assert_equal(meta.get("name"), "大熊猫繁育研究基地", "metadata.name 正确")
    assert_equal(meta.get("ticket"), 55, "metadata.ticket 正确")
    assert_equal(meta.get("level"), "4A", "metadata.level 正确")
    assert_true("动物园" in meta.get("tags", []), "tags 包含'动物园'")

    print("  [INFO] 完整元数据: " + str(meta))


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  向量库模块单元测试")
    print("  项目根目录: " + PROJECT_ROOT)
    print("  测试集合: " + TEST_COLLECTION_NAME)
    print("  测试路径: " + TEST_PERSIST_DIR)
    print("=" * 60)

    try:
        test_init_and_persistence()
        test_add_scenic_docs()
        test_base_similarity_search()
        test_get_retriever()
        test_metadata_preservation()
        test_delete_by_city()
        test_error_scenarios()

    finally:
        print("\n" + "=" * 60)
        print("  清理测试环境...")
        print("=" * 60)
        EmbeddingModel.reset_instance()
        cleanup_test_store()
        print("  清理完成")

    total = _pass_count + _fail_count
    print("\n" + "=" * 60)
    print("  测试结果: " + str(_pass_count) + "/" + str(total) + " 通过, " + str(_fail_count) + " 失败")
    print("=" * 60)

    if _fail_count > 0:
        sys.exit(1)
