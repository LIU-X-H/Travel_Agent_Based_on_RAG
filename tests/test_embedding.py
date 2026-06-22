"""
Embedding 模块单元测试
======================
测试覆盖：
1. 单例模式验证 —— 多次实例化返回同一对象
2. embed_query 单条文本向量化
3. embed_documents 批量文本向量化
4. 向量维度与配置一致性
5. 异常场景：空字符串、空列表、错误类型
6. 模型延迟加载验证

运行方式：
    cd travel_scenic_rag
    python tests/test_embedding.py

或使用 pytest：
    pytest tests/test_embedding.py -v
"""

import sys
import os

# 将项目根目录加入 sys.path，确保能导入项目模块
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.embedding import (
    EmbeddingModel,
    EmbeddingInputError,
    EmbeddingModelLoadError,
    EmbeddingInferenceError,
)


# ============================================================
# 测试辅助函数
# ============================================================
def assert_equal(actual, expected, test_name: str) -> None:
    """简单的断言辅助：比较实际值与预期值。"""
    if actual == expected:
        print(f"  [PASS] {test_name}")
    else:
        print(f"  [FAIL] {test_name}: 期望 {expected!r}, 实际 {actual!r}")


def assert_true(condition: bool, test_name: str) -> None:
    """断言条件为 True。"""
    if condition:
        print(f"  [PASS] {test_name}")
    else:
        print(f"  [FAIL] {test_name}: 条件不满足")


def assert_raises(exception_class, callable_fn, test_name: str) -> None:
    """断言无参可调用对象抛出指定类型的异常。"""
    try:
        callable_fn()
        print(f"  [FAIL] {test_name}: 未抛出异常")
    except exception_class:
        print(f"  [PASS] {test_name}")
    except Exception as e:
        print(
            f"  [FAIL] {test_name}: 抛出了错误的异常类型 "
            f"{type(e).__name__}: {e}"
        )


# ============================================================
# 测试用例
# ============================================================
def test_singleton() -> None:
    """测试单例模式：多次实例化返回同一对象。"""
    print("\n[测试] 单例模式")

    # 重置单例（确保测试独立性）
    EmbeddingModel.reset_instance()

    model_a = EmbeddingModel()
    model_b = EmbeddingModel()
    model_c = EmbeddingModel()

    assert_true(model_a is model_b, "同一对象 (a is b)")
    assert_true(model_b is model_c, "同一对象 (b is c)")
    assert_equal(id(model_a), id(model_b), "内存地址相同 (a == b)")


def test_embed_query() -> None:
    """测试单条文本向量化。"""
    print("\n[测试] embed_query 单条向量化")

    EmbeddingModel.reset_instance()
    model = EmbeddingModel()

    # 测试基本向量化
    query = "北京故宫是中国最著名的旅游景点之一"
    vec = model.embed_query(query)

    assert_true(isinstance(vec, list), "返回类型为 list")
    assert_true(len(vec) > 0, "向量长度 > 0")
    assert_true(all(isinstance(v, float) for v in vec), "所有元素为 float")
    assert_equal(len(vec), model.dim, f"向量维度 = {model.dim}")


def test_embed_documents() -> None:
    """测试批量文本向量化。"""
    print("\n[测试] embed_documents 批量向量化")

    EmbeddingModel.reset_instance()
    model = EmbeddingModel()

    texts = [
        "故宫是世界上规模最大的宫殿建筑群",
        "长城是中国古代伟大的防御工程",
        "西湖是杭州最著名的自然景观",
    ]
    vecs = model.embed_documents(texts)

    assert_true(isinstance(vecs, list), "返回类型为 list")
    assert_equal(len(vecs), len(texts), f"返回文档数 = {len(texts)}")

    for i, (vec, text) in enumerate(zip(vecs, texts)):
        assert_true(isinstance(vec, list), f"第 {i} 条: 类型为 list")
        assert_equal(len(vec), model.dim, f"第 {i} 条: 向量维度 = {model.dim}")
        assert_true(
            all(isinstance(v, float) for v in vec),
            f"第 {i} 条: 所有元素为 float",
        )

    # 验证不同文本产生不同向量
    assert_true(vecs[0] != vecs[1], "不同文本产生不同向量")


def test_lazy_loading() -> None:
    """测试延迟加载：__init__ 不加载模型，首次调用时才加载。"""
    print("\n[测试] 延迟加载")

    EmbeddingModel.reset_instance()
    model = EmbeddingModel()

    # 初始化后模型应未加载
    assert_true(not model.is_loaded, "初始化后 is_loaded = False")

    # 首次调用 embed_query 触发加载
    _ = model.embed_query("测试文本")
    assert_true(model.is_loaded, "首次调用后 is_loaded = True")


def test_embed_query_empty_text() -> None:
    """测试空字符串输入应抛出 EmbeddingInputError。"""
    print("\n[测试] 异常场景: 空字符串")

    EmbeddingModel.reset_instance()
    model = EmbeddingModel()

    # 先确保模型加载
    _ = model.embed_query("预加载模型文本")
    assert_true(model.is_loaded, "模型已加载")

    # 空字符串
    assert_raises(
        EmbeddingInputError,
        lambda: model.embed_query(""),
        "空字符串抛出 EmbeddingInputError",
    )

    # 纯空格字符串
    assert_raises(
        EmbeddingInputError,
        lambda: model.embed_query("   "),
        "纯空格字符串抛出 EmbeddingInputError",
    )


def test_embed_query_wrong_type() -> None:
    """测试错误类型输入应抛出 EmbeddingInputError。"""
    print("\n[测试] 异常场景: 错误类型")

    EmbeddingModel.reset_instance()
    model = EmbeddingModel()
    _ = model.embed_query("预加载模型文本")

    # 传入 int
    assert_raises(
        EmbeddingInputError,
        lambda: model.embed_query(123),
        "传入 int 抛出 EmbeddingInputError",
    )

    # 传入 None
    assert_raises(
        EmbeddingInputError,
        lambda: model.embed_query(None),
        "传入 None 抛出 EmbeddingInputError",
    )


def test_embed_documents_empty_list() -> None:
    """测试空列表输入应抛出 EmbeddingInputError。"""
    print("\n[测试] 异常场景: 空列表")

    EmbeddingModel.reset_instance()
    model = EmbeddingModel()
    _ = model.embed_query("预加载模型文本")

    assert_raises(
        EmbeddingInputError,
        lambda: model.embed_documents([]),
        "空列表抛出 EmbeddingInputError",
    )


def test_embed_documents_mixed_types() -> None:
    """测试列表中含非字符串元素应抛出 EmbeddingInputError。"""
    print("\n[测试] 异常场景: 列表中含非字符串元素")

    EmbeddingModel.reset_instance()
    model = EmbeddingModel()
    _ = model.embed_query("预加载模型文本")

    # 列表中混入非字符串
    assert_raises(
        EmbeddingInputError,
        lambda: model.embed_documents(["正常文本", 123, "另一条文本"]),
        "含 int 的列表抛出 EmbeddingInputError",
    )

    # 列表中含空字符串
    assert_raises(
        EmbeddingInputError,
        lambda: model.embed_documents(["正常文本", "", "另一条文本"]),
        "含空字符串的列表抛出 EmbeddingInputError",
    )


def test_embed_documents_wrong_type() -> None:
    """测试传入非列表类型应抛出 EmbeddingInputError。"""
    print("\n[测试] 异常场景: embed_documents 传入非列表")

    EmbeddingModel.reset_instance()
    model = EmbeddingModel()
    _ = model.embed_query("预加载模型文本")

    assert_raises(
        EmbeddingInputError,
        lambda: model.embed_documents("不是列表的字符串"),
        "传入 str 抛出 EmbeddingInputError",
    )


def test_vector_consistency() -> None:
    """测试相同文本产生的向量一致（确定性推理）。"""
    print("\n[测试] 向量一致性")

    EmbeddingModel.reset_instance()
    model = EmbeddingModel()

    text = "北京故宫"
    vec1 = model.embed_query(text)
    vec2 = model.embed_query(text)

    # 相同输入应产生完全相同的向量
    assert_equal(vec1, vec2, "相同文本产生相同向量")

    # embed_query 与 embed_documents 对同一文本应产生相同向量
    vec3 = model.embed_documents([text])[0]
    assert_equal(vec1, vec3, "embed_query 与 embed_documents 结果一致")


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Embedding 模块单元测试")
    print(f"项目根目录: {PROJECT_ROOT}")
    print("=" * 60)

    # 按顺序执行测试
    test_singleton()
    test_lazy_loading()
    test_embed_query()
    test_embed_documents()
    test_vector_consistency()
    test_embed_query_empty_text()
    test_embed_query_wrong_type()
    test_embed_documents_empty_list()
    test_embed_documents_mixed_types()
    test_embed_documents_wrong_type()

    print("\n" + "=" * 60)
    print("所有测试执行完毕")
    print("=" * 60)
