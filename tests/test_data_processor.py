"""
数据处理器单元测试
==================
覆盖：
1. load_raw_json        —— 加载 JSON / 自动选择文件 / 空目录报错
2. clean_raw_data       —— 票价标准化 / 标签枚举 / city清洗 / 空记录跳过
3. convert_to_document   —— 拼接 page_content / 元数据完整性
4. batch_import_to_vector —— 全量入库 + 幂等验证
5. process_and_import    —— 一键处理全流程
6. 异常场景              —— 文件缺失 / JSON 解析失败 / 空数据

运行方式：
    cd travel_scenic_rag
    python tests/test_data_processor.py
    pytest tests/test_data_processor.py -v -s

注意：
    测试使用独立目录 data/test_processor_raw/ 和临时向量库
"""

import json
import os
import shutil
import sys
import stat
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from langchain_core.documents import Document

from modules.data_processor import (
    ScenicDataProcessor,
    DataLoadError,
    DataCleanError,
    DataImportError,
)
from modules.vector_store import ScenicVectorStore
from modules.embedding import EmbeddingModel


# ============================================================
# 测试配置
# ============================================================
TEST_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "test_processor_raw")
TEST_JSON = os.path.join(TEST_RAW_DIR, "test_scenic.json")
TEST_VECTOR_DIR = os.path.join(PROJECT_ROOT, "data", "test_processor_db")
TEST_COLLECTION = "test_processor_spots"


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


def cleanup_vector():
    if os.path.exists(TEST_VECTOR_DIR):
        for attempt in range(3):
            try:
                for root, dirs, files in os.walk(TEST_VECTOR_DIR):
                    for name in dirs + files:
                        try:
                            os.chmod(os.path.join(root, name), stat.S_IWRITE)
                        except OSError:
                            pass
                shutil.rmtree(TEST_VECTOR_DIR)
                print("[清理] 已删除: " + TEST_VECTOR_DIR)
                return
            except PermissionError as e:
                if attempt < 2:
                    print("[清理] 文件占用，1秒后重试... (" + str(e) + ")")
                    time.sleep(1)
                else:
                    print("[清理] 跳过: 文件被占用 " + TEST_VECTOR_DIR)
            except OSError as e:
                if hasattr(e, 'winerror') and e.winerror == 32:
                    print("[清理] 跳过: WinError 32, 请手动删除 " + TEST_VECTOR_DIR)
                else:
                    print("[清理] 失败: " + str(e))
                break
            except Exception as e:
                print("[清理] 失败: " + str(e))
                break


# ============================================================
# 测试：加载原始 JSON
# ============================================================
def test_load_raw_json():
    print("\n" + "=" * 60)
    print("[测试] load_raw_json")
    print("=" * 60)

    # 指定文件名加载
    processor = ScenicDataProcessor(raw_data_dir=TEST_RAW_DIR)
    data = processor.load_raw_json("test_scenic.json")
    assert_true(isinstance(data, list), "返回 list")
    assert_equal(len(data), 6, "test_scenic.json 含 6 条记录")
    for i, item in enumerate(data):
        assert_true(isinstance(item, dict), "第" + str(i) + "条为 dict")
    print("  [INFO] 已加载 6 条原始记录")

    # 自动选择文件
    processor2 = ScenicDataProcessor(raw_data_dir=TEST_RAW_DIR)
    data2 = processor2.load_raw_json()
    assert_equal(len(data2), 6, "自动选择加载 6 条")


def test_load_raw_json_errors():
    print("\n[测试] load_raw_json 异常")

    processor = ScenicDataProcessor(raw_data_dir=TEST_RAW_DIR)

    # 文件不存在
    assert_raises(
        DataLoadError,
        lambda: processor.load_raw_json("nonexistent.json"),
        "不存在的文件抛出 DataLoadError",
    )

    # 空目录
    empty_dir = os.path.join(TEST_RAW_DIR, "_empty")
    os.makedirs(empty_dir, exist_ok=True)
    processor_empty = ScenicDataProcessor(raw_data_dir=empty_dir)
    assert_raises(
        DataLoadError,
        lambda: processor_empty.load_raw_json(),
        "空目录抛出 DataLoadError",
    )
    os.rmdir(empty_dir)


# ============================================================
# 测试：数据清洗
# ============================================================
def test_clean_raw_data():
    print("\n" + "=" * 60)
    print("[测试] clean_raw_data")
    print("=" * 60)

    processor = ScenicDataProcessor(raw_data_dir=TEST_RAW_DIR)
    raw = processor.load_raw_json("test_scenic.json")
    cleaned = processor.clean_raw_data(raw)

    # 原始 6 条：第5条 name="" 应被跳过 → 清洗后 5 条
    assert_equal(len(cleaned), 5, "清洗后 5 条（跳过1条空name）")

    for i, r in enumerate(cleaned):
        print("  [INFO] 第" + str(i+1) + "条: name=" + repr(r["name"])
              + " city=" + repr(r["city"]) + " ticket=" + str(r["ticket"])
              + " tags=" + str(r["tags"]))

    # 票价标准化
    assert_equal(cleaned[0]["ticket"], 60.0, "故宫 '60元' → 60.0")
    assert_equal(cleaned[2]["ticket"], 0.0, "西湖 '免费' → 0.0")
    assert_equal(cleaned[3]["ticket"], 55.0, "熊猫基地 int 55 → 55.0")
    assert_equal(cleaned[4]["ticket"], 80.0, "都江堰 '80元' → 80.0")

    # 城市清洗
    assert_equal(cleaned[1]["city"], "北京", "八达岭 '北京市' → '北京'")
    assert_equal(cleaned[4]["city"], "成都", "都江堰 '成都市' → '成都'")

    # 标签枚举
    assert_true("世界文化遗产" in cleaned[0]["tags"], "故宫 tags 含世界文化遗产")
    assert_true("免费" in cleaned[2]["tags"], "西湖 tags 含免费")

    # name trim
    assert_equal(cleaned[1]["name"], "八达岭长城", "name 去首尾空格")


def test_clean_raw_data_errors():
    print("\n[测试] clean_raw_data 异常")

    processor = ScenicDataProcessor(raw_data_dir=TEST_RAW_DIR)

    # 空列表
    assert_raises(
        DataCleanError,
        lambda: processor.clean_raw_data([]),
        "空列表抛出 DataCleanError",
    )

    # 非 list
    assert_raises(
        DataCleanError,
        lambda: processor.clean_raw_data("not a list"),
        "非 list 抛出 DataCleanError",
    )


# ============================================================
# 测试：convert_to_document
# ============================================================
def test_convert_to_document():
    print("\n" + "=" * 60)
    print("[测试] convert_to_document")
    print("=" * 60)

    processor = ScenicDataProcessor(raw_data_dir=TEST_RAW_DIR)
    raw = processor.load_raw_json("test_scenic.json")
    cleaned = processor.clean_raw_data(raw)

    for i, record in enumerate(cleaned):
        doc = processor.convert_to_document(record)
        assert_true(isinstance(doc, Document), "第" + str(i+1) + "条: 返回 Document")
        assert_true(len(doc.page_content) > 0, "第" + str(i+1) + "条: page_content 非空")
        assert_true("name" in doc.metadata, "第" + str(i+1) + "条: metadata 含 name")
        assert_true("city" in doc.metadata, "第" + str(i+1) + "条: metadata 含 city")
        assert_true("ticket" in doc.metadata, "第" + str(i+1) + "条: metadata 含 ticket")
        assert_true("tags" in doc.metadata, "第" + str(i+1) + "条: metadata 含 tags")

        print("  [INFO] " + str(i+1) + ". " + record["name"] + " → page_content 前80字:")
        print("    " + doc.page_content[:80] + "...")
        print("    metadata: city=" + repr(doc.metadata["city"])
              + " ticket=" + str(doc.metadata["ticket"])
              + " tags=" + str(doc.metadata["tags"]))


def test_convert_errors():
    print("\n[测试] convert_to_document 异常")

    processor = ScenicDataProcessor(raw_data_dir=TEST_RAW_DIR)

    # 缺少 name
    assert_raises(
        DataCleanError,
        lambda: processor.convert_to_document({"name": "", "description": "有描述"}),
        "空 name 抛出 DataCleanError",
    )

    # 缺少 description
    assert_raises(
        DataCleanError,
        lambda: processor.convert_to_document({"name": "有名称", "description": ""}),
        "空 description 抛出 DataCleanError",
    )

    # 非 dict
    assert_raises(
        DataCleanError,
        lambda: processor.convert_to_document("不是dict"),
        "非 dict 抛出 DataCleanError",
    )


# ============================================================
# 测试：batch_import_to_vector
# ============================================================
def test_batch_import():
    print("\n" + "=" * 60)
    print("[测试] batch_import_to_vector")
    print("=" * 60)

    cleanup_vector()

    processor = ScenicDataProcessor(raw_data_dir=TEST_RAW_DIR)
    raw = processor.load_raw_json("test_scenic.json")
    cleaned = processor.clean_raw_data(raw)
    docs = [processor.convert_to_document(r) for r in cleaned]

    vector_store = ScenicVectorStore(
        collection_name=TEST_COLLECTION,
        persist_directory=TEST_VECTOR_DIR,
    )

    # 批量入库
    count = processor.batch_import_to_vector(docs, vector_store=vector_store, batch_size=2)
    assert_equal(count, 5, "入库 5 条")
    assert_equal(vector_store.count(), 5, "向量库文档总数 = 5")

    # 幂等入库（重复添加）
    count2 = processor.batch_import_to_vector(docs[:2], vector_store=vector_store)
    assert_equal(count2, 2, "幂等补充 2 条")


def test_batch_import_errors():
    print("\n[测试] batch_import_to_vector 异常")

    processor = ScenicDataProcessor(raw_data_dir=TEST_RAW_DIR)

    # 空列表
    assert_raises(
        DataImportError,
        lambda: processor.batch_import_to_vector([]),
        "空列表抛出 DataImportError",
    )


# ============================================================
# 测试：一键处理
# ============================================================
def test_process_and_import():
    print("\n" + "=" * 60)
    print("[测试] process_and_import 一键处理")
    print("=" * 60)

    cleanup_vector()

    processor = ScenicDataProcessor(raw_data_dir=TEST_RAW_DIR)
    vector_store = ScenicVectorStore(
        collection_name=TEST_COLLECTION,
        persist_directory=TEST_VECTOR_DIR,
    )

    count = processor.process_and_import("test_scenic.json", vector_store=vector_store)
    assert_equal(count, 5, "一键处理入库 5 条")

    # 验证数据可检索
    results = vector_store.base_similarity_search("故宫", top_k=3)
    assert_greater(len(results), 0, "入库后检索有结果")
    print("  [INFO] 检索'故宫'返回: " + str([r["metadata"].get("name") for r in results]))


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  数据处理器单元测试")
    print("  项目根目录: " + PROJECT_ROOT)
    print("  测试数据: " + TEST_JSON)
    print("=" * 60)

    try:
        test_load_raw_json()
        test_load_raw_json_errors()
        test_clean_raw_data()
        test_clean_raw_data_errors()
        test_convert_to_document()
        test_convert_errors()
        test_batch_import()
        test_batch_import_errors()
        test_process_and_import()

    finally:
        print("\n" + "=" * 60)
        print("  清理测试环境...")
        print("=" * 60)
        EmbeddingModel.reset_instance()
        cleanup_vector()
        print("  清理完成")

    total = _pass_count + _fail_count
    print("\n" + "=" * 60)
    print("  测试结果: " + str(_pass_count) + "/" + str(total) + " 通过, " + str(_fail_count) + " 失败")
    print("=" * 60)

    if _fail_count > 0:
        sys.exit(1)
