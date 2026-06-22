"""
旅游景点数据处理器
==================
负责原始景点数据的加载、清洗、转换与批量入库。

核心职责：
1. load_raw_json      —— 从 data/raw 加载原始 JSON 数据集
2. clean_raw_data      —— 文本清洗、票价标准化、标签统一、空值填充
3. convert_to_document —— 结构化数据 → LangChain Document（拼接 page_content + 绑定元数据）
4. batch_import_to_vector —— 批量转换 + 调用 ScenicVectorStore 幂等入库

所有路径、规则、批次大小从 config.settings 读取，禁止硬编码。

使用示例：
    from modules.data_processor import ScenicDataProcessor

    processor = ScenicDataProcessor()
    raw = processor.load_raw_json()                    # 加载
    records = processor.clean_raw_data(raw)            # 清洗
    docs = [processor.convert_to_document(r) for r in records]  # 转换
    count = processor.batch_import_to_vector(docs)     # 入库
"""

import json
import os
import re
from pathlib import Path
from typing import List, Optional, Dict, Any, Union

from langchain_core.documents import Document

from config.settings import settings
from modules.vector_store import ScenicVectorStore


# ============================================================
# 自定义异常类
# ============================================================
class DataProcessorError(Exception):
    """数据处理器基础异常。"""
    pass


class DataLoadError(DataProcessorError):
    """数据加载失败 —— 文件缺失、JSON 解析错误等。"""
    pass


class DataCleanError(DataProcessorError):
    """数据清洗失败 —— 字段格式异常、必填字段缺失等。"""
    pass


class DataImportError(DataProcessorError):
    """数据入库失败 —— 向量库写入异常等。"""
    pass


# ============================================================
# 合法标签枚举（清洗时自动标准化）
# ============================================================
_VALID_TAGS: set = {
    "世界文化遗产", "自然风光", "古迹", "博物馆", "皇家园林",
    "登山", "历史遗迹", "湖泊", "寺庙", "石窟", "动物园",
    "亲子", "自然", "熊猫", "水利工程", "考古", "免费",
    "佛教", "5A", "4A", "3A",
}


# ============================================================
# 景点数据处理器
# ============================================================
class ScenicDataProcessor:
    """
    旅游景点数据处理器
    ------------------
    封装原始 JSON 加载 → 清洗 → Document 转换 → 向量库入库的完整数据管线。

    属性：
        raw_data_dir : Path  原始数据目录
    """

    def __init__(self, raw_data_dir: Optional[str] = None) -> None:
        """
        初始化数据处理器。

        参数：
            raw_data_dir: 原始数据目录路径，默认从 settings.RAW_DATA_DIR 读取
        """
        self._raw_data_dir: Path = (
            Path(raw_data_dir) if raw_data_dir else settings.RAW_DATA_DIR
        )
        self._batch_size: int = settings.DATA_IMPORT_BATCH_SIZE
        # 确保目录存在
        try:
            self._raw_data_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            print(f"[ScenicDataProcessor] 警告: 无法创建原始数据目录: {e}")

        print(
            f"[ScenicDataProcessor] 处理器就绪: "
            f"raw_dir={self._raw_data_dir}, batch_size={self._batch_size}"
        )

    # ============================================================
    # 1. 加载原始 JSON
    # ============================================================
    def load_raw_json(self, file_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        从 data/raw 目录加载原始景点 JSON 数据集。

        参数：
            file_name: JSON 文件名。若为 None，自动加载目录下第一个 .json 文件。

        返回：
            原始数据记录列表，每项为一个 dict

        异常：
            DataLoadError: 目录为空、文件不存在、JSON 解析失败

        示例：
            processor = ScenicDataProcessor()
            data = processor.load_raw_json("scenic_spots.json")
        """
        # ---- 确定文件路径 ----
        if file_name:
            file_path: Path = self._raw_data_dir / file_name
        else:
            # 自动扫描目录下第一个 .json 文件
            json_files: list = sorted(self._raw_data_dir.glob("*.json"))
            if not json_files:
                raise DataLoadError(
                    f"原始数据目录中没有 JSON 文件: {self._raw_data_dir}。"
                    f"请将景点数据 JSON 文件放入该目录。"
                )
            file_path = json_files[0]
            print(
                f"[ScenicDataProcessor] 自动选择 JSON 文件: {file_path.name}"
            )

        if not file_path.exists():
            raise DataLoadError(
                f"原始数据文件不存在: {file_path}。请确认文件已放入 data/raw/ 目录。"
            )

        # ---- 读取并解析 ----
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise DataLoadError(
                f"JSON 解析失败: {file_path}。请检查 JSON 格式是否合法。"
                f"错误位置: 第 {e.lineno} 行, 第 {e.colno} 列。原始错误: {e}"
            ) from e
        except Exception as e:
            raise DataLoadError(
                f"读取原始数据文件失败: {file_path}。原始错误: {type(e).__name__}: {e}"
            ) from e

        # ---- 校验 ----
        if not isinstance(data, list):
            raise DataLoadError(
                f"JSON 顶层结构应为数组(list)，实际为: {type(data).__name__}。"
                f"请确保 JSON 文件格式为 [{{...}}, {{...}}, ...]"
            )

        if len(data) == 0:
            raise DataLoadError(
                f"原始数据文件 {file_path.name} 中没有任何景点记录（空数组）。"
            )

        print(
            f"[ScenicDataProcessor] 加载成功: {file_path.name}, "
            f"共 {len(data)} 条原始记录"
        )
        return data

    # ============================================================
    # 2. 数据清洗
    # ============================================================
    def clean_raw_data(
        self, raw_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        清洗原始景点数据。

        清洗规则（全部可从 setting 扩展）：
            1. 移除文本字段中的 HTML 标签、多余空白、不可见字符
            2. 票价标准化：字符串 "60元" → float 60.0，空值 → 0.0
            3. tags 标准化：逗号分隔字符串 → list，非法标签自动过滤
            4. city 名称清洗：去除首尾空格、"市"后缀，统一中文
            5. 空值填充：缺失字段用空字符串/0.0 占位
            6. 跳过所有关键字段（name/description）为空的无效记录

        参数：
            raw_data: 原始 JSON 解析后的记录列表

        返回：
            清洗后的记录列表

        异常：
            DataCleanError: 清洗后无有效记录

        示例：
            processor = ScenicDataProcessor()
            raw = processor.load_raw_json()
            clean = processor.clean_raw_data(raw)
        """
        if not isinstance(raw_data, list):
            raise DataCleanError(
                f"clean_raw_data 只接受 list 类型，实际: {type(raw_data).__name__}"
            )

        if len(raw_data) == 0:
            raise DataCleanError("原始数据为空列表，无法清洗。")

        cleaned: List[Dict[str, Any]] = []
        skipped: int = 0

        for idx, item in enumerate(raw_data):
            if not isinstance(item, dict):
                print(
                    f"[ScenicDataProcessor] 跳过第 {idx} 条: 非 dict 类型 "
                    f"({type(item).__name__})"
                )
                skipped += 1
                continue

            try:
                record = self._clean_single_record(item, idx)
                if record is not None:
                    cleaned.append(record)
                else:
                    skipped += 1
            except Exception as e:
                print(
                    f"[ScenicDataProcessor] 跳过第 {idx} 条: 清洗异常 "
                    f"({type(e).__name__}: {e})"
                )
                skipped += 1

        if not cleaned:
            raise DataCleanError(
                f"清洗后无有效记录（原始 {len(raw_data)} 条, 跳过 {skipped} 条）。"
                f"请检查原始数据格式是否符合要求。"
            )

        print(
            f"[ScenicDataProcessor] 清洗完成: "
            f"{len(raw_data)} → {len(cleaned)} 条 (跳过 {skipped} 条)"
        )
        return cleaned

    def _clean_single_record(
        self, item: Dict[str, Any], idx: int
    ) -> Optional[Dict[str, Any]]:
        """
        清洗单条景点记录。

        返回：
            清洗后的 dict，若关键字段缺失返回 None（标记为跳过）
        """
        # ---- 1) 提取字段 ----
        name = self._clean_text(item.get("name", ""))
        city = self._clean_text(item.get("city", ""))
        description = self._clean_text(item.get("description", ""))
        address = self._clean_text(item.get("address", ""))
        open_time = self._clean_text(item.get("open_time", ""))

        # ---- 2) 关键字段缺失检查 ----
        if not name or not description:
            print(
                f"[ScenicDataProcessor] 跳过第 {idx} 条: "
                f"name 或 description 为空 (name='{name[:20] if name else ''}', "
                f"desc='{description[:20] if description else ''}')"
            )
            return None

        # ---- 3) 票价标准化 ----
        ticket: float = self._normalize_ticket(item.get("ticket"))

        # ---- 4) 等级清洗 ----
        level: str = self._clean_text(item.get("level", "")).upper()
        if level and not re.match(r"^\dA$", level):
            level = ""  # 非法等级清空

        # ---- 5) tags 标准化 ----
        tags: List[str] = self._normalize_tags(item.get("tags", ""))

        # ---- 6) city 名称清洗 ----
        city = self._normalize_city(city)

        return {
            "name": name,
            "city": city,
            "description": description,
            "ticket": ticket,
            "level": level,
            "tags": tags,
            "address": address,
            "open_time": open_time,
        }

    # ============================================================
    # 3. 单条记录 → LangChain Document
    # ============================================================
    def convert_to_document(self, record: Dict[str, Any]) -> Document:
        """
        将单条结构化景点数据转换为 LangChain Document。

        page_content 拼接规则（自然语言描述）:
            "{name}位于{city}{address}。{description}门票{xx}元。
             开放时间：{open_time}。"

        metadata 绑定的字段:
            name, city, ticket, level, tags, address, open_time

        参数：
            record: 清洗后的单条景点记录 dict

        返回：
            langchain_core.documents.Document 对象

        异常：
            DataCleanError: record 缺少必填字段 name 或 description

        示例：
            processor = ScenicDataProcessor()
            doc = processor.convert_to_document({
                "name": "故宫博物院",
                "city": "北京",
                "description": "明清两代皇家宫殿...",
                "ticket": 60.0,
                "level": "5A",
                "tags": ["世界文化遗产", "古迹"],
                "address": "北京市东城区景山前街4号",
                "open_time": "旺季8:30-17:00",
            })
        """
        if not isinstance(record, dict):
            raise DataCleanError(
                f"convert_to_document 只接受 dict 类型，实际: {type(record).__name__}"
            )

        name: str = record.get("name", "")
        description: str = record.get("description", "")

        if not name.strip() or not description.strip():
            raise DataCleanError(
                f"景点记录缺少必填字段: name='{name}', description='{description[:30]}...'"
            )

        # ---- 构建 page_content ----
        parts: List[str] = [name]

        city: str = record.get("city", "")
        address: str = record.get("address", "")
        if city or address:
            location = f"位于{city}{address}" if address else f"位于{city}"
            parts.append(location + "。")

        parts.append(description)

        ticket: float = record.get("ticket", 0.0)
        if ticket > 0:
            parts.append(f"门票{int(ticket) if ticket == int(ticket) else ticket}元。")
        else:
            parts.append("免费开放。")

        open_time: str = record.get("open_time", "")
        if open_time:
            parts.append(f"开放时间：{open_time}。")

        page_content: str = "".join(parts)

        # ---- 构建 metadata ----
        metadata: Dict[str, Any] = {
            "name": name,
            "city": city,
            "ticket": ticket,
            "level": record.get("level", ""),
            "tags": record.get("tags", []),
            "address": address,
            "open_time": open_time,
        }

        return Document(page_content=page_content, metadata=metadata)

    # ============================================================
    # 4. 批量转换 + 向量库入库
    # ============================================================
    def batch_import_to_vector(
        self,
        documents: List[Document],
        vector_store: Optional[ScenicVectorStore] = None,
        batch_size: Optional[int] = None,
    ) -> int:
        """
        批量将 Document 列表写入向量库（幂等入库）。

        流程：
            1. 若未传入 vector_store，自动创建默认实例
            2. 按 batch_size 分批调用 ScenicVectorStore.add_scenic_docs
            3. 每批写入后自动持久化

        参数：
            documents:     LangChain Document 对象列表
            vector_store:  向量库实例，为 None 时自动创建默认实例
            batch_size:    每批入库条数，默认从 settings 读取

        返回：
            成功入库的文档总数

        异常：
            DataImportError: 文档列表为空、向量库写入失败

        示例：
            processor = ScenicDataProcessor()
            docs = [processor.convert_to_document(r) for r in cleaned]
            count = processor.batch_import_to_vector(docs)
        """
        if not documents:
            raise DataImportError(
                "batch_import_to_vector 文档列表为空，无数据可入库。"
            )

        batch_size = batch_size or self._batch_size

        if vector_store is None:
            vector_store = ScenicVectorStore()

        total: int = len(documents)
        imported: int = 0

        print(
            f"[ScenicDataProcessor] 开始批量入库: 共 {total} 条, "
            f"batch_size={batch_size}"
        )

        for start in range(0, total, batch_size):
            batch = documents[start: start + batch_size]
            try:
                count = vector_store.add_scenic_docs(batch)
                imported += count
                print(
                    f"[ScenicDataProcessor] 批次入库: {count} 条 "
                    f"({min(start + batch_size, total)}/{total})"
                )
            except Exception as e:
                raise DataImportError(
                    f"批量入库失败 (已入库 {imported}/{total} 条), "
                    f"批次 [{start}:{start + batch_size}]。"
                    f"原始错误: {type(e).__name__}: {e}"
                ) from e

        print(
            f"[ScenicDataProcessor] 入库完成: {imported}/{total} 条"
        )
        return imported

    # ============================================================
    # 内部：清洗辅助方法
    # ============================================================
    @staticmethod
    def _clean_text(text: Any) -> str:
        """
        清洗文本字段：去 HTML 标签、多余空白、不可见字符。

        参数：
            text: 原始值（可能是 None / 非字符串）

        返回：
            清洗后的字符串
        """
        if text is None:
            return ""
        if not isinstance(text, str):
            text = str(text)

        # 去除 HTML 标签
        text = re.sub(r"<[^>]+>", "", text)
        # 合并连续空白
        text = re.sub(r"\s+", " ", text)
        # 去除首尾空白
        text = text.strip()
        return text

    @staticmethod
    def _normalize_ticket(value: Any) -> float:
        """
        票价标准化：字符串 "60元" / "免费" → float。

        规则：
            - 数字 → float
            - "免费" / "free" / 空 / None → 0.0
            - "60元" / "￥60" → 60.0
            - 非法值 → 0.0

        参数：
            value: 原始票价值

        返回：
            标准化后的票价 float
        """
        if value is None:
            return 0.0

        if isinstance(value, (int, float)):
            return float(max(0, value))

        if isinstance(value, str):
            s = value.strip()
            if not s or s.lower() in ("免费", "free", "无", "none"):
                return 0.0
            # 尝试提取数字
            match = re.search(r"[\d.]+", s)
            if match:
                try:
                    return float(match.group())
                except ValueError:
                    pass

        return 0.0

    @staticmethod
    def _normalize_tags(raw_tags: Any) -> List[str]:
        """
        标签标准化：逗号/顿号分隔字符串 → 去重去空格 list，过滤非法标签。

        参数：
            raw_tags: 原始标签值（字符串 / 列表）

        返回：
            标准化后的标签列表
        """
        if raw_tags is None:
            return []

        tags_list: List[str] = []

        if isinstance(raw_tags, str):
            # 逗号/顿号/空格分隔
            parts = re.split(r"[,，、\s]+", raw_tags.strip())
            tags_list = [p.strip() for p in parts if p.strip()]
        elif isinstance(raw_tags, list):
            tags_list = [
                str(t).strip() for t in raw_tags if t and str(t).strip()
            ]
        else:
            return []

        # 去重 + 过滤非法标签
        seen: set = set()
        result: List[str] = []
        for tag in tags_list:
            if tag in _VALID_TAGS and tag not in seen:
                seen.add(tag)
                result.append(tag)

        return result

    @staticmethod
    def _normalize_city(city: str) -> str:
        """
        城市名称标准化：去空格、去末尾"市"字。

        参数：
            city: 原始城市名称

        返回：
            标准化后的城市名称
        """
        city = city.strip()
        if city.endswith("市"):
            city = city[:-1]
        return city

    # ============================================================
    # 辅助：一键处理
    # ============================================================
    def process_and_import(
        self,
        file_name: Optional[str] = None,
        vector_store: Optional[ScenicVectorStore] = None,
    ) -> int:
        """
        一键执行「加载 → 清洗 → 转换 → 入库」全流程。

        参数：
            file_name:     JSON 文件名（默认自动选择）
            vector_store:  向量库实例（默认自动创建）

        返回：
            入库文档总数

        异常：
            DataLoadError / DataCleanError / DataImportError

        示例：
            processor = ScenicDataProcessor()
            count = processor.process_and_import("scenic_spots.json")
            print(f"入库 {count} 条景点数据")
        """
        print("[ScenicDataProcessor] ===== 开始一键处理 =====")

        # 1) 加载
        raw = self.load_raw_json(file_name)

        # 2) 清洗
        cleaned = self.clean_raw_data(raw)

        # 3) 转换
        docs = [self.convert_to_document(record) for record in cleaned]
        print(f"[ScenicDataProcessor] 转换完成: {len(docs)} 条 Document")

        # 4) 入库
        count = self.batch_import_to_vector(docs, vector_store=vector_store)

        print("[ScenicDataProcessor] ===== 一键处理完毕 =====")
        return count
