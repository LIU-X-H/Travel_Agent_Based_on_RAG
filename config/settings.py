"""
全局配置模块
-----------
所有可调参数在此集中定义，通过环境变量或默认值统一管理。
使用方式：
    from config import Settings
    settings = Settings()
    print(settings.VECTOR_DB_PATH)
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv


# ============================================================
# 加载 .env 环境变量文件（优先级：系统环境变量 > .env > 默认值）
# ============================================================
load_dotenv()


# ============================================================
# 项目根目录
# ============================================================
PROJECT_ROOT: Path = Path(__file__).parent.parent.resolve()


class Settings:
    """
    全局配置单例
    ----------
    所有参数均为类属性，通过环境变量覆盖默认值。
    禁止在业务代码中硬编码路径/阈值/模型名，统一从此处读取。
    """

    # ======================== 项目路径 ========================
    # 项目根目录（绝对路径）
    PROJECT_ROOT: Path = PROJECT_ROOT

    # 数据根目录
    DATA_DIR: Path = PROJECT_ROOT / "data"

    # 原始数据目录：存放爬取的景点原始文本、JSON 等
    RAW_DATA_DIR: Path = DATA_DIR / "raw"

    # 处理后数据目录：存放清洗、分块后的文档
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

    # 向量数据库持久化目录
    VECTOR_DB_PATH: Path = Path(
        os.getenv("VECTOR_DB_PATH", str(DATA_DIR / "vector_db"))
    )

    # ======================== Embedding 模型 ========================
    # BGE 中文 Embedding 模型名称（HuggingFace 模型 ID 或本地路径）
    # bge-small-zh-v1.5: 轻量版，维度512，适合本地开发与快速推理
    EMBEDDING_MODEL_NAME: str = os.getenv(
        "EMBEDDING_MODEL_NAME", "BAAI/bge-small-zh-v1.5"
    )

    # Embedding 运行设备：cpu / cuda / auto（自动选择 GPU > CPU）
    EMBEDDING_DEVICE: str = os.getenv("EMBEDDING_DEVICE", "auto")

    # Embedding 向量维度（由模型决定，bge-small-zh-v1.5 为 512）
    EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", "512"))

    # 是否对 Embedding 做归一化（BGE 模型推荐开启）
    EMBEDDING_NORMALIZE: bool = os.getenv("EMBEDDING_NORMALIZE", "true").lower() == "true"

    # ======================== 向量库 (Chroma) ========================
    # Chroma 集合名称
    CHROMA_COLLECTION_NAME: str = os.getenv(
        "CHROMA_COLLECTION_NAME", "travel_scenic_spots"
    )

    # Chroma 距离度量方式：cosine / l2 / ip
    CHROMA_DISTANCE_METRIC: str = os.getenv(
        "CHROMA_DISTANCE_METRIC", "cosine"
    )

    # ======================== 检索参数 ========================
    # 语义检索返回的候选文档数（向量检索 top_k）
    SEMANTIC_TOP_K: int = int(os.getenv("SEMANTIC_TOP_K", "10"))

    # BM25 关键词检索返回的候选文档数
    BM25_TOP_K: int = int(os.getenv("BM25_TOP_K", "10"))

    # 混合检索最终返回的文档数
    HYBRID_TOP_K: int = int(os.getenv("HYBRID_TOP_K", "5"))

    # 相似度阈值：低于此分数的文档将被过滤（0.0 ~ 1.0）
    SIMILARITY_THRESHOLD: float = float(
        os.getenv("SIMILARITY_THRESHOLD", "0.0")
    )

    # ======================== 混合检索权重 ========================
    # 语义检索权重 α（0.0 ~ 1.0），BM25 权重 = 1 - α
    # α 越大，语义检索占比越高；α 越小，关键词匹配占比越高
    HYBRID_ALPHA: float = float(os.getenv("HYBRID_ALPHA", "0.7"))

    # ======================== 重排序 (ReRank) ========================
    # 是否启用重排序
    RERANK_ENABLED: bool = os.getenv("RERANK_ENABLED", "false").lower() == "true"

    # ReRank 模型名称（BGE Reranker 系列）
    RERANK_MODEL_NAME: str = os.getenv(
        "RERANK_MODEL_NAME", "BAAI/bge-reranker-large"
    )

    # ReRank 保留的文档数
    RERANK_TOP_K: int = int(os.getenv("RERANK_TOP_K", "3"))

    # ======================== 文本分块参数 ========================
    # 文本分块大小（字符数）
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "512"))

    # 文本分块重叠大小（字符数）
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "64"))

    # 数据批量导入向量库的批次大小
    DATA_IMPORT_BATCH_SIZE: int = int(os.getenv("DATA_IMPORT_BATCH_SIZE", "32"))

    # ======================== LLM 大模型参数 ========================
    # LLM 模型名称（兼容 OpenAI API 格式）
    LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "gpt-3.5-turbo")

    # LLM API 地址
    LLM_API_BASE: Optional[str] = os.getenv("LLM_API_BASE")

    # LLM API Key
    LLM_API_KEY: Optional[str] = os.getenv("LLM_API_KEY")

    # LLM 生成温度
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))

    # LLM 最大 Token 数
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "2048"))

    # ======================== API 服务参数 ========================
    # FastAPI 服务 host
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")

    # FastAPI 服务端口
    API_PORT: int = int(os.getenv("API_PORT", "8000"))

    # 是否开启调试模式
    API_DEBUG: bool = os.getenv("API_DEBUG", "true").lower() == "true"

    # ======================== 日志参数 ========================
    # 日志级别：DEBUG / INFO / WARNING / ERROR
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # 日志文件路径（为空则仅输出到控制台）
    LOG_FILE: Optional[str] = os.getenv("LOG_FILE")

    # ======================== 辅助方法 ========================
    @classmethod
    def as_dict(cls) -> Dict[str, Any]:
        """
        将配置导出为字典（仅公有参数，过滤私有/魔术属性）
        用于日志记录或传递给下游组件。
        """
        return {
            key: str(value) if isinstance(value, Path) else value
            for key, value in cls.__dict__.items()
            if not key.startswith("_") and not callable(value)
        }

    @classmethod
    def ensure_directories(cls) -> None:
        """
        确保所有必需的目录存在，不存在则自动创建。
        在应用启动时调用，防止运行时因目录缺失报错。
        """
        dirs_to_create: list[Path] = [
            cls.DATA_DIR,
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.VECTOR_DB_PATH,
        ]
        for dir_path in dirs_to_create:
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                print(f"[WARNING] 无法创建目录 {dir_path}: {e}")


# ============================================================
# 模块级单例：方便 from config import settings 直接使用
# ============================================================
settings = Settings()
