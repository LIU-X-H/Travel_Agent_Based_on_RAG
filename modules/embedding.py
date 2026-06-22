"""
Embedding 向量化模块
====================
基于 LangChain HuggingFaceEmbeddings 封装 BGE 中文 Embedding 模型。

特性：
- 单例模式：全局只加载一次模型，避免重复占用显存/内存
- 参数零硬编码：模型名称、设备、归一化等全部从 config.settings 读取
- 接口兼容 LangChain：embed_query / embed_documents 可直接对接 Chroma 等向量库
- 完善异常处理：覆盖模型下载失败、显存不足、空输入等场景

使用示例：
    from modules.embedding import EmbeddingModel

    model = EmbeddingModel()           # 获取单例（首次触发模型加载）
    vec = model.embed_query("故宫")     # 单条查询向量化
    vecs = model.embed_documents([     # 批量文档向量化
        "故宫是中国著名的旅游景点",
        "长城是世界文化遗产",
    ])
"""

import threading
from typing import List, Optional

from config.settings import settings


# ============================================================
# 自定义异常类
# ============================================================
class EmbeddingError(Exception):
    """Embedding 模块基础异常。"""
    pass


class EmbeddingModelLoadError(EmbeddingError):
    """模型加载失败异常 —— 网络下载失败、模型文件损坏等。"""
    pass


class EmbeddingInferenceError(EmbeddingError):
    """向量推理失败异常 —— 显存不足、输入异常等。"""
    pass


class EmbeddingInputError(EmbeddingError):
    """输入参数异常 —— 空文本、类型错误等。"""
    pass


# ============================================================
# Embedding 模型单例
# ============================================================
class EmbeddingModel:
    """
    BGE 中文 Embedding 模型封装（单例模式）
    ---------------------------------------
    基于 LangChain HuggingFaceEmbeddings，提供与向量库完全兼容的接口。

    单例保证：
        - 全局仅一个实例，重复调用 EmbeddingModel() 返回同一对象
        - 线程安全：通过 threading.Lock 防止并发初始化时重复加载
        - 模型仅加载一次，后续调用零开销

    LangChain 兼容接口：
        - embed_query(text: str) -> List[float]
        - embed_documents(texts: List[str]) -> List[List[float]]
    """

    # ---- 单例相关 ----
    _instance: Optional["EmbeddingModel"] = None
    _lock: threading.Lock = threading.Lock()
    _initialized: bool = False

    def __new__(cls) -> "EmbeddingModel":
        """
        单例构造：若实例已存在则直接返回，否则创建新实例。
        线程安全：双重检查锁定（DCL），防止并发场景下重复创建。
        """
        if cls._instance is None:
            with cls._lock:
                # 进入锁后再次检查 —— 可能已有其他线程完成初始化
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """
        初始化 Embedding 模型。

        注意：
            - 模型采用延迟加载：__init__ 只读取配置，不加载模型权重
            - 首次调用 embed_query / embed_documents 时自动触发加载
            - 单例保证 __init__ 只执行一次（通过 _initialized 标志位）
        """
        if EmbeddingModel._initialized:
            return

        # ---- 从全局配置读取所有参数（零硬编码） ----
        self._model_name: str = settings.EMBEDDING_MODEL_NAME
        self._device: str = settings.EMBEDDING_DEVICE
        self._normalize: bool = settings.EMBEDDING_NORMALIZE
        self._dim: int = settings.EMBEDDING_DIM

        # ---- 底层 LangChain Embeddings 实例（延迟加载） ----
        self._embeddings = None

        EmbeddingModel._initialized = True

        print(
            f"[EmbeddingModel] 配置就绪: model={self._model_name}, "
            f"device={self._device}, normalize={self._normalize}, dim={self._dim}"
        )

    # ============================================================
    # 模型加载
    # ============================================================
    def _load_model(self) -> None:
        """
        延迟加载 Embedding 模型权重到内存。

        加载策略：
            - device='auto' 时优先 CUDA，不可用则回退 CPU
            - 加载成功后设置 self._embeddings，后续调用直接复用

        异常场景：
            - 模型文件不存在 / 网络不可达 -> EmbeddingModelLoadError
            - CUDA 显存不足 -> EmbeddingModelLoadError（引导回退 CPU）
        """
        if self._embeddings is not None:
            return  # 已加载，跳过

        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings

            # 处理设备参数：auto 模式下自动检测 CUDA 可用性
            device: str = self._device
            if device == "auto":
                try:
                    import torch
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                except ImportError:
                    device = "cpu"

            # 构造 LangChain HuggingFaceEmbeddings
            model_kwargs: dict = {"device": device}
            encode_kwargs: dict = {"normalize_embeddings": self._normalize}

            self._embeddings = HuggingFaceEmbeddings(
                model_name=self._model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
            )

            print(
                f"[EmbeddingModel] 模型加载成功: {self._model_name} "
                f"(device={device}, dim={self._dim})"
            )

        except ImportError as e:
            raise EmbeddingModelLoadError(
                f"缺少依赖包，请安装: pip install langchain-community sentence-transformers。"
                f"原始错误: {e}"
            ) from e

        except OSError as e:
            # 网络错误、磁盘空间不足、模型路径不存在等
            raise EmbeddingModelLoadError(
                f"模型文件下载或读取失败，请检查网络连接与磁盘空间。"
                f"模型名称: {self._model_name}，原始错误: {e}"
            ) from e

        except RuntimeError as e:
            error_msg = str(e).lower()
            if "out of memory" in error_msg or "cuda" in error_msg:
                raise EmbeddingModelLoadError(
                    f"CUDA 显存不足，模型加载失败。"
                    f"请在 .env 中设置 EMBEDDING_DEVICE=cpu 使用 CPU 模式。"
                    f"原始错误: {e}"
                ) from e
            raise EmbeddingModelLoadError(
                f"模型加载时发生运行时错误: {e}"
            ) from e

        except Exception as e:
            raise EmbeddingModelLoadError(
                f"加载 Embedding 模型时发生未知错误: {type(e).__name__}: {e}"
            ) from e

    # ============================================================
    # 对外接口（兼容 LangChain 向量库标准）
    # ============================================================
    def embed_query(self, text: str) -> List[float]:
        """
        将单条查询文本转换为向量。

        参数：
            text: 查询文本（单条字符串），例如 "北京故宫门票多少钱"

        返回：
            浮点数列表，长度 = embedding_dim（默认 512）。
            可直接传给 Chroma / FAISS 等向量库的 query 接口。

        异常：
            EmbeddingInputError: 输入为空字符串或非字符串类型
            EmbeddingInferenceError: 推理过程出错（显存不足等）
            EmbeddingModelLoadError: 模型尚未加载且加载失败

        示例：
            model = EmbeddingModel()
            query_vec = model.embed_query("推荐北京的热门景点")
        """
        # ---- 输入校验 ----
        if not isinstance(text, str):
            raise EmbeddingInputError(
                f"embed_query 只接受 str 类型，实际传入: {type(text).__name__}。"
            )

        if not text.strip():
            raise EmbeddingInputError(
                "embed_query 输入文本为空字符串，请提供有效的查询文本。"
            )

        # ---- 确保模型已加载 ----
        if self._embeddings is None:
            self._load_model()

        # ---- 执行推理 ----
        try:
            vector: List[float] = self._embeddings.embed_query(text)
            return vector

        except RuntimeError as e:
            error_msg = str(e).lower()
            if "out of memory" in error_msg:
                raise EmbeddingInferenceError(
                    f"查询向量化时 CUDA 显存不足。"
                    f"建议: (1) 减小批量大小 (2) 在 .env 中切换 EMBEDDING_DEVICE=cpu。"
                    f"原始错误: {e}"
                ) from e
            raise EmbeddingInferenceError(
                f"查询向量化时发生运行时错误: {e}"
            ) from e

        except Exception as e:
            raise EmbeddingInferenceError(
                f"查询向量化失败: {type(e).__name__}: {e}"
            ) from e

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        将批量文档文本转换为向量。

        参数：
            texts: 文档文本列表，例如 ["故宫简介文本...", "长城介绍文本..."]

        返回：
            二维浮点数列表，shape = (len(texts), embedding_dim)。
            可直接传给 Chroma / FAISS 等向量库的 add_documents 接口。

        异常：
            EmbeddingInputError: 输入为空列表、列表中有非字符串元素
            EmbeddingInferenceError: 推理过程出错
            EmbeddingModelLoadError: 模型尚未加载且加载失败

        示例：
            model = EmbeddingModel()
            doc_vecs = model.embed_documents([
                "故宫是世界上规模最大的宫殿建筑群",
                "长城是中国古代伟大的防御工程",
            ])
        """
        # ---- 输入校验 ----
        if not isinstance(texts, list):
            raise EmbeddingInputError(
                f"embed_documents 只接受 list 类型，实际传入: {type(texts).__name__}。"
            )

        if len(texts) == 0:
            raise EmbeddingInputError(
                "embed_documents 输入列表为空，请提供至少一条文档文本。"
            )

        # 检查列表中每个元素是否为非空字符串
        for idx, item in enumerate(texts):
            if not isinstance(item, str):
                raise EmbeddingInputError(
                    f"embed_documents 第 {idx} 个元素不是 str 类型，"
                    f"实际类型: {type(item).__name__}。"
                )
            if not item.strip():
                raise EmbeddingInputError(
                    f"embed_documents 第 {idx} 个元素为空字符串，请移除或填充有效文本。"
                )

        # ---- 确保模型已加载 ----
        if self._embeddings is None:
            self._load_model()

        # ---- 执行批量推理 ----
        try:
            vectors: List[List[float]] = self._embeddings.embed_documents(texts)
            return vectors

        except RuntimeError as e:
            error_msg = str(e).lower()
            if "out of memory" in error_msg:
                raise EmbeddingInferenceError(
                    f"批量向量化时 CUDA 显存不足（共 {len(texts)} 条文本）。"
                    f"建议: (1) 减小批次大小 (2) 在 .env 中切换 EMBEDDING_DEVICE=cpu。"
                    f"原始错误: {e}"
                ) from e
            raise EmbeddingInferenceError(
                f"批量向量化时发生运行时错误: {e}"
            ) from e

        except Exception as e:
            raise EmbeddingInferenceError(
                f"批量向量化失败: {type(e).__name__}: {e}"
            ) from e

    # ============================================================
    # 辅助属性
    # ============================================================
    @property
    def is_loaded(self) -> bool:
        """检查底层模型是否已加载到内存。"""
        return self._embeddings is not None

    @property
    def model_name(self) -> str:
        """返回当前使用的模型名称。"""
        return self._model_name

    @property
    def dim(self) -> int:
        """返回 Embedding 向量的维度。"""
        return self._dim

    @classmethod
    def reset_instance(cls) -> None:
        """
        重置单例（主要用于测试场景）。
        调用后下一次 EmbeddingModel() 会重新创建实例并加载模型。

        注意：生产环境通常不需要调用此方法。
        """
        with cls._lock:
            cls._instance = None
            cls._initialized = False
        print("[EmbeddingModel] 单例已重置")
