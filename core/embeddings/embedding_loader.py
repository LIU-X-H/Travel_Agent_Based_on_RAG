"""
Embedding 加载器
----------------
负责加载 BGE / SentenceTransformer 模型并提供向量化接口。
所有模型参数统一从 config.settings 读取，不在此处硬编码。
"""

from typing import List, Optional, Union
import numpy as np
from config.settings import settings


class EmbeddingLoader:
    """
    Embedding 模型加载器
    --------------------
    封装 SentenceTransformer 模型的加载、推理与资源管理。

    使用示例：
        loader = EmbeddingLoader()
        loader.load()
        vectors = loader.encode(["故宫是中国著名的旅游景点", "长城是世界文化遗产"])
    """

    def __init__(self) -> None:
        """初始化加载器，不立即加载模型（延迟加载）。"""
        self._model = None
        self._model_name: str = settings.EMBEDDING_MODEL_NAME
        self._device: str = settings.EMBEDDING_DEVICE
        self._normalize: bool = settings.EMBEDDING_NORMALIZE

    def load(self) -> None:
        """
        加载 Embedding 模型到内存。
        若已加载则跳过，避免重复加载。
        """
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer

            # auto 模式：优先使用 GPU，否则回退 CPU
            device: Optional[str] = self._device
            if device == "auto":
                try:
                    import torch
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                except ImportError:
                    device = "cpu"

            self._model = SentenceTransformer(
                self._model_name,
                device=device,
            )
            print(
                f"[EmbeddingLoader] 模型已加载: {self._model_name} "
                f"(设备: {device}, 归一化: {self._normalize})"
            )

        except ImportError as e:
            raise ImportError(
                "请安装 sentence-transformers: pip install sentence-transformers"
            ) from e
        except Exception as e:
            raise RuntimeError(f"加载 Embedding 模型失败: {e}") from e

    def encode(
        self,
        texts: Union[str, List[str]],
        *,
        normalize: Optional[bool] = None,
    ) -> np.ndarray:
        """
        将文本转换为向量。

        参数：
            texts: 单个文本字符串或文本列表
            normalize: 是否归一化，默认使用全局配置

        返回：
            numpy 数组，shape = (n_texts, embedding_dim)
        """
        if self._model is None:
            self.load()

        if normalize is None:
            normalize = self._normalize

        try:
            # 统一转为列表
            if isinstance(texts, str):
                texts = [texts]

            embeddings = self._model.encode(
                texts,
                normalize_embeddings=normalize,
                show_progress_bar=False,
            )
            return embeddings

        except Exception as e:
            raise RuntimeError(f"文本向量化失败: {e}") from e

    @property
    def dim(self) -> int:
        """返回 Embedding 向量的维度。"""
        return settings.EMBEDDING_DIM

    @property
    def is_loaded(self) -> bool:
        """检查模型是否已加载。"""
        return self._model is not None
