"""
LLM 客户端封装
--------------
封装 LangChain ChatOpenAI / 兼容 API，提供统一的 LLM 调用接口。
所有参数统一从 config.settings 读取。
"""

from typing import Optional, List, Dict, Any
from config.settings import settings


class LLMClient:
    """
    大模型客户端封装
    ----------------
    封装 LangChain ChatModel，提供统一的对话生成接口。

    使用示例：
        client = LLMClient()
        response = client.chat("请推荐北京的热门旅游景点")
    """

    def __init__(self) -> None:
        """初始化 LLM 参数，延迟加载模型实例。"""
        self._model = None
        self._model_name: str = settings.LLM_MODEL_NAME
        self._api_base: Optional[str] = settings.LLM_API_BASE
        self._api_key: Optional[str] = settings.LLM_API_KEY
        self._temperature: float = settings.LLM_TEMPERATURE
        self._max_tokens: int = settings.LLM_MAX_TOKENS

    def load(self) -> None:
        """
        初始化 LangChain ChatModel 实例。
        若已初始化则跳过。
        """
        if self._model is not None:
            return

        try:
            from langchain_openai import ChatOpenAI

            kwargs: Dict[str, Any] = {
                "model": self._model_name,
                "temperature": self._temperature,
                "max_tokens": self._max_tokens,
            }

            if self._api_base:
                kwargs["base_url"] = self._api_base
            if self._api_key:
                kwargs["api_key"] = self._api_key

            self._model = ChatOpenAI(**kwargs)
            print(f"[LLMClient] 模型已加载: {self._model_name}")

        except ImportError as e:
            raise ImportError(
                "请安装 langchain-openai: pip install langchain-openai"
            ) from e
        except Exception as e:
            raise RuntimeError(f"加载 LLM 失败: {e}") from e

    def chat(self, prompt: str) -> str:
        """
        对话生成（非流式）。

        参数：
            prompt: 用户输入的提示词

        返回：
            模型生成的回复文本
        """
        if self._model is None:
            self.load()

        try:
            response = self._model.invoke(prompt)
            return response.content
        except Exception as e:
            raise RuntimeError(f"LLM 调用失败: {e}") from e

    @property
    def model(self):
        """返回原生 LangChain ChatModel（供高级使用）。"""
        return self._model

    @property
    def is_loaded(self) -> bool:
        """检查模型是否已加载。"""
        return self._model is not None
