"""
Chroma 向量库封装模块
=====================
基于 LangChain Chroma 向量库封装的旅游景点向量存储层。

特性：
- 自动创建/加载持久化向量库，无需手动管理文件
- 所有参数从 config.settings 统一读取，零硬编码
- 内置 EmbeddingModel 单例，自动复用已加载的 BGE 模型
- 支持按城市（metadata.city）增量删除，实现数据热更新
- 提供 LangChain 标准 Retriever 接口，无缝对接上层检索模块
- 完善的异常处理：覆盖向量库读写失败、空文档、Embedding 未就绪等场景

使用示例：
    from modules.vector_store import ScenicVectorStore
    from langchain_core.documents import Document

    store = ScenicVectorStore()
    docs = [Document(page_content="故宫...", metadata={"city": "北京"})]
    store.add_scenic_docs(docs)
    results = store.base_similarity_search("北京有哪些古迹", top_k=5)
"""

import os
import shutil
from typing import List, Optional, Tuple, Dict, Any

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from config.settings import settings
from modules.embedding import EmbeddingModel


# ============================================================
# 自定义异常类
# ============================================================
class VectorStoreError(Exception):
    """向量库模块基础异常。"""
    pass


class VectorStoreInitError(VectorStoreError):
    """向量库初始化失败异常 —— 路径无权限、依赖缺失等。"""
    pass


class VectorStoreWriteError(VectorStoreError):
    """向量库写入失败异常 —— 磁盘满、权限不足、数据格式错误等。"""
    pass


class VectorStoreQueryError(VectorStoreError):
    """向量库查询/检索失败异常。"""
    pass


class VectorStoreDeleteError(VectorStoreError):
    """向量库删除操作失败异常。"""
    pass


# ============================================================
# 旅游景点向量库
# ============================================================
class ScenicVectorStore:
    """
    旅游景点 Chroma 向量库封装
    --------------------------
    封装 LangChain Chroma 向量库，提供景点文档的入库、检索、删除功能。

    核心设计：
        - 构造时自动连接或创建 Chroma 持久化集合
        - 内嵌 EmbeddingModel 单例，自动处理文本向量化
        - 所有配置参数从 config.settings 读取，零硬编码
        - 提供 LangChain Retriever 对象，供上层 RAG 检索链调用

    属性：
        store: 底层 LangChain Chroma 向量库实例
        embedding_model: EmbeddingModel 单例引用
        collection_name: Chroma 集合名称
    """

    def __init__(
        self,
        collection_name: Optional[str] = None,
        persist_directory: Optional[str] = None,
    ) -> None:
        """
        初始化向量库连接。

        行为：
            - 若本地持久化目录已存在集合数据，直接加载
            - 若不存在，自动创建新集合（首次调用 add_scenic_docs 时持久化）

        参数：
            collection_name: Chroma 集合名称，默认从 settings 读取
            persist_directory: 持久化目录路径，默认从 settings 读取

        异常：
            VectorStoreInitError: Chroma 依赖缺失或路径创建失败
        """
        # ---- 从全局配置读取参数（零硬编码） ----
        self._collection_name: str = (
            collection_name or settings.CHROMA_COLLECTION_NAME
        )
        self._persist_dir: str = (
            persist_directory or str(settings.VECTOR_DB_PATH)
        )
        self._distance_metric: str = settings.CHROMA_DISTANCE_METRIC
        self._default_top_k: int = settings.SEMANTIC_TOP_K
        self._similarity_threshold: float = settings.SIMILARITY_THRESHOLD

        # ---- 获取 Embedding 模型单例 ----
        # 注意：此处不立即加载模型权重（延迟加载），仅持有引用
        try:
            self._embedding_model: EmbeddingModel = EmbeddingModel()
        except Exception as e:
            raise VectorStoreInitError(
                f"获取 EmbeddingModel 单例失败: {e}"
            ) from e

        # ---- 初始化 Chroma 向量库 ----
        self._store = None  # 底层 LangChain Chroma 实例
        self._init_vector_store()

        print(
            f"[ScenicVectorStore] 向量库就绪: collection={self._collection_name}, "
            f"persist_dir={self._persist_dir}, top_k={self._default_top_k}, "
            f"threshold={self._similarity_threshold}"
        )

    # ============================================================
    # 内部：向量库初始化
    # ============================================================
    def _init_vector_store(self) -> None:
        """
        初始化 LangChain Chroma 向量库实例。

        逻辑：
            1. 确保持久化目录存在
            2. 使用 EmbeddingModel 作为 embedding_function
            3. 构造 Chroma 实例（自动加载已有数据或创建新集合）

        异常：
            VectorStoreInitError: 依赖缺失或初始化失败
        """
        # 确保持久化目录存在
        try:
            os.makedirs(self._persist_dir, exist_ok=True)
        except OSError as e:
            raise VectorStoreInitError(
                f"无法创建向量库持久化目录: {self._persist_dir}，"
                f"请检查磁盘空间和写入权限。原始错误: {e}"
            ) from e

        # 导入 Chroma（延迟导入，避免未安装依赖时阻塞其他模块加载）
        try:
            from langchain_community.vectorstores import Chroma
        except ImportError as e:
            raise VectorStoreInitError(
                f"缺少 langchain-community 依赖，请安装: "
                f"pip install langchain-community。原始错误: {e}"
            ) from e

        # 确保 Embedding 模型已加载
        if not self._embedding_model.is_loaded:
            try:
                self._embedding_model._load_model()
            except Exception as e:
                raise VectorStoreInitError(
                    f"Embedding 模型加载失败，向量库无法初始化: {e}"
                ) from e

        # 构造 Chroma 实例
        # collection_metadata 中的 "hnsw:space" 控制距离度量方式
        try:
            self._store = Chroma(
                collection_name=self._collection_name,
                embedding_function=self._embedding_model,
                persist_directory=self._persist_dir,
                collection_metadata={
                    "hnsw:space": self._distance_metric,
                },
            )
        except Exception as e:
            raise VectorStoreInitError(
                f"Chroma 向量库初始化失败，"
                f"请检查持久化路径是否可写、数据是否损坏。"
                f"路径: {self._persist_dir}，原始错误: {e}"
            ) from e

    # ============================================================
    # 对外方法：入库
    # ============================================================
    def add_scenic_docs(
        self,
        documents: List[Document],
        batch_size: int = 64,
    ) -> int:
        """
        批量添加景点文档到向量库并持久化。

        参数：
            documents: LangChain Document 对象列表。
                       每个 Document 需包含:
                         - page_content (str): 景点描述文本
                         - metadata   (dict): 至少包含 city 字段，
                           例如 {"city": "北京", "ticket": 60, "tags": ["古迹", "5A"]}
            batch_size: 每批次向量化的文档数（避免显存溢出），默认 64

        返回：
            成功入库的文档总数 (int)

        异常：
            VectorStoreWriteError: 文档列表为空、向量化失败、持久化失败
            EmbeddingInputError: 文档内容为空字符串（由 EmbeddingModel 抛出）

        示例：
            from langchain_core.documents import Document

            docs = [
                Document(
                    page_content="故宫是明清两代的皇家宫殿...",
                    metadata={"city": "北京", "ticket": 60, "tags": ["5A", "古迹"]}
                ),
            ]
            store = ScenicVectorStore()
            count = store.add_scenic_docs(docs)
        """
        # ---- 输入校验 ----
        if not isinstance(documents, list):
            raise VectorStoreWriteError(
                f"add_scenic_docs 只接受 list 类型，实际传入: "
                f"{type(documents).__name__}。"
            )

        if len(documents) == 0:
            raise VectorStoreWriteError(
                "add_scenic_docs 文档列表为空，请提供至少一个 LangChain Document 对象。"
            )

        # 校验每个元素都是 Document 且 page_content 非空
        for idx, doc in enumerate(documents):
            if not isinstance(doc, Document):
                raise VectorStoreWriteError(
                    f"add_scenic_docs 第 {idx} 个元素不是 LangChain Document 类型，"
                    f"实际类型: {type(doc).__name__}。"
                )
            if not doc.page_content or not doc.page_content.strip():
                raise VectorStoreWriteError(
                    f"add_scenic_docs 第 {idx} 个 Document 的 page_content 为空，"
                    f"请填充景点描述文本。metadata={doc.metadata}"
                )

        # ---- 确保向量库已初始化 ----
        if self._store is None:
            self._init_vector_store()

        # ---- 分批向量化并入库 ----
        total_count: int = 0
        total_docs: int = len(documents)

        try:
            for start in range(0, total_docs, batch_size):
                batch = documents[start: start + batch_size]

                # 为每条文档生成唯一 ID（基于内容哈希 + 序号，支持幂等入库）
                import hashlib
                batch_ids: List[str] = []
                for doc in batch:
                    content_hash = hashlib.md5(
                        doc.page_content.encode("utf-8")
                    ).hexdigest()[:12]
                    doc_id = f"scenic_{doc.metadata.get('city', 'unknown')}_{content_hash}"
                    batch_ids.append(doc_id)

                # 调用 LangChain Chroma 的 add_documents（自动向量化 + 持久化）
                self._store.add_documents(
                    documents=batch,
                    ids=batch_ids,
                )
                total_count += len(batch)
                print(
                    f"[ScenicVectorStore] 批次入库: {len(batch)} 条 "
                    f"({total_count}/{total_docs})"
                )

        except RuntimeError as e:
            error_msg = str(e).lower()
            if "out of memory" in error_msg:
                raise VectorStoreWriteError(
                    f"向量化时显存不足（已入库 {total_count}/{total_docs} 条）。"
                    f"建议: (1) 减小 batch_size (当前 {batch_size}) "
                    f"(2) 在 .env 中切换 EMBEDDING_DEVICE=cpu。"
                    f"原始错误: {e}"
                ) from e
            raise VectorStoreWriteError(
                f"批量入库时发生运行时错误（已入库 {total_count}/{total_docs} 条）: {e}"
            ) from e

        except OSError as e:
            raise VectorStoreWriteError(
                f"向量库持久化写入失败，请检查磁盘空间。"
                f"持久化路径: {self._persist_dir}，原始错误: {e}"
            ) from e

        except Exception as e:
            raise VectorStoreWriteError(
                f"批量入库失败（已入库 {total_count}/{total_docs} 条）: "
                f"{type(e).__name__}: {e}"
            ) from e

        print(
            f"[ScenicVectorStore] 入库完成: 共 {total_count} 条文档，"
            f"集合总计约 {self.count()} 条"
        )
        return total_count

    # ============================================================
    # 对外方法：删除
    # ============================================================
    def delete_by_city(self, city: str) -> int:
        """
        按城市删除对应全部景点数据（用于增量更新）。

        参数：
            city: 城市名称，与入库时 metadata.city 精确匹配。
                  例如 "北京"、"杭州"、"成都"

        返回：
            实际删除的文档数量 (int)

        异常：
            VectorStoreDeleteError: Chroma 查询/删除操作失败
            VectorStoreError: city 参数为空

        示例：
            store = ScenicVectorStore()
            deleted = store.delete_by_city("北京")
            print(f"已删除 {deleted} 条北京景点数据")
        """
        # ---- 输入校验 ----
        if not isinstance(city, str) or not city.strip():
            raise VectorStoreError(
                f"delete_by_city 参数 city 必须是非空字符串，实际: {city!r}"
            )

        if self._store is None:
            self._init_vector_store()

        try:
            # 1) 获取底层 Chroma 原生 collection
            collection = self._store._collection

            # 2) 按 metadata.city 精确匹配查询所有相关文档 ID
            result = collection.get(
                where={"city": city.strip()},
                include=["metadatas"],  # 只需 metadata，无需加载向量
            )

            ids_to_delete: List[str] = result.get("ids", [])

            if not ids_to_delete:
                print(
                    f"[ScenicVectorStore] delete_by_city: "
                    f"未找到 city='{city}' 的文档，跳过删除"
                )
                return 0

            # 3) 批量删除
            collection.delete(ids=ids_to_delete)
            deleted_count: int = len(ids_to_delete)

            print(
                f"[ScenicVectorStore] delete_by_city: "
                f"已删除 city='{city}' 共 {deleted_count} 条文档"
            )
            return deleted_count

        except KeyError as e:
            raise VectorStoreDeleteError(
                f"删除时访问 Chroma 字段失败（集合结构异常）: {e}"
            ) from e
        except Exception as e:
            raise VectorStoreDeleteError(
                f"按城市删除失败 (city='{city}'): {type(e).__name__}: {e}"
            ) from e

    # ============================================================
    # 对外方法：检索
    # ============================================================
    def base_similarity_search(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        基础语义向量检索。

        检索流程：
            1. 将查询文本向量化
            2. 在 Chroma 集合中按余弦相似度检索 top_k 个最相似文档
            3. 将 Chroma 距离转换为相似度分数（cosine similarity = 1 - distance）
            4. 按相似度阈值过滤低分文档

        参数：
            query:          查询文本，例如 "北京有哪些5A级景点"
            top_k:          返回文档数量，默认从 settings.SEMANTIC_TOP_K 读取
            score_threshold: 相似度阈值（0.0 ~ 1.0），低于此值的文档被过滤。
                             默认从 settings.SIMILARITY_THRESHOLD 读取。
                             cosine 距离下 1.0 = 完全相同，0.0 = 完全不相关

        返回：
            字典列表，每项包含：
                {
                    "content":  str,   # 景点描述文本
                    "metadata": dict,  # 原始元数据（city/ticket/tags等）
                    "score":    float, # 余弦相似度分数（0.0 ~ 1.0，越高越相似）
                }

        异常：
            VectorStoreQueryError: 查询文本为空、检索过程出错

        示例：
            store = ScenicVectorStore()
            results = store.base_similarity_search(
                "西湖有哪些著名景点",
                top_k=3,
                score_threshold=0.5,
            )
            for r in results:
                print(f"[{r['score']:.3f}] {r['content'][:50]}...")
        """
        # ---- 输入校验 ----
        if not isinstance(query, str) or not query.strip():
            raise VectorStoreQueryError(
                f"检索查询文本必须是非空字符串，实际: {query!r}"
            )

        # ---- 参数默认值（从配置读取） ----
        k: int = top_k if top_k is not None else self._default_top_k
        threshold: float = (
            score_threshold
            if score_threshold is not None
            else self._similarity_threshold
        )

        if k < 1:
            raise VectorStoreQueryError(f"top_k 必须 >= 1，实际: {k}")

        if not (0.0 <= threshold <= 1.0):
            raise VectorStoreQueryError(
                f"score_threshold 必须在 0.0 ~ 1.0 之间，实际: {threshold}"
            )

        # ---- 确保向量库可用 ----
        if self._store is None:
            self._init_vector_store()

        # ---- 执行检索 ----
        try:
            # similarity_search_with_score 返回 List[Tuple[Document, float]]
            # 其中 score 是 Chroma 的余弦距离 (0~2)，需转换为相似度
            raw_results: List[Tuple[Document, float]] = (
                self._store.similarity_search_with_score(
                    query=query,
                    k=k,
                )
            )

        except RuntimeError as e:
            error_msg = str(e).lower()
            if "out of memory" in error_msg:
                raise VectorStoreQueryError(
                    f"检索时显存不足。建议: 在 .env 中设置 EMBEDDING_DEVICE=cpu。"
                    f"原始错误: {e}"
                ) from e
            raise VectorStoreQueryError(
                f"向量检索时发生运行时错误: {e}"
            ) from e

        except Exception as e:
            raise VectorStoreQueryError(
                f"向量检索失败: {type(e).__name__}: {e}"
            ) from e

        # ---- 转换距离为相似度分数 ----
        # Chroma cosine distance 范围 [0, 2]，公式: similarity = 1 - (distance / 2)
        # 转换后范围 [0, 1]，1.0 = 完全相同
        cooked_results: List[Dict[str, Any]] = []
        for doc, distance in raw_results:
            # 余弦距离转余弦相似度
            similarity: float = 1.0 - (distance / 2.0)
            # 确保分数在 [0, 1] 范围内
            similarity = max(0.0, min(1.0, similarity))

            # 相似度阈值过滤
            if similarity < threshold:
                continue

            cooked_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata or {},
                "score": round(similarity, 6),
            })

        return cooked_results

    # ============================================================
    # 对外方法：获取 Retriever
    # ============================================================
    def get_retriever(
        self,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> VectorStoreRetriever:
        """
        返回 LangChain 标准 Retriever 对象。

        用途：
            供上层 RAG 检索链调用，例如：
                retriever = store.get_retriever(top_k=5)
                docs = retriever.invoke("北京有哪些著名景点")

        参数：
            top_k:          检索返回文档数，默认从 settings 读取
            score_threshold: 相似度阈值，默认从 settings 读取

        返回：
            langchain_core.vectorstores.VectorStoreRetriever 实例

        异常：
            VectorStoreInitError: 向量库未初始化

        示例：
            store = ScenicVectorStore()
            retriever = store.get_retriever(top_k=3)
            relevant_docs = retriever.invoke("推荐杭州的景点")
        """
        if self._store is None:
            self._init_vector_store()

        k: int = top_k if top_k is not None else self._default_top_k

        # 使用 LangChain 内置 search_kwargs 配置检索行为
        search_kwargs: Dict[str, Any] = {"k": k}

        # 若设置了相似度阈值，注入过滤逻辑
        threshold = (
            score_threshold
            if score_threshold is not None
            else self._similarity_threshold
        )
        if threshold > 0.0:
            search_kwargs["score_threshold"] = threshold

        retriever: VectorStoreRetriever = self._store.as_retriever(
            search_kwargs=search_kwargs,
        )
        return retriever

    # ============================================================
    # 辅助方法
    # ============================================================
    def count(self) -> int:
        """
        返回向量库中文档总数。

        返回：
            文档数量 (int)，查询失败返回 0
        """
        if self._store is None:
            return 0
        try:
            return self._store._collection.count()
        except Exception as e:
            print(f"[ScenicVectorStore] 获取文档计数失败: {e}")
            return 0

    def get_cities(self) -> List[str]:
        """
        获取向量库中所有不重复的城市列表。

        返回：
            城市名称列表 (按字母排序)

        异常：
            VectorStoreQueryError: 查询失败
        """
        if self._store is None:
            return []

        try:
            collection = self._store._collection
            result = collection.get(include=["metadatas"])

            cities: set = set()
            for meta in (result.get("metadatas") or []):
                city = meta.get("city") if meta else None
                if city:
                    cities.add(city.strip())

            return sorted(cities)

        except Exception as e:
            raise VectorStoreQueryError(
                f"获取城市列表失败: {type(e).__name__}: {e}"
            ) from e

    def reset(self) -> None:
        """
        清空向量库（删除集合及持久化数据）。

        危险操作：会删除所有景点数据，不可恢复。
        仅在开发调试或数据重建时使用。
        """
        if self._store is not None:
            try:
                collection = self._store._collection
                # 获取所有文档 ID 并删除
                ids_result = collection.get(include=[])
                all_ids = ids_result.get("ids", [])
                if all_ids:
                    collection.delete(ids=all_ids)
                    print(
                        f"[ScenicVectorStore] reset: 已删除 {len(all_ids)} 条文档"
                    )
                else:
                    print("[ScenicVectorStore] reset: 向量库已为空")
            except Exception as e:
                raise VectorStoreDeleteError(
                    f"重置向量库失败: {type(e).__name__}: {e}"
                ) from e

    # ============================================================
    # 属性
    # ============================================================
    @property
    def store(self):
        """返回底层 LangChain Chroma 实例（供高级操作使用）。"""
        if self._store is None:
            self._init_vector_store()
        return self._store

    @property
    def embedding_model(self) -> EmbeddingModel:
        """返回 EmbeddingModel 单例引用。"""
        return self._embedding_model

    @property
    def collection_name(self) -> str:
        """返回 Chroma 集合名称。"""
        return self._collection_name
