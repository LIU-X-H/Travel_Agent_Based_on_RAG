"""
多策略检索模块
==============
封装旅游景点知识库的三类检索能力：

1. filter_search   —— 元数据过滤 + 语义检索（城市/免费/标签/票价区间）
2. hybrid_search   —— BM25 关键词 + 向量语义加权融合（RRF 或线性加权）
3. rerank_search    —— 混合检索后接入 BGE Reranker 精排

统一返回格式：
    [{"content": str, "metadata": dict, "score": float}, ...]

所有参数从 config.settings 读取，零硬编码。
依赖 modules.vector_store.ScenicVectorStore 提供底层向量库能力。

使用示例：
    from modules.retriever import ScenicRetriever

    retriever = ScenicRetriever()
    results = retriever.filter_search("故宫", city="北京", free_only=False)
    results = retriever.hybrid_search("杭州有哪些免费的5A景点")
    results = retriever.rerank_search("成都熊猫基地怎么去")
"""

import math
import threading
from typing import List, Optional, Dict, Any, Tuple

from langchain_core.documents import Document

from config.settings import settings
from modules.vector_store import ScenicVectorStore


# ============================================================
# 自定义异常类
# ============================================================
class RetrieverError(Exception):
    """检索模块基础异常。"""
    pass


class RetrieverInitError(RetrieverError):
    """检索器初始化失败异常 —— 向量库未就绪、依赖缺失等。"""
    pass


class RetrieverParamError(RetrieverError):
    """检索参数非法异常 —— 无效过滤条件、越界参数等。"""
    pass


class RetrieverSearchError(RetrieverError):
    """检索执行失败异常 —— BM25 构建失败、Reranker 推理出错等。"""
    pass


# ============================================================
# 多策略检索器
# ============================================================
class ScenicRetriever:
    """
    旅游景点多策略检索器
    --------------------
    整合向量语义检索、BM25 关键词检索、BGE 重排序，
    对外提供统一的过滤、混合、精排三种检索接口。

    架构：
        ScenicRetriever
        ├── ScenicVectorStore (向量库 + 语义检索)
        ├── BM25Okapi         (关键词检索，延迟构建)
        └── FlagReranker      (BGE 重排序，延迟加载)

    属性：
        vector_store : ScenicVectorStore  底层向量库实例
    """

    def __init__(
        self,
        collection_name: Optional[str] = None,
        persist_directory: Optional[str] = None,
    ) -> None:
        """
        初始化检索器。

        参数：
            collection_name:  Chroma 集合名称，默认从 settings 读取。
                             传入自定义值可指向测试/隔离集合。
            persist_directory: Chroma 持久化目录，默认从 settings 读取。
                               传入自定义值可指向测试/隔离目录。

        行为：
            - 创建 ScenicVectorStore 实例（复用已加载的 Embedding 模型）
            - BM25 索引与 Reranker 模型采用延迟加载，首次调用对应方法时才构建

        异常：
            RetrieverInitError: 向量库初始化失败
        """
        # ---- 向量库（核心依赖） ----
        try:
            self._vector_store: ScenicVectorStore = ScenicVectorStore(
                collection_name=collection_name,
                persist_directory=persist_directory,
            )
        except Exception as e:
            raise RetrieverInitError(
                f"检索器初始化失败：向量库无法加载。请检查 Chroma 持久化路径与权限。"
                f"原始错误: {e}"
            ) from e

        # ---- BM25 索引（延迟构建） ----
        self._bm25_index = None           # BM25Okapi 实例
        self._bm25_docs: List[Dict] = []  # 与 BM25 索引对应的文档列表
        self._bm25_lock = threading.Lock()

        # ---- Reranker（延迟加载） ----
        self._reranker = None
        self._reranker_lock = threading.Lock()

        # ---- 配置快照 ----
        self._semantic_top_k: int = settings.SEMANTIC_TOP_K
        self._bm25_top_k: int = settings.BM25_TOP_K
        self._hybrid_top_k: int = settings.HYBRID_TOP_K
        self._hybrid_alpha: float = settings.HYBRID_ALPHA
        self._score_threshold: float = settings.SIMILARITY_THRESHOLD
        self._rerank_enabled: bool = settings.RERANK_ENABLED
        self._rerank_model: str = settings.RERANK_MODEL_NAME
        self._rerank_top_k: int = settings.RERANK_TOP_K

        print(
            f"[ScenicRetriever] 检索器就绪: "
            f"collection={self._vector_store.collection_name}, "
            f"semantic_k={self._semantic_top_k}, "
            f"bm25_k={self._bm25_top_k}, hybrid_k={self._hybrid_top_k}, "
            f"alpha={self._hybrid_alpha}, rerank={self._rerank_enabled}"
        )

    # ============================================================
    # 1. 元数据过滤检索
    # ============================================================
    def filter_search(
        self,
        query: str = "",
        city: Optional[str] = None,
        free_only: bool = False,
        tags: Optional[List[str]] = None,
        ticket_min: Optional[float] = None,
        ticket_max: Optional[float] = None,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        元数据过滤 + 语义检索。

        先按元数据条件过滤候选集，再在过滤后的子集中做向量语义检索。
        支持多条件组合（城市、免费、票价区间、标签）。

        参数：
            query:           查询文本。为空时返回所有匹配过滤条件的文档（不排序）
            city:            城市名称精确匹配，例如 "北京"、"杭州"
            free_only:       仅免费景点（ticket=0）
            tags:            标签列表，文档需包含其中至少一个标签
                             例如 ["5A", "古迹", "世界文化遗产"]
            ticket_min:      最低票价（含）
            ticket_max:      最高票价（含）
            top_k:           返回文档数，默认从 settings 读取
            score_threshold: 相似度阈值（0.0~1.0），默认从 settings 读取

        返回：
            统一格式的检索结果列表，按相似度降序排列

        异常：
            RetrieverParamError:  过滤参数非法
            RetrieverSearchError: 检索执行失败

        示例：
            retriever = ScenicRetriever()
            # 北京免费5A景点
            results = retriever.filter_search(
                query="古迹", city="北京", free_only=True, tags=["5A"]
            )
            # 票价 50~100 元的景点
            results = retriever.filter_search(
                query="自然风光", ticket_min=50, ticket_max=100
            )
        """
        # ---- 参数校验 ----
        top_k = top_k or self._semantic_top_k
        score_threshold = score_threshold or self._score_threshold

        if top_k < 1:
            raise RetrieverParamError(f"top_k 必须 >= 1，实际: {top_k}")
        if not (0.0 <= score_threshold <= 1.0):
            raise RetrieverParamError(
                f"score_threshold 必须在 [0.0, 1.0] 之间，实际: {score_threshold}"
            )
        if ticket_min is not None and ticket_max is not None and ticket_min > ticket_max:
            raise RetrieverParamError(
                f"ticket_min ({ticket_min}) 不能大于 ticket_max ({ticket_max})"
            )

        # ---- 构建 Chroma where 过滤条件 ----
        # 对 city 值做 trim，兼容书写时带首尾空格的情况
        city_trimmed: Optional[str] = city.strip() if (city and city.strip()) else None

        where_clause = self._build_where_clause(
            city=city_trimmed,
            free_only=free_only,
            ticket_min=ticket_min,
            ticket_max=ticket_max,
        )

        # ---- 判断查询模式：空查询走纯元数据过滤，非空走语义检索 ----
        is_empty_query: bool = not query or not query.strip()

        # ---- 调试打印：过滤条件 & 向量库内 city 分布 ----
        mode_label: str = "纯元数据过滤（无向量化）" if is_empty_query else "语义+元数据过滤"
        print(
            f"[ScenicRetriever] filter_search [{mode_label}]: query='{query}', "
            f"where={where_clause}, top_k={top_k}, threshold={score_threshold}"
        )
        self._debug_show_collection_cities()

        # ================================================================
        # 分支 A：空查询 → 纯元数据过滤，不走 Embedding，规避空文本报错
        # ================================================================
        if is_empty_query:
            try:
                collection = self._vector_store.store._collection
                # 使用 Chroma 原生 collection.get 按 where 条件拉取全量文档
                chroma_result = collection.get(
                    where=where_clause,
                    include=["documents", "metadatas"],
                )
                raw_docs = chroma_result.get("documents") or []
                raw_metas = chroma_result.get("metadatas") or []
                print(
                    f"[ScenicRetriever] filter_search [{mode_label}]: "
                    f"Chroma collection.get 返回 {len(raw_docs)} 条"
                )
            except Exception as e:
                raise RetrieverSearchError(
                    f"纯元数据过滤查询失败。where={where_clause}。"
                    f"原始错误: {type(e).__name__}: {e}"
                ) from e

            # 组装结果，统一分数为 1.0（纯元数据命中，无语义匹配）
            results: List[Dict[str, Any]] = []
            for content, meta in zip(raw_docs, raw_metas):
                clean_meta: Dict[str, Any] = {}
                for k, v in (meta or {}).items():
                    if isinstance(v, str):
                        clean_meta[k] = v.strip()
                    else:
                        clean_meta[k] = v

                results.append({
                    "content": content,
                    "metadata": clean_meta,
                    "score": 1.0,  # 纯元数据命中，无向量相似度
                })

        # ================================================================
        # 分支 B：非空查询 → 完整语义检索 + where 过滤（原有正常逻辑）
        # ================================================================
        else:
            try:
                raw_results: List[Tuple[Document, float]] = (
                    self._vector_store.store.similarity_search_with_score(
                        query=query,
                        k=top_k,
                        filter=where_clause,
                    )
                )
                print(
                    f"[ScenicRetriever] filter_search [{mode_label}]: "
                    f"Chroma 返回 {len(raw_results)} 条原始结果"
                )
            except Exception as e:
                raise RetrieverSearchError(
                    f"语义+元数据过滤检索执行失败。query='{query}', where={where_clause}。"
                    f"原始错误: {type(e).__name__}: {e}"
                ) from e

            # ---- 转换距离为相似度分数 ----
            results = []
            for doc, distance in raw_results:
                similarity: float = 1.0 - (distance / 2.0)
                similarity = max(0.0, min(1.0, similarity))

                if similarity < score_threshold:
                    continue

                clean_meta = {}
                for k, v in (doc.metadata or {}).items():
                    if isinstance(v, str):
                        clean_meta[k] = v.strip()
                    else:
                        clean_meta[k] = v

                results.append({
                    "content": doc.page_content,
                    "metadata": clean_meta,
                    "score": round(similarity, 6),
                })

        # ================================================================
        # 后置处理（两个分支共用）：标签过滤、排序、截断、容错
        # ================================================================
        if tags:
            before_tag_filter = len(results)
            results = [
                r for r in results
                if self._has_any_tag(r.get("metadata", {}), tags)
            ]
            print(
                f"[ScenicRetriever] filter_search: "
                f"标签过滤 tags={tags}, {before_tag_filter} → {len(results)} 条"
            )

        # 按分数降序排列（空查询分支全为 1.0，排序无影响）
        results.sort(key=lambda r: r["score"], reverse=True)

        # 容错截断
        final_results = results[:top_k]

        if not final_results:
            print(
                f"[ScenicRetriever] filter_search: 最终无结果。"
                f"mode={mode_label}, where={where_clause}, tags={tags}, query='{query}'"
            )

        return final_results

    # ============================================================
    # 2. 混合检索（BM25 + 向量语义）
    # ============================================================
    def hybrid_search(
        self,
        query: str,
        top_k: Optional[int] = None,
        alpha: Optional[float] = None,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        BM25 关键词检索 + 向量语义检索加权融合。

        融合策略：
            1. 从向量库获取语义检索 top_k_semantic 个候选及分数
            2. 构建/复用 BM25 索引，获取 BM25 分数
            3. 对两个来源的文档取并集，计算综合分数：
               final_score = alpha * semantic_score + (1-alpha) * bm25_score
            4. 按 final_score 降序排列，返回 top_k 条

        参数：
            query:           查询文本（必填）
            top_k:           最终返回文档数，默认从 settings 读取
            alpha:           语义检索权重（0.0~1.0），默认从 settings.HYBRID_ALPHA 读取
                             alpha=1.0 纯语义，alpha=0.0 纯关键词
            score_threshold: 综合分数阈值，低于此值的文档被过滤

        返回：
            统一格式的检索结果列表

        异常：
            RetrieverParamError:  查询为空或参数越界
            RetrieverSearchError: BM25 索引构建失败或检索异常

        示例：
            retriever = ScenicRetriever()
            results = retriever.hybrid_search("杭州免费5A景点", top_k=5)
        """
        # ---- 参数校验 ----
        if not isinstance(query, str) or not query.strip():
            raise RetrieverParamError(
                f"hybrid_search 查询文本必须是非空字符串，实际: {query!r}"
            )

        top_k = top_k or self._hybrid_top_k
        alpha = alpha if alpha is not None else self._hybrid_alpha
        score_threshold = score_threshold or self._score_threshold

        if top_k < 1:
            raise RetrieverParamError(f"top_k 必须 >= 1，实际: {top_k}")
        if not (0.0 <= alpha <= 1.0):
            raise RetrieverParamError(f"alpha 必须在 [0.0, 1.0] 之间，实际: {alpha}")

        try:
            # ---- 1) 语义检索：多取一些候选，确保融合池充足 ----
            semantic_candidates: int = max(top_k * 3, self._semantic_top_k)
            semantic_results = self._vector_store.base_similarity_search(
                query=query,
                top_k=semantic_candidates,
                score_threshold=0.0,  # 先不过滤，融合后再统一阈值
            )
            # 转为 {content_hash: (content, metadata, score)} 字典
            semantic_map: Dict[str, Dict] = {}
            for r in semantic_results:
                key = self._doc_key(r["content"])
                semantic_map[key] = {
                    "content": r["content"],
                    "metadata": r["metadata"],
                    "semantic_score": r["score"],
                }

            # ---- 2) BM25 关键词检索 ----
            self._ensure_bm25_index()
            bm25_scores = self._compute_bm25_scores(query)

            # ---- 3) 加权融合 ----
            fused: Dict[str, Dict] = {}

            # 合并语义候选
            for key, entry in semantic_map.items():
                bm25_score = bm25_scores.get(key, 0.0)
                final_score = alpha * entry["semantic_score"] + (1.0 - alpha) * bm25_score
                fused[key] = {
                    "content": entry["content"],
                    "metadata": entry["metadata"],
                    "score": round(final_score, 6),
                }

            # 合并仅 BM25 命中的文档（语义检索未召回但关键词匹配的）
            for key, bm25_score in bm25_scores.items():
                if key not in fused and bm25_score > 0:
                    # 找到对应文档内容
                    matched_doc = self._find_doc_by_key(key)
                    if matched_doc:
                        final_score = (1.0 - alpha) * bm25_score
                        fused[key] = {
                            "content": matched_doc["content"],
                            "metadata": matched_doc["metadata"],
                            "score": round(final_score, 6),
                        }

            # ---- 4) 排序 + 阈值过滤 + 截断 ----
            sorted_results = sorted(
                fused.values(), key=lambda r: r["score"], reverse=True
            )
            filtered = [
                r for r in sorted_results if r["score"] >= score_threshold
            ]
            return filtered[:top_k]

        except (RetrieverParamError, RetrieverSearchError):
            raise
        except Exception as e:
            raise RetrieverSearchError(
                f"混合检索失败: {type(e).__name__}: {e}"
            ) from e

    # ============================================================
    # 3. 重排序检索
    # ============================================================
    def rerank_search(
        self,
        query: str,
        top_k: Optional[int] = None,
        alpha: Optional[float] = None,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        混合检索 + BGE Reranker 精排。

        流程：
            1. 执行 hybrid_search 获取初排候选池（候选数 = top_k * 3）
            2. 使用 BGE Reranker (CrossEncoder) 对每个 (query, document) 对打分
            3. 按 Reranker 分数降序排列，取 top_k 条
            4. 过滤低于阈值的低相关文档

        参数：
            query:           查询文本（必填）
            top_k:           最终返回文档数，默认从 settings.RERANK_TOP_K 读取
            alpha:           初排阶段的语义权重，默认从 settings 读取
            score_threshold: Reranker 归一化后分数阈值，默认 0.0

        返回：
            统一格式的检索结果列表，score 为 Reranker 归一化后的分数

        异常：
            RetrieverParamError:  查询为空
            RetrieverSearchError: Reranker 模型加载或推理失败

        示例：
            retriever = ScenicRetriever()
            results = retriever.rerank_search(
                "成都大熊猫基地门票多少钱", top_k=3
            )
        """
        # ---- 参数校验 ----
        if not isinstance(query, str) or not query.strip():
            raise RetrieverParamError(
                f"rerank_search 查询文本必须是非空字符串，实际: {query!r}"
            )

        top_k = top_k or self._rerank_top_k
        score_threshold = score_threshold or self._score_threshold

        if top_k < 1:
            raise RetrieverParamError(f"top_k 必须 >= 1，实际: {top_k}")

        # ---- 配置开关检查：RERANK_ENABLED=False 时跳过重排 ----
        if not self._rerank_enabled:
            print(
                f"[ScenicRetriever] rerank_search: RERANK_ENABLED=false, "
                f"跳过重排，直接返回混合检索结果"
            )
            return self.hybrid_search(
                query=query,
                top_k=top_k,
                alpha=alpha or self._hybrid_alpha,
                score_threshold=score_threshold,
            )

        # ---- transformers 版本兼容检查 ----
        # FlagEmbedding 低版本 + transformers>=4.36 会触发
        # AttributeError: XLMRobertaTokenizer has no attribute prepare_for_model
        if self._check_transformers_too_new():
            print(
                f"[ScenicRetriever] rerank_search: 检测到 transformers 版本过高, "
                f"自动降级为混合检索结果（Reranker 已禁用）"
            )
            return self.hybrid_search(
                query=query,
                top_k=top_k,
                alpha=alpha or self._hybrid_alpha,
                score_threshold=score_threshold,
            )

        try:
            # ---- 1) 初排：扩大候选池 ----
            candidate_pool_size: int = max(top_k * 3, 15)
            candidates = self.hybrid_search(
                query=query,
                top_k=candidate_pool_size,
                alpha=alpha or self._hybrid_alpha,
                score_threshold=0.0,  # 初排不过滤，交给 Reranker 判断
            )

            if not candidates:
                print(f"[ScenicRetriever] rerank_search: 初排无候选文档，直接返回")
                return []

            # ---- 2) 加载 Reranker 并精排 ----
            self._ensure_reranker()

            # 构建 (query, document) 对
            pairs: List[List[str]] = [
                [query, cand["content"]] for cand in candidates
            ]

            # Reranker 推理
            rerank_scores: List[float] = self._reranker.compute_score(
                pairs, normalize=True
            )
            # normalize=True 将分数归一化到 [0, 1]

            # 确保 rerank_scores 是列表格式
            if not isinstance(rerank_scores, list):
                rerank_scores = [rerank_scores]

            # ---- 3) 组装结果 ----
            for i, cand in enumerate(candidates):
                cand["score"] = round(float(rerank_scores[i]), 6)

            # ---- 4) 排序 + 阈值过滤 + 截断 ----
            candidates.sort(key=lambda r: r["score"], reverse=True)
            filtered = [
                r for r in candidates if r["score"] >= score_threshold
            ]
            return filtered[:top_k]

        except (RetrieverParamError, RetrieverSearchError):
            raise
        except ImportError as e:
            raise RetrieverSearchError(
                f"BGE Reranker 依赖缺失，请安装: "
                f"pip install FlagEmbedding。原始错误: {e}"
            ) from e
        except AttributeError as e:
            # transformers>=4.36 移除 prepare_for_model 导致该异常
            error_msg = str(e)
            print(
                f"[ScenicRetriever] rerank_search: 分词器兼容报错 "
                f"(transformers 版本过高), 降级返回混合检索结果。"
                f"原始错误: {error_msg}"
            )
            # 降级返回混合检索结果
            return self.hybrid_search(
                query=query,
                top_k=top_k,
                alpha=alpha or self._hybrid_alpha,
                score_threshold=score_threshold,
            )
        except RuntimeError as e:
            error_msg = str(e).lower()
            if "out of memory" in error_msg:
                raise RetrieverSearchError(
                    f"Reranker 推理时显存不足。建议: 使用 CPU 模式或减少候选池。"
                    f"原始错误: {e}"
                ) from e
            raise RetrieverSearchError(
                f"Reranker 推理失败: {e}"
            ) from e
        except Exception as e:
            raise RetrieverSearchError(
                f"重排序检索失败: {type(e).__name__}: {e}"
            ) from e

    # ============================================================
    # 内部：BM25 索引管理
    # ============================================================
    def _ensure_bm25_index(self) -> None:
        """
        确保 BM25 索引已构建（延迟构建，线程安全）。

        从向量库拉取全部文档，用 jieba 分词后构建 BM25Okapi 索引。
        若已构建则跳过，避免重复计算。
        """
        if self._bm25_index is not None:
            return

        with self._bm25_lock:
            # 双重检查
            if self._bm25_index is not None:
                return

            try:
                from rank_bm25 import BM25Okapi

                # 从向量库获取全部文档
                all_docs = self._get_all_documents()
                if not all_docs:
                    print("[ScenicRetriever] 向量库为空，BM25 索引跳过构建")
                    self._bm25_index = BM25Okapi([])
                    self._bm25_docs = []
                    return

                # 中文分词
                tokenized_corpus: List[List[str]] = [
                    self._tokenize(doc["content"]) for doc in all_docs
                ]

                self._bm25_index = BM25Okapi(tokenized_corpus)
                self._bm25_docs = all_docs
                print(
                    f"[ScenicRetriever] BM25 索引已构建: "
                    f"{len(all_docs)} 篇文档"
                )

            except ImportError as e:
                raise RetrieverSearchError(
                    f"BM25 依赖缺失，请安装: pip install rank-bm25。"
                    f"原始错误: {e}"
                ) from e
            except Exception as e:
                raise RetrieverSearchError(
                    f"BM25 索引构建失败: {type(e).__name__}: {e}"
                ) from e

    def _compute_bm25_scores(self, query: str) -> Dict[str, float]:
        """
        计算查询对所有文档的 BM25 分数。

        参数：
            query: 查询文本

        返回：
            {doc_key: normalized_bm25_score} 字典，分数已归一化到 [0, 1]
        """
        if self._bm25_index is None or len(self._bm25_docs) == 0:
            return {}

        tokenized_query: List[str] = self._tokenize(query)
        # 注意：rank_bm25.BM25Okapi.get_scores 返回 numpy.ndarray，非 list
        raw_scores = self._bm25_index.get_scores(tokenized_query)

        # numpy 数组判断：size==0 无文档，max()==0 所有分数为0（无匹配）
        if raw_scores.size == 0 or raw_scores.max() == 0:
            return {}

        # 归一化到 [0, 1]
        max_score: float = float(raw_scores.max())
        normalized: Dict[str, float] = {}
        for i, score in enumerate(raw_scores):
            if score > 0:
                key = self._doc_key(self._bm25_docs[i]["content"])
                normalized[key] = round(score / max_score, 6)

        return normalized

    # ============================================================
    # 内部：Reranker 管理
    # ============================================================
    @staticmethod
    def _check_transformers_too_new() -> bool:
        """
        检查 transformers 版本是否过高导致 FlagEmbedding 不兼容。

        transformers>=4.36 移除了 XLMRobertaTokenizer.prepare_for_model,
        而低版本 FlagEmbedding 仍依赖此方法, 触发 AttributeError。

        返回:
            True 表示版本过高, 应降级跳过 Reranker
        """
        try:
            import transformers
            ver = transformers.__version__
            major, minor = ver.split(".")[:2]
            if int(major) > 4 or (int(major) == 4 and int(minor) >= 36):
                print(
                    f"[ScenicRetriever] 检测到 transformers=={ver} (>=4.36), "
                    f"与 FlagEmbedding 存在 prepare_for_model 兼容问题, "
                    f"自动禁用 BGE Reranker"
                )
                return True
        except Exception:
            pass
        return False

    def _ensure_reranker(self) -> None:
        """
        确保 BGE Reranker 已加载（延迟加载，线程安全）。

        首次调用 rerank_search 时自动下载并加载模型。

        异常:
            RetrieverSearchError: 依赖缺失 / 版本不兼容 / 加载失败
        """
        if self._reranker is not None:
            return

        with self._reranker_lock:
            if self._reranker is not None:
                return

            try:
                from FlagEmbedding import FlagReranker

                self._reranker = FlagReranker(
                    self._rerank_model,
                    use_fp16=True,  # 半精度节省显存
                )
                print(
                    f"[ScenicRetriever] Reranker 已加载: {self._rerank_model}"
                )

            except ImportError as e:
                raise RetrieverSearchError(
                    f"BGE Reranker 依赖缺失，请安装: "
                    f"pip install FlagEmbedding。原始错误: {e}"
                ) from e
            except AttributeError as e:
                # transformers>=4.36 移除 prepare_for_model, 初始化时即触发
                raise RetrieverSearchError(
                    f"Reranker 加载失败: transformers 版本过高, "
                    f"FlagEmbedding 调用的 XLMRobertaTokenizer.prepare_for_model "
                    f"已被移除。建议: pip install transformers==4.35.2 或升级 FlagEmbedding。"
                    f"原始错误: {e}"
                ) from e
            except Exception as e:
                raise RetrieverSearchError(
                    f"Reranker 模型加载失败 ({self._rerank_model}): {e}"
                ) from e

    # ============================================================
    # 内部：调试辅助
    # ============================================================
    def _debug_show_collection_cities(self) -> None:
        """
        调试打印：输出向量库内所有文档的 metadata.city 值。

        用于排查 where 过滤不匹配问题。
        若向量库为空，打印提示。
        """
        try:
            collection = self._vector_store.store._collection
            result = collection.get(include=["metadatas"])
            metadatas = result.get("metadatas") or []
            if not metadatas:
                print("[ScenicRetriever] DEBUG: 向量库为空，无任何文档")
                return

            city_counts: Dict[str, int] = {}
            for meta in metadatas:
                if meta and "city" in meta:
                    raw_city = str(meta["city"])
                    city_counts[raw_city] = city_counts.get(raw_city, 0) + 1
            # 显示所有 city 值及其 repr，方便发现空格/空白差异
            city_info = ", ".join(
                f"{k!r}({v})" for k, v in sorted(city_counts.items())
            )
            print(
                f"[ScenicRetriever] DEBUG: 向量库 city 分布 "
                f"(共{len(metadatas)}条): {city_info}"
            )
        except Exception as e:
            print(f"[ScenicRetriever] DEBUG: 获取 city 分布失败: {e}")

    # ============================================================
    # 内部：辅助方法
    # ============================================================
    @staticmethod
    def _build_where_clause(
        city: Optional[str] = None,
        free_only: bool = False,
        ticket_min: Optional[float] = None,
        ticket_max: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        构建 Chroma 元数据过滤条件。

        参数：
            city:       城市名称精确匹配
            free_only:  仅免费景点
            ticket_min: 最低票价
            ticket_max: 最高票价

        返回：
            Chroma where 字典，无条件时返回 None
        """
        conditions: List[Dict[str, Any]] = []

        if city and city.strip():
            conditions.append({"city": city.strip()})

        if free_only:
            conditions.append({"ticket": 0})

        if ticket_min is not None:
            conditions.append({"ticket": {"$gte": ticket_min}})

        if ticket_max is not None:
            conditions.append({"ticket": {"$lte": ticket_max}})

        if not conditions:
            return None
        elif len(conditions) == 1:
            return conditions[0]
        else:
            return {"$and": conditions}

    @staticmethod
    def _has_any_tag(metadata: Dict[str, Any], tags: List[str]) -> bool:
        """
        判断文档元数据中是否包含至少一个指定标签。

        参数：
            metadata: 文档元数据字典
            tags:     目标标签列表

        返回：
            True 如果 doc_tags 与 target_tags 有交集
        """
        doc_tags: list = metadata.get("tags", [])
        if not doc_tags:
            return False
        return bool(set(doc_tags) & set(tags))

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """
        中文文本分词。

        优先使用 jieba 分词，不可用时回退到字符级 bigram。
        """
        try:
            import jieba
            return list(jieba.cut(text))
        except ImportError:
            # 回退：字符级 unigram + bigram 混合
            chars = list(text.replace(" ", ""))
            bigrams = [chars[i] + chars[i + 1] for i in range(len(chars) - 1)]
            return chars + bigrams

    @staticmethod
    def _doc_key(content: str) -> str:
        """基于文档内容生成唯一 key（用于去重和映射）。"""
        import hashlib
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    def _get_all_documents(self) -> List[Dict[str, Any]]:
        """
        从向量库获取全部文档（content + metadata）。

        返回：
            文档字典列表 [{"content": str, "metadata": dict}, ...]
        """
        try:
            collection = self._vector_store.store._collection
            result = collection.get(include=["documents", "metadatas"])
            docs: List[Dict[str, Any]] = []
            for content, meta in zip(
                result.get("documents", []),
                result.get("metadatas", []),
            ):
                docs.append({
                    "content": content,
                    "metadata": meta or {},
                })
            return docs
        except Exception as e:
            print(f"[ScenicRetriever] 获取全量文档失败: {e}")
            return []

    def _find_doc_by_key(self, key: str) -> Optional[Dict[str, Any]]:
        """
        在 BM25 文档列表中按 key 查找文档。

        参数：
            key: MD5 内容哈希

        返回：
            {"content": ..., "metadata": ...} 或 None
        """
        for doc in self._bm25_docs:
            if self._doc_key(doc["content"]) == key:
                return doc
        return None

    # ============================================================
    # 辅助属性
    # ============================================================
    @property
    def vector_store(self) -> ScenicVectorStore:
        """返回底层向量库实例。"""
        return self._vector_store

    @property
    def bm25_ready(self) -> bool:
        """检查 BM25 索引是否已构建。"""
        return self._bm25_index is not None

    @property
    def reranker_ready(self) -> bool:
        """检查 Reranker 是否已加载。"""
        return self._reranker is not None
