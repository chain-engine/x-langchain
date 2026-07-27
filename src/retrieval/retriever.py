# -*- coding: utf-8 -*-
"""
检索器抽象

提供统一的检索器接口和多种实现：
- VectorRetriever: 基于向量相似度的检索器
- EnsembleRetriever: 混合检索器（向量 + 关键词）
- MultiQueryRetriever: 多查询检索器
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document as LCDocument

from core.logger import logger


# =============================================================================
# 抽象接口
# =============================================================================


class Retriever(ABC):
    """
    检索器抽象基类

    所有检索器实现需继承此类。
    """

    @abstractmethod
    def get_relevant_documents(self, query: str, **kwargs: Any) -> list[Any]:
        """
        获取与查询相关的文档

        Args:
            query: 查询文本
            **kwargs: 其他参数

        Returns:
            相关文档列表
        """
        raise NotImplementedError

    def invoke(self, query: str, config: Optional[Any] = None) -> list[Any]:
        """Runnable 接口入口"""
        return self.get_relevant_documents(query)


# =============================================================================
# 向量检索器
# =============================================================================


class VectorRetriever(BaseRetriever):
    """
    基于向量存储的检索器

    包装 VectorStore，提供标准 Retriever 接口。
    支持元数据过滤和相似度阈值。
    """

    def __init__(
        self,
        vectorstore: Any,
        search_type: str = "similarity",
        k: int = 4,
        score_threshold: Optional[float] = None,
        filter: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ):
        """
        初始化向量检索器

        Args:
            vectorstore: 向量存储实例
            search_type: 搜索类型（similarity / similarity_score_threshold / mmr）
            k: 返回结果数量
            score_threshold: 相似度阈值（仅 similarity_score_threshold 模式）
            filter: 元数据过滤条件
            **kwargs: 透传给底层搜索
        """
        super().__init__()
        self._vectorstore = vectorstore
        self._search_type = search_type
        self._k = k
        self._score_threshold = score_threshold
        self._filter = filter
        self._kwargs = kwargs

    def _get_relevant_documents(self, query: str) -> list[LCDocument]:
        """langchain BaseRetriever 要求的内部方法"""
        if self._search_type == "similarity":
            return self._vectorstore.similarity_search(
                query, k=self._k, filter=self._filter, **self._kwargs
            )
        elif self._search_type == "similarity_score_threshold":
            docs_with_scores = self._vectorstore.similarity_search_with_score(
                query, k=self._k, filter=self._filter
            )
            if self._score_threshold is not None:
                docs_with_scores = [
                    (doc, score)
                    for doc, score in docs_with_scores
                    if score >= self._score_threshold
                ]
            return [doc for doc, _ in docs_with_scores]
        elif self._search_type == "mmr":
            # Max Marginal Relevance
            try:
                return self._vectorstore.max_marginal_relevance_search(
                    query, k=self._k, fetch_k=self._kwargs.get("fetch_k", self._k * 4), filter=self._filter
                )
            except AttributeError:
                logger.warning("当前 VectorStore 不支持 MMR，退化为相似度检索")
                return self._vectorstore.similarity_search(query, k=self._k, filter=self._filter)
        else:
            return self._vectorstore.similarity_search(query, k=self._k, filter=self._filter)

    @property
    def k(self) -> int:
        return self._k

    def update_k(self, k: int) -> None:
        """动态更新 k 值"""
        self._k = k


# =============================================================================
# 混合检索器
# =============================================================================


class EnsembleRetriever(BaseRetriever):
    """
    混合检索器

    组合多个检索器的结果，通过 Reciprocal Rank Fusion 排序。
    适用于需要同时考虑语义相似度和关键词匹配的场景。
    """

    def __init__(
        self,
        retrievers: list[Any],
        weights: Optional[list[float]] = None,
        c: int = 60,
    ):
        """
        初始化混合检索器

        Args:
            retrievers: 检索器列表
            weights: 各检索器权重（与 retrievers 等长），None 则等权
            c: Reciprocal Rank Fusion 参数，越大越倾向于多样结果
        """
        super().__init__()
        self._retrievers = retrievers
        self._weights = weights or [1.0] * len(retrievers)
        self._c = c

        if len(self._retrievers) != len(self._weights):
            raise ValueError("retrievers 和 weights 长度必须一致")

    def _get_relevant_documents(self, query: str) -> list[LCDocument]:
        """执行混合检索"""
        # 各检索器独立检索
        all_results: dict[str, tuple[Any, float]] = {}

        for retriever, weight in zip(self._retrievers, self._weights):
            try:
                docs = retriever.get_relevant_documents(query)
                for rank, doc in enumerate(docs, 1):
                    doc_id = getattr(doc, "id", None) or str(hash(doc.page_content))
                    # Reciprocal Rank Fusion 分数
                    score = weight / (self._c + rank)
                    if doc_id in all_results:
                        all_results[doc_id] = (doc, all_results[doc_id][1] + score)
                    else:
                        all_results[doc_id] = (doc, score)
            except Exception as e:
                logger.warning(f"检索器 {type(retriever).__name__} 失败: {e}")

        # 按融合分数排序
        sorted_docs = sorted(all_results.values(), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in sorted_docs]


# =============================================================================
# 多查询检索器
# =============================================================================


class MultiQueryRetriever(BaseRetriever):
    """
    多查询检索器

    使用 LLM 从用户查询生成多个不同视角的查询，
    再从各查询的结果中综合选取最相关的文档。
    适用于复杂查询或检索结果不理想时。
    """

    def __init__(
        self,
        retriever: Any,
        llm: Any,
        num_queries: int = 3,
        k: int = 4,
    ):
        """
        初始化多查询检索器

        Args:
            retriever: 基础检索器
            llm: 用于生成查询的 LLM
            num_queries: 生成的查询数量
            k: 每个查询返回的文档数
        """
        super().__init__()
        self._retriever = retriever
        self._llm = llm
        self._num_queries = num_queries
        self._k = k

    def _get_relevant_documents(self, query: str) -> list[LCDocument]:
        """生成多查询并综合结果"""
        # 生成多个查询
        queries = self._generate_queries(query)

        # 各查询独立检索
        all_docs: dict[str, LCDocument] = {}
        for q in queries:
            try:
                docs = self._retriever.get_relevant_documents(q)
                for doc in docs:
                    doc_id = getattr(doc, "id", None) or str(hash(doc.page_content))
                    if doc_id not in all_docs:
                        all_docs[doc_id] = doc
            except Exception as e:
                logger.warning(f"检索失败: {e}")

        # 返回综合结果（保留去重后的所有文档）
        return list(all_docs.values())

    def _generate_queries(self, query: str) -> list[str]:
        """使用 LLM 生成多个查询"""
        prompt = f"""Given a user query, generate {self._num_queries} different versions 
of the query to retrieve relevant documents. Return each version on a new line.

Original query: {query}

Rewritten queries (one per line):"""

        try:
            response = self._llm.invoke([{"role": "user", "content": prompt}])
            content = getattr(response, "content", str(response))
            queries = [
                line.strip()
                for line in content.split("\n")
                if line.strip() and not line.strip().startswith("#")
            ]
            return queries[: self._num_queries]
        except Exception as e:
            logger.warning(f"查询生成失败: {e}")
            return [query]


__all__ = [
    "Retriever",
    "VectorRetriever",
    "EnsembleRetriever",
    "MultiQueryRetriever",
]
