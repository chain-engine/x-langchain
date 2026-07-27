# -*- coding: utf-8 -*-
"""
语义记忆 - Agent 的长期语义记忆组件

基于向量检索的 Agent 记忆系统：
- 将对话/文档内容持久化为语义向量
- 通过自然语言查询检索相关记忆
- 支持与 LCAgent 的无缝集成

与 ChatMessageHistory（对话历史）的区别：
- ChatMessageHistory: 保留完整的对话消息（消息级别）
- SemanticMemory: 保留语义片段（向量级别，支持相似度检索）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from core.logger import logger


# =============================================================================
# 配置
# =============================================================================


@dataclass
class SemanticMemoryConfig:
    """语义记忆配置"""

    # 向量存储
    vectorstore_provider: str = "memory"  # chroma / faiss / memory
    persist_directory: Optional[str] = None
    collection_name: str = "semantic_memory"

    # 检索
    search_k: int = 3
    search_score_threshold: Optional[float] = None  # None = 不过滤

    # 分块
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Embedding
    embedding_provider: str = "mock"
    embedding_model: Optional[str] = None


# =============================================================================
# 语义记忆
# =============================================================================


class SemanticMemory:
    """
    Agent 语义记忆

    提供基于向量检索的长期记忆能力：
    - add: 添加记忆（自动分块 + 向量化）
    - search: 自然语言检索相关记忆
    - delete: 删除记忆
    - clear: 清空所有记忆

    使用方式：
        ```python
        from retrieval import SemanticMemory, EmbeddingFactory
        from retrieval.vectorstore import VectorStoreFactory

        # 创建语义记忆（使用内存向量存储）
        config = SemanticMemoryConfig(
            vectorstore_provider="memory",
            embedding_provider="mock",
        )
        memory = SemanticMemory(config)

        # 添加记忆
        memory.add("用户的项目配置存放在 config.yaml 中")
        memory.add("用户偏好使用 DeepSeek 模型")

        # 检索相关记忆
        results = memory.search("用户用什么模型？", k=2)
        for r in results:
            print(r.page_content)

        # 清空
        memory.clear()
        ```
    """

    def __init__(
        self,
        config: SemanticMemoryConfig,
        embedding: Optional[Any] = None,
        vectorstore: Optional[Any] = None,
    ):
        """
        初始化语义记忆

        Args:
            config: SemanticMemoryConfig 配置
            embedding: 嵌入模型（None 则自动创建）
            vectorstore: 向量存储（None 则自动创建）
        """
        self._config = config

        # 初始化 embedding
        if embedding is None:
            from retrieval import EmbeddingFactory

            embed_config = {"provider": config.embedding_provider}
            if config.embedding_model:
                embed_config["model"] = config.embedding_model
            self._embedding = EmbeddingFactory.create(embed_config)
        else:
            self._embedding = embedding

        # 初始化向量存储
        if vectorstore is None:
            from retrieval import VectorStoreFactory

            self._vectorstore = VectorStoreFactory.create(
                {
                    "provider": config.vectorstore_provider,
                    "persist_directory": config.persist_directory,
                    "collection_name": config.collection_name,
                },
                embedding=self._embedding,
            )
        else:
            self._vectorstore = vectorstore

        # 初始化分块器
        from retrieval import create_text_splitter

        self._splitter = create_text_splitter(
            splitter_type="recursive",
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )

        self._added_count = 0
        logger.info(
            f"语义记忆初始化完成: provider={config.vectorstore_provider}, "
            f"embedding={config.embedding_provider}"
        )

    @property
    def embedding(self) -> Any:
        """获取嵌入模型"""
        return self._embedding

    @property
    def vectorstore(self) -> Any:
        """获取向量存储"""
        return self._vectorstore

    @property
    def document_count(self) -> int:
        """当前存储的文档数量"""
        if hasattr(self._vectorstore, "collection_count"):
            return self._vectorstore.collection_count
        return self._added_count

    # ------------------------------------------------------------------ #
    # 核心操作
    # ------------------------------------------------------------------ #

    def add(
        self,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
        doc_id: Optional[str] = None,
    ) -> list[str]:
        """
        添加记忆到向量存储

        Args:
            content: 记忆内容（长文本会自动分块）
            metadata: 元数据（来源、时间戳等）
            doc_id: 可选的文档 ID

        Returns:
            存储的文档 ID 列表
        """
        if not content or not content.strip():
            return []

        meta = metadata or {}
        meta.setdefault("added_at", self._now())
        meta.setdefault("content_hash", str(hash(content)))

        # 分块
        chunks = self._splitter.split_text(content)
        if not chunks:
            return []

        # 构建元数据
        chunk_metas = []
        for i, chunk in enumerate(chunks):
            chunk_meta = {
                **meta,
                "chunk_index": i,
                "chunk_count": len(chunks),
            }
            chunk_metas.append(chunk_meta)

        # 存储
        ids = self._vectorstore.add_texts(
            texts=chunks,
            metadatas=chunk_metas,
            ids=[f"{doc_id or self._next_id()}_{i}" for i in range(len(chunks))] if doc_id else None,
        )

        self._added_count += len(chunks)
        logger.debug(f"添加记忆: {len(chunks)} 个块, content={content[:50]}...")
        return ids

    def add_documents(
        self,
        documents: list[Any],
        metadata: Optional[dict[str, Any]] = None,
    ) -> list[str]:
        """
        批量添加 Document 对象

        Args:
            documents: Document 列表
            metadata: 附加元数据

        Returns:
            文档 ID 列表
        """
        all_ids = []
        for doc in documents:
            content = getattr(doc, "page_content", str(doc))
            doc_meta = {**(getattr(doc, "metadata", {}) or {}), **(metadata or {})}
            ids = self.add(content, doc_meta)
            all_ids.extend(ids)
        return all_ids

    def search(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[dict[str, Any]] = None,
    ) -> list[Any]:
        """
        检索相关记忆

        Args:
            query: 自然语言查询
            k: 返回数量（None 使用配置值）
            filter: 元数据过滤条件

        Returns:
            相关文档列表（按相关性排序）
        """
        k = k or self._config.search_k
        threshold = self._config.search_score_threshold

        try:
            if threshold is not None:
                results = self._vectorstore.similarity_search_with_score(
                    query, k=k, filter=filter
                )
                # 过滤低分结果
                results = [
                    (doc, score)
                    for doc, score in results
                    if score >= threshold
                ]
                return [doc for doc, _ in results]
            else:
                return self._vectorstore.similarity_search(
                    query, k=k, filter=filter
                )
        except Exception as e:
            logger.error(f"记忆检索失败: {e}")
            return []

    def search_with_scores(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[dict[str, Any]] = None,
    ) -> list[tuple[Any, float]]:
        """
        检索相关记忆（带分数）

        Args:
            query: 自然语言查询
            k: 返回数量
            filter: 元数据过滤

        Returns:
            (文档, 相似度分数) 列表
        """
        k = k or self._config.search_k
        try:
            return self._vectorstore.similarity_search_with_score(query, k=k, filter=filter)
        except Exception as e:
            logger.error(f"记忆检索失败: {e}")
            return []

    def delete(self, ids: list[str]) -> None:
        """
        删除指定 ID 的记忆

        Args:
            ids: 文档 ID 列表
        """
        try:
            self._vectorstore.delete(ids)
            logger.info(f"删除记忆: {len(ids)} 个文档")
        except Exception as e:
            logger.error(f"删除记忆失败: {e}")

    def clear(self) -> None:
        """清空所有记忆"""
        try:
            # Chroma: 删除集合
            if hasattr(self._vectorstore, "_collection"):
                self._vectorstore.delete(where={})
            logger.info("语义记忆已清空")
        except Exception as e:
            logger.warning(f"清空记忆时出现问题: {e}")

    # ------------------------------------------------------------------ #
    # 工具方法
    # ------------------------------------------------------------------ #

    def get_context(self, query: str, k: Optional[int] = None) -> str:
        """
        获取检索到的记忆内容拼接为上下文字符串

        适用于直接注入 LLM prompt。

        Args:
            query: 查询
            k: 数量

        Returns:
            格式化的上下文字符串
        """
        docs = self.search(query, k=k)
        if not docs:
            return ""

        parts = []
        for i, doc in enumerate(docs, 1):
            content = getattr(doc, "page_content", str(doc))
            parts.append(f"[记忆{i}]: {content}")

        return "\n\n".join(parts)

    def persist(self) -> None:
        """持久化向量存储（仅本地存储有效）"""
        if hasattr(self._vectorstore, "persist"):
            self._vectorstore.persist()
            logger.info("语义记忆已持久化")

    # ------------------------------------------------------------------ #
    # 内部方法
    # ------------------------------------------------------------------ #

    def _next_id(self) -> str:
        """生成唯一 ID"""
        import uuid

        return str(uuid.uuid4())[:8]

    def _now(self) -> str:
        """获取当前时间戳"""
        from datetime import datetime

        return datetime.now().isoformat()

    def __len__(self) -> int:
        """记忆数量"""
        return self.document_count

    def __repr__(self) -> str:
        return f"<SemanticMemory: docs={self.document_count}, embedding={self._config.embedding_provider}>"


__all__ = ["SemanticMemory", "SemanticMemoryConfig"]
