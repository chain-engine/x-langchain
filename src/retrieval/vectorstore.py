# -*- coding: utf-8 -*-
"""
向量存储抽象

提供统一的向量存储接口，支持多种后端：
- Chroma（本地 / 客户端模式）
- FAISS（本地）
- Qdrant（云端 / 自部署）
- LanceDB（本地）
- InMemory（内存，测试用）
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from langchain_core.documents import Document as LCDocument
from langchain_core.vectorstores import VectorStore as LCVectorStore
from pydantic import BaseModel, Field

from core.logger import logger


# =============================================================================
# 配置
# =============================================================================


class VectorStoreConfig(BaseModel):
    """向量存储配置"""

    provider: str = Field(default="chroma", description="提供者：chroma / faiss / qdrant / lancedb / memory")
    persist_directory: Optional[str] = Field(default=None, description="持久化目录")
    collection_name: str = Field(default="default", description="集合名称")
    # Chroma 专用
    host: Optional[str] = None
    port: Optional[int] = None
    # Qdrant 专用
    qdrant_url: Optional[str] = None
    qdrant_port: int = 6333
    qdrant_grpc_port: Optional[int] = None
    qdrant_https: bool = False
    # LanceDB 专用
    lancedb_db_path: Optional[str] = None


# =============================================================================
# 抽象接口
# =============================================================================


class BaseVectorStore(ABC):
    """
    向量存储抽象基类

    所有向量存储实现需继承此类并实现核心接口。
    """

    @abstractmethod
    def add_texts(
        self,
        texts: list[str],
        metadatas: Optional[list[dict[str, Any]]] = None,
        ids: Optional[list[str]] = None,
    ) -> list[str]:
        """
        添加文本到向量存储

        Args:
            texts: 文本列表
            metadatas: 元数据列表（与 texts 一一对应）
            ids: 可选的 ID 列表

        Returns:
            文档 ID 列表
        """
        raise NotImplementedError

    @abstractmethod
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> list[Any]:
        """
        相似度检索

        Args:
            query: 查询文本
            k: 返回结果数量
            filter: 元数据过滤条件

        Returns:
            文档列表（langchain Document）
        """
        raise NotImplementedError

    @abstractmethod
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict[str, Any]] = None,
    ) -> list[tuple[Any, float]]:
        """
        相似度检索（带分数）

        Returns:
            (文档, 距离分数) 列表
        """
        raise NotImplementedError

    @abstractmethod
    def delete(self, ids: Optional[list[str]] = None, **kwargs: Any) -> None:
        """删除文档"""
        raise NotImplementedError

    def as_retriever(self, **kwargs: Any) -> Any:
        """转换为 Retriever"""
        from langchain_core.retrievers import VectorStoreRetriever

        return VectorStoreRetriever(vectorstore=self, **kwargs)


# =============================================================================
# Chroma 向量存储
# =============================================================================


class ChromaVectorStore(BaseVectorStore):
    """
    Chroma 向量存储

    支持本地持久化和客户端模式。
    需要安装：pip install chromadb
    """

    def __init__(
        self,
        embedding: Any,
        persist_directory: Optional[str] = None,
        collection_name: str = "default",
        host: Optional[str] = None,
        port: Optional[int] = None,
    ):
        try:
            from langchain_community.vectorstores import Chroma
        except ImportError:
            raise ImportError("请安装 chromadb: pip install chromadb")

        self._embedding = embedding

        if host and port:
            client = Chroma(
                client_type="http",
                host=host,
                port=port,
            )
            self._impl = Chroma(
                client=client,
                embedding_function=embedding,
                collection_name=collection_name,
            )
        elif persist_directory:
            self._impl = Chroma(
                embedding_function=embedding,
                persist_directory=persist_directory,
                collection_name=collection_name,
            )
        else:
            self._impl = Chroma(
                embedding_function=embedding,
                collection_name=collection_name,
            )

        self._collection_name = collection_name
        logger.info(f"Chroma 向量存储已初始化: collection={collection_name}")

    def add_texts(
        self,
        texts: list[str],
        metadatas: Optional[list[dict[str, Any]]] = None,
        ids: Optional[list[str]] = None,
    ) -> list[str]:
        return self._impl.add_texts(texts, metadatas=metadatas, ids=ids)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> list[Any]:
        return self._impl.similarity_search(query, k=k, filter=filter, **kwargs)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict[str, Any]] = None,
    ) -> list[tuple[Any, float]]:
        return self._impl.similarity_search_with_score(query, k=k, filter=filter)

    def delete(self, ids: Optional[list[str]] = None, **kwargs: Any) -> None:
        self._impl.delete(ids=ids, **kwargs)

    def persist(self) -> None:
        """持久化到磁盘（仅本地模式有效）"""
        if hasattr(self._impl, "_persist_directory"):
            self._impl.persist()
            logger.info("Chroma 已持久化")

    @property
    def collection_count(self) -> int:
        """集合中的文档数量"""
        return self._impl._collection.count()


# =============================================================================
# FAISS 向量存储
# =============================================================================


class FAISSVectorStore(BaseVectorStore):
    """
    FAISS 向量存储

    适用于本地小规模向量检索。
    需要安装：pip install faiss-cpu（或 faiss-gpu）
    """

    def __init__(
        self,
        embedding: Any,
        persist_directory: Optional[str] = None,
        index_name: str = "index",
    ):
        from langchain_community.vectorstores import FAISS

        self._embedding = embedding
        self._persist_directory = persist_directory
        self._index_name = index_name
        self._impl: Optional[Any] = None

        if persist_directory:
            try:
                self._impl = FAISS.load_local(
                    persist_directory,
                    embedding,
                    index_name=index_name,
                )
                logger.info(f"FAISS 从 {persist_directory} 加载")
            except FileNotFoundError:
                logger.info(f"FAISS 持久化文件不存在，将创建新索引")
                self._impl = None

    def add_texts(
        self,
        texts: list[str],
        metadatas: Optional[list[dict[str, Any]]] = None,
        ids: Optional[list[str]] = None,
    ) -> list[str]:
        if self._impl is None:
            from langchain_community.vectorstores import FAISS

            self._impl = FAISS.from_texts(
                texts,
                self._embedding,
                metadatas=metadatas,
                ids=ids,
            )
        else:
            self._impl.add_texts(texts, metadatas=metadatas, ids=ids)
        return [str(i) for i in range(len(texts))]

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> list[Any]:
        if self._impl is None:
            return []
        return self._impl.similarity_search(query, k=k, filter=filter, **kwargs)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict[str, Any]] = None,
    ) -> list[tuple[Any, float]]:
        if self._impl is None:
            return []
        return self._impl.similarity_search_with_score(query, k=k, filter=filter)

    def delete(self, ids: Optional[list[str]] = None, **kwargs: Any) -> None:
        if self._impl:
            self._impl.delete(ids=ids, **kwargs)

    def save_local(self, directory: Optional[str] = None) -> None:
        """持久化到磁盘"""
        if self._impl:
            path = directory or self._persist_directory
            if path:
                self._impl.save_local(path, index_name=self._index_name)
                logger.info(f"FAISS 已保存到 {path}")


# =============================================================================
# 内存向量存储（测试用）
# =============================================================================


class InMemoryVectorStore(BaseVectorStore):
    """
    内存向量存储

    适用于测试和小规模数据。
    """

    def __init__(self, embedding: Any):
        from langchain_community.vectorstores import InMemoryVectorStore as _LCInMem

        self._impl = _LCInMem(embedding=embedding)
        self._embedding = embedding

    def add_texts(
        self,
        texts: list[str],
        metadatas: Optional[list[dict[str, Any]]] = None,
        ids: Optional[list[str]] = None,
    ) -> list[str]:
        return self._impl.add_texts(texts, metadatas=metadatas, ids=ids)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> list[Any]:
        return self._impl.similarity_search(query, k=k, filter=filter, **kwargs)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict[str, Any]] = None,
    ) -> list[tuple[Any, float]]:
        return self._impl.similarity_search_with_score(query, k=k, filter=filter)

    def delete(self, ids: Optional[list[str]] = None, **kwargs: Any) -> None:
        self._impl.delete(ids=ids, **kwargs)


# =============================================================================
# 工厂函数
# =============================================================================


class VectorStoreFactory:
    """
    向量存储工厂类

    根据配置创建合适的向量存储实例。

    使用方式：
        ```python
        config = VectorStoreConfig(provider="chroma", persist_directory="./chroma_db")
        store = VectorStoreFactory.create(config, embedding=embed)
        ```
    """

    _REGISTRY: dict[str, type[BaseVectorStore]] = {
        "chroma": ChromaVectorStore,
        "faiss": FAISSVectorStore,
        "memory": InMemoryVectorStore,
    }

    @classmethod
    def create(
        cls,
        config: VectorStoreConfig | dict[str, Any],
        embedding: Any,
    ) -> BaseVectorStore:
        """
        根据配置创建向量存储

        Args:
            config: VectorStoreConfig 实例或 dict
            embedding: 嵌入模型实例

        Returns:
            BaseVectorStore 实例
        """
        if isinstance(config, dict):
            config = VectorStoreConfig(**config)

        provider = config.provider.lower()
        if provider not in cls._REGISTRY:
            available = ", ".join(cls._REGISTRY.keys())
            raise ValueError(
                f"不支持的 VectorStore 提供者: {provider}，支持的: {available}"
            )

        impl_cls = cls._REGISTRY[provider]
        kwargs: dict[str, Any] = {"embedding": embedding}

        if config.persist_directory:
            kwargs["persist_directory"] = config.persist_directory
        if config.collection_name and provider != "memory":
            kwargs["collection_name"] = config.collection_name
        if config.host:
            kwargs["host"] = config.host
        if config.port:
            kwargs["port"] = config.port

        logger.info(f"创建 VectorStore: provider={config.provider}")
        return impl_cls(**kwargs)

    @classmethod
    def register(cls, name: str, impl_cls: type[BaseVectorStore]) -> None:
        """注册自定义向量存储"""
        cls._REGISTRY[name.lower()] = impl_cls


__all__ = [
    "VectorStoreConfig",
    "BaseVectorStore",
    "ChromaVectorStore",
    "FAISSVectorStore",
    "InMemoryVectorStore",
    "VectorStoreFactory",
]
