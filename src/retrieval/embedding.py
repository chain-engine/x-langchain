# -*- coding: utf-8 -*-
"""
Embedding 嵌入模型抽象

提供统一的嵌入模型接口，支持多种后端：
- OpenAI (text-embedding-3-small / text-embedding-3-large)
- DashScope (text-embedding-v1 / text-embedding-v2)
- 本地模型（SentenceTransformers）
- Mock（测试用）
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, Field

from core.logger import logger


# =============================================================================
# 配置
# =============================================================================


class EmbeddingConfig(BaseModel):
    """Embedding 配置"""

    provider: str = Field(default="openai", description="提供者：openai / dashscope / local / mock")
    model: Optional[str] = Field(default=None, description="模型名称")
    api_key: Optional[str] = Field(default=None, description="API Key")
    api_base: Optional[str] = Field(default=None, description="API Base URL")
    dimensions: Optional[int] = Field(default=None, description="向量维度（仅部分模型支持）")
    batch_size: Optional[int] = Field(default=None, description="批量编码大小")
    timeout: Optional[int] = Field(default=None, description="超时时间（秒）")


# =============================================================================
# 抽象接口
# =============================================================================


class BaseEmbedding(ABC):
    """
    嵌入模型抽象基类

    所有嵌入模型实现需继承此类并实现：
    - embed_query: 单条文本嵌入
    - embed_documents: 批量文本嵌入
    """

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """
        单条查询文本嵌入

        Args:
            text: 查询文本

        Returns:
            嵌入向量（list[float]）
        """
        raise NotImplementedError

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        批量文档嵌入

        Args:
            texts: 文档文本列表

        Returns:
            嵌入向量列表
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """向量维度"""
        raise NotImplementedError

    def __len__(self) -> int:
        """向量维度（兼容 len()）"""
        return self.dimensions


# =============================================================================
# OpenAI Embedding
# =============================================================================


class OpenAIEmbedding(BaseEmbedding, Embeddings):
    """
    OpenAI 兼容的嵌入模型

    支持所有使用 OpenAI API 格式的嵌入服务，包括：
    - OpenAI 官方 API
    - DeepSeek Embedding
    - 通义千问 Embedding
    - 其他 OpenAI 兼容后端
    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        api_base: Optional[str] = None,
        dimensions: Optional[int] = None,
        batch_size: int = 100,
        timeout: int = 60,
    ):
        from langchain_openai import OpenAIEmbeddings

        self._impl = OpenAIEmbeddings(
            api_key=api_key,
            model=model,
            base_url=api_base,
            embedding_ctx_length=8191,
            max_retries=3,
            timeout=timeout,
        )
        self._dimensions = dimensions
        self._batch_size = batch_size

    def embed_query(self, text: str) -> list[float]:
        return self._impl.embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # 批量处理，避免超限
        results = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            results.extend(self._impl.embed_documents(batch))
        return results

    @property
    def dimensions(self) -> int:
        if self._dimensions:
            return self._dimensions
        return {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }.get(self._impl.model, 1536)


# =============================================================================
# DashScope Embedding
# =============================================================================


class DashScopeEmbedding(BaseEmbedding, Embeddings):
    """
    阿里云 DashScope 嵌入模型

    支持 text-embedding-v1 / text-embedding-v2
    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-v2",
        dimensions: Optional[int] = None,
        batch_size: int = 25,
        timeout: int = 60,
    ):
        from langchain_community.embeddings import DashScopeEmbeddings

        self._impl = DashScopeEmbeddings(
            dashscope_api_key=api_key,
            model=model,
        )
        self._dimensions = dimensions
        self._batch_size = batch_size

    def embed_query(self, text: str) -> list[float]:
        return self._impl.embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        results = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            results.extend(self._impl.embed_documents(batch))
        return results

    @property
    def dimensions(self) -> int:
        if self._dimensions:
            return self._dimensions
        return {"text-embedding-v1": 1536, "text-embedding-v2": 1536}.get(
            getattr(self._impl, "model", ""), 1536
        )


# =============================================================================
# 本地模型（SentenceTransformers）
# =============================================================================


class LocalEmbedding(BaseEmbedding, Embeddings):
    """
    本地 SentenceTransformers 嵌入模型

    适用于离线或对隐私有要求的场景。
    需要安装：pip install sentence-transformers
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        encode_kwargs: Optional[dict] = None,
    ):
        try:
            from langchain_community.embeddings import SentenceTransformerEmbeddings

            self._impl = SentenceTransformerEmbeddings(
                model_name=model_name,
                device=device,
                encode_kwargs=encode_kwargs or {},
            )
        except ImportError:
            raise ImportError(
                "请安装 sentence-transformers: pip install sentence-transformers"
            )

    def embed_query(self, text: str) -> list[float]:
        return self._impl.embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._impl.embed_documents(texts)

    @property
    def dimensions(self) -> int:
        if hasattr(self._impl, "embedding_dimension"):
            return self._impl.embedding_dimension or 384
        return 384


# =============================================================================
# Mock Embedding（测试用）
# =============================================================================


class MockEmbedding(BaseEmbedding, Embeddings):
    """
    模拟嵌入模型（用于测试）

    所有输入返回固定维度的随机向量。
    """

    def __init__(self, dimensions: int = 384, seed: int = 42):
        import random

        self._dimensions = dimensions
        self._rng = random.Random(seed)

    def embed_query(self, text: str) -> list[float]:
        return [self._rng.uniform(-1, 1) for _ in range(self._dimensions)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]

    @property
    def dimensions(self) -> int:
        return self._dimensions


# =============================================================================
# 工厂函数
# =============================================================================


class EmbeddingFactory:
    """
    Embedding 工厂类

    根据配置创建合适的嵌入模型实例。

    使用方式：
        ```python
        # 从配置创建
        config = EmbeddingConfig(provider="openai", model="text-embedding-3-small")
        embed = EmbeddingFactory.create(config)

        # 从环境变量创建
        embed = EmbeddingFactory.create_from_env(provider="openai")
        ```
    """

    _REGISTRY: dict[str, type[BaseEmbedding]] = {
        "openai": OpenAIEmbedding,
        "dashscope": DashScopeEmbedding,
        "local": LocalEmbedding,
        "mock": MockEmbedding,
    }

    @classmethod
    def create(cls, config: EmbeddingConfig | dict[str, Any]) -> BaseEmbedding:
        """
        根据配置创建嵌入模型

        Args:
            config: EmbeddingConfig 实例或 dict

        Returns:
            BaseEmbedding 实例
        """
        if isinstance(config, dict):
            config = EmbeddingConfig(**config)

        provider = config.provider.lower()
        if provider not in cls._REGISTRY:
            available = ", ".join(cls._REGISTRY.keys())
            raise ValueError(
                f"不支持的 Embedding 提供者: {provider}，支持的: {available}"
            )

        impl_cls = cls._REGISTRY[provider]

        # 构造参数（只传递非 None 且对应该 provider 的参数）
        kwargs: dict[str, Any] = {}
        # model: DashScope 和 OpenAI 支持
        if config.model and provider in ("openai", "dashscope"):
            kwargs["model"] = config.model
        if config.api_key:
            kwargs["api_key"] = config.api_key
        if config.api_base:
            kwargs["api_base"] = config.api_base
        if config.dimensions:
            kwargs["dimensions"] = config.dimensions
        if config.batch_size:
            kwargs["batch_size"] = config.batch_size
        if config.timeout:
            kwargs["timeout"] = config.timeout

        logger.info(f"创建 Embedding: provider={config.provider}, model={config.model}")
        return impl_cls(**kwargs)

    @classmethod
    def create_from_env(cls, provider: str = "openai") -> BaseEmbedding:
        """
        从环境变量创建嵌入模型

        Args:
            provider: 提供者名称

        Returns:
            BaseEmbedding 实例
        """
        from core.config import settings

        config = EmbeddingConfig(provider=provider)

        if provider == "openai":
            config.api_key = settings.DEEPSEEK_API_KEY or settings.llm_providers.deepseek_api_key
            config.api_base = settings.DEEPSEEK_API_BASE
        elif provider == "dashscope":
            config.api_key = settings.llm_providers.aliyun_api_key
        elif provider == "local":
            pass
        elif provider == "mock":
            pass

        return cls.create(config)

    @classmethod
    def register(cls, name: str, impl_cls: type[BaseEmbedding]) -> None:
        """
        注册自定义嵌入模型

        Args:
            name: 提供者名称
            impl_cls: 实现类
        """
        cls._REGISTRY[name.lower()] = impl_cls


__all__ = [
    "EmbeddingConfig",
    "BaseEmbedding",
    "OpenAIEmbedding",
    "DashScopeEmbedding",
    "LocalEmbedding",
    "MockEmbedding",
    "EmbeddingFactory",
]
