# -*- coding: utf-8 -*-
"""
嵌入模型模块

提供统一的嵌入模型接口：
- Embeddings: 嵌入模型抽象基类
- OpenAIEmbeddings: OpenAI 嵌入模型
- LocalEmbeddings: 本地 Ollama 嵌入模型

支持:
- embed_documents: 批量嵌入文档
- embed_query: 嵌入单个查询
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from langchain_openai import OpenAIEmbeddings as LCOpenAIEmbeddings

from core.logger import logger

# Ollama 嵌入模型 - 延迟导入
try:
    from langchain_ollama import OllamaEmbeddings as LCOllamaEmbeddings
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    LCOllamaEmbeddings = None


class Embeddings(ABC):
    """
    嵌入模型抽象基类

    定义嵌入接口，所有嵌入模型实现都应继承此类。
    """

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        批量嵌入文档

        Args:
            texts: 文本列表

        Returns:
            嵌入向量列表，每个向量为 float 列表
        """
        pass

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """
        嵌入单个查询

        Args:
            text: 查询文本

        Returns:
            嵌入向量
        """
        pass

    def embed(self, text: str) -> list[float]:
        """
        嵌入单个文本的简写方法

        Args:
            text: 文本

        Returns:
            嵌入向量
        """
        return self.embed_query(text)


class OpenAIEmbeddings(Embeddings):
    """
    OpenAI 嵌入模型

    基于 OpenAI API 的嵌入模型封装，支持：
    - text-embedding-3-small
    - text-embedding-3-large
    - text-embedding-ada-002
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "text-embedding-3-small",
        dimensions: int | None = None,
        timeout: float | None = None,
        max_retries: int = 3,
    ):
        """
        初始化 OpenAI 嵌入模型

        Args:
            api_key: OpenAI API Key，None 则从环境变量 OPENAI_API_KEY 读取
            model: 模型名称，默认 text-embedding-3-small
            dimensions: 嵌入维度，None 则使用模型默认维度
            timeout: 请求超时时间（秒）
            max_retries: 最大重试次数
        """
        self.api_key = api_key
        self.model = model
        self.dimensions = dimensions
        self.timeout = timeout
        self.max_retries = max_retries

        self._embeddings: LCOpenAIEmbeddings | None = None

    @property
    def _client(self) -> LCOpenAIEmbeddings:
        """懒加载 OpenAI 嵌入客户端"""
        if self._embeddings is None:
            self._embeddings = LCOpenAIEmbeddings(
                api_key=self.api_key,
                model=self.model,
                dimensions=self.dimensions,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )
        return self._embeddings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        批量嵌入文档

        Args:
            texts: 文本列表

        Returns:
            嵌入向量列表
        """
        if not texts:
            return []

        try:
            return self._client.embed_documents(texts)
        except Exception as e:
            logger.error(f"OpenAI 嵌入文档失败: {e}")
            raise

    def embed_query(self, text: str) -> list[float]:
        """
        嵌入单个查询

        Args:
            text: 查询文本

        Returns:
            嵌入向量
        """
        if not text:
            return []

        try:
            return self._client.embed_query(text)
        except Exception as e:
            logger.error(f"OpenAI 嵌入查询失败: {e}")
            raise

    def __repr__(self) -> str:
        return f"OpenAIEmbeddings(model={self.model}, dimensions={self.dimensions})"


class LocalEmbeddings(Embeddings):
    """
    本地 Ollama 嵌入模型

    基于 Ollama 服务运行本地嵌入模型，支持：
    - nomic-embed-text
    - mxbai-embed-large
    - 其他 Ollama 支持的嵌入模型
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        timeout: float | None = None,
    ):
        """
        初始化本地嵌入模型

        Args:
            model: Ollama 模型名称
            base_url: Ollama 服务地址
            timeout: 请求超时时间（秒）

        Raises:
            ImportError: langchain_ollama 未安装
        """
        if not OLLAMA_AVAILABLE:
            raise ImportError(
                "langchain-ollama 未安装。请运行: uv add langchain-ollama"
            )

        self.model = model
        self.base_url = base_url
        self.timeout = timeout

        self._embeddings: LCOllamaEmbeddings | None = None  # type: ignore[assignment]

    @property
    def _client(self) -> LCOllamaEmbeddings:  # type: ignore[valid-type]
        """懒加载 Ollama 嵌入客户端"""
        if self._embeddings is None:
            kwargs: dict[str, Any] = {
                "model": self.model,
                "base_url": self.base_url,
            }
            if self.timeout is not None:
                kwargs["timeout"] = self.timeout

            self._embeddings = LCOllamaEmbeddings(**kwargs)  # type: ignore[operator]
        return self._embeddings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        批量嵌入文档

        Args:
            texts: 文本列表

        Returns:
            嵌入向量列表
        """
        if not texts:
            return []

        try:
            return self._client.embed_documents(texts)
        except Exception as e:
            logger.error(f"Ollama 嵌入文档失败: {e}")
            raise

    def embed_query(self, text: str) -> list[float]:
        """
        嵌入单个查询

        Args:
            text: 查询文本

        Returns:
            嵌入向量
        """
        if not text:
            return []

        try:
            return self._client.embed_query(text)
        except Exception as e:
            logger.error(f"Ollama 嵌入查询失败: {e}")
            raise

    def __repr__(self) -> str:
        return f"LocalEmbeddings(model={self.model}, base_url={self.base_url})"


__all__ = [
    "Embeddings",
    "OpenAIEmbeddings",
    "LocalEmbeddings",
]
