# -*- coding: utf-8 -*-
"""
文本分割器模块

提供文档分块能力：
- TextSplitter: 文本分割器抽象基类
- RecursiveCharacterTextSplitter: 基于字符的递归分块

支持多种分割策略，可配置：
- chunk_size: 块大小
- chunk_overlap: 块重叠
- separators: 分隔符列表
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from langchain_core.documents import Document as LCDocument
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter as LCTextSplitter,
)

from core.logger import logger


class TextSplitter(ABC):
    """
    文本分割器抽象基类

    定义文档分割接口。
    """

    @abstractmethod
    def split_documents(self, documents: list[Any]) -> list[Any]:
        """
        分割文档列表

        Args:
            documents: Document 对象列表

        Returns:
            分割后的文档列表
        """
        pass

    def split_text(self, text: str) -> list[str]:
        """
        分割单个文本

        Args:
            text: 待分割的文本

        Returns:
            分割后的文本片段列表
        """
        raise NotImplementedError("子类必须实现 split_text 方法")


class RecursiveCharacterTextSplitter(TextSplitter):
    """
    递归字符文本分割器

    基于 langchain_text_splitters.RecursiveCharacterTextSplitter 实现，
    递归尝试多种分隔符进行文本分割，确保语义完整性。

    默认分隔符（按优先级）：
    1. 双换行（段落）
    2. 换行
    3. 空格
    4. 单个字符

    Attributes:
        chunk_size: 每个块的最大字符数
        chunk_overlap: 相邻块之间的重叠字符数
        separators: 分隔符列表，按优先级排序
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: list[str] | None = None,
        keep_separator: bool = True,
        add_start_index: bool = True,
    ):
        """
        初始化文本分割器

        Args:
            chunk_size: 每个块的最大字符数，默认 500
            chunk_overlap: 相邻块之间的重叠字符数，默认 50
            separators: 自定义分隔符列表，None 使用默认分隔符
            keep_separator: 是否保留分隔符，默认 True
            add_start_index: 是否在每个块中添加起始索引，默认 True
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]
        self.keep_separator = keep_separator
        self.add_start_index = add_start_index

        self._splitter = LCTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            keep_separator=keep_separator,
            add_start_index=add_start_index,
        )

    def split_documents(self, documents: list[Any]) -> list[Any]:
        """
        分割文档列表

        Args:
            documents: Document 对象列表

        Returns:
            分割后的文档列表
        """
        if not documents:
            return []

        try:
            return self._splitter.split_documents(documents)
        except Exception as e:
            logger.error(f"分割文档失败: {e}")
            raise

    def split_text(self, text: str) -> list[str]:
        """
        分割单个文本

        Args:
            text: 待分割的文本

        Returns:
            分割后的文本片段列表
        """
        if not text:
            return []

        try:
            return self._splitter.split_text(text)
        except Exception as e:
            logger.error(f"分割文本失败: {e}")
            raise

    def create_documents(
        self, texts: list[str], metadatas: list[dict] | None = None
    ) -> list[LCDocument]:
        """
        从文本列表创建文档

        Args:
            texts: 文本列表
            metadatas: 对应的元数据列表

        Returns:
            Document 对象列表
        """
        if metadatas is None:
            metadatas = [{}] * len(texts)

        try:
            return self._splitter.create_documents(texts, metadatas)
        except Exception as e:
            logger.error(f"创建文档失败: {e}")
            raise


__all__ = [
    "TextSplitter",
    "RecursiveCharacterTextSplitter",
]
