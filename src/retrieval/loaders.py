# -*- coding: utf-8 -*-
"""
文档加载器模块

提供统一的文档加载接口，支持多种格式：
- TextLoader: 纯文本文件 (.txt)
- CSVLoader: CSV 表格文件
- JSONLoader: JSON 数据文件

每个加载器都支持：
- load(): 加载文档
- load_and_split(): 加载并分割文档
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_community.document_loaders import TextLoader as LCTextLoader
from langchain_community.document_loaders import CSVLoader as LCCSVLoader
from langchain_community.document_loaders import JSONLoader as LCJSONLoader
from langchain_core.documents import Document as LCDocument

from core.logger import logger


@dataclass
class Document:
    """
    文档数据类

    Attributes:
        page_content: 文档文本内容
        metadata: 文档元数据（来源、页码等）
    """

    page_content: str
    metadata: dict = field(default_factory=dict)

    def to_langchain_document(self) -> LCDocument:
        """转换为 LangChain Document 格式"""
        return LCDocument(page_content=self.page_content, metadata=self.metadata)

    @classmethod
    def from_langchain_document(cls, doc: LCDocument) -> "Document":
        """从 LangChain Document 创建"""
        return cls(page_content=doc.page_content, metadata=doc.metadata)


class DocumentLoader(ABC):
    """
    文档加载器抽象基类

    所有具体加载器都应继承此类并实现 load 方法。
    """

    @abstractmethod
    def load(self) -> list[Document]:
        """
        加载文档

        Returns:
            Document 列表

        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 文件格式错误
        """
        pass

    def load_and_split(self, text_splitter: Any = None) -> list[Document]:
        """
        加载并分割文档

        Args:
            text_splitter: 文本分割器（可选）

        Returns:
            分割后的 Document 列表
        """
        docs = self.load()

        if text_splitter is None:
            return docs

        from .splitters import RecursiveCharacterTextSplitter

        if isinstance(text_splitter, RecursiveCharacterTextSplitter):
            lc_docs = [doc.to_langchain_document() for doc in docs]
            split_docs = text_splitter.split_documents(lc_docs)
            return [Document.from_langchain_document(doc) for doc in split_docs]

        raise TypeError(
            f"text_splitter 类型错误，期望 RecursiveCharacterTextSplitter，实际 {type(text_splitter)}"
        )


class TextDocumentLoader(DocumentLoader):
    """
    纯文本文件加载器

    用于加载 .txt 等纯文本文件。
    """

    def __init__(self, file_path: str):
        """
        初始化文本加载器

        Args:
            file_path: 文件路径
        """
        self.file_path = Path(file_path)

    def load(self) -> list[Document]:
        """
        加载纯文本文件

        Returns:
            包含文件内容的 Document 列表

        Raises:
            FileNotFoundError: 文件不存在
            UnicodeDecodeError: 文件编码错误
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"文件不存在: {self.file_path}")

        if not self.file_path.is_file():
            raise ValueError(f"路径不是文件: {self.file_path}")

        try:
            loader = LCTextLoader(str(self.file_path), autodetect_encoding=True)
            lc_docs = loader.load()
            return [Document.from_langchain_document(doc) for doc in lc_docs]
        except UnicodeDecodeError as e:
            logger.error(f"文件编码错误 {self.file_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"加载文件失败 {self.file_path}: {e}")
            raise


class CSVLoader(DocumentLoader):
    """
    CSV 文件加载器

    用于加载 CSV 表格文件，每行作为一个 Document。
    """

    def __init__(self, file_path: str, encoding: str = "utf-8"):
        """
        初始化 CSV 加载器

        Args:
            file_path: 文件路径
            encoding: 文件编码，默认 utf-8
        """
        self.file_path = Path(file_path)
        self.encoding = encoding

    def load(self) -> list[Document]:
        """
        加载 CSV 文件

        Returns:
            包含 CSV 行的 Document 列表

        Raises:
            FileNotFoundError: 文件不存在
            csv.Error: CSV 格式错误
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"文件不存在: {self.file_path}")

        if not self.file_path.is_file():
            raise ValueError(f"路径不是文件: {self.file_path}")

        try:
            loader = LCCSVLoader(
                str(self.file_path),
                encoding=self.encoding,
                source_column=None,
            )
            lc_docs = loader.load()
            return [Document.from_langchain_document(doc) for doc in lc_docs]
        except UnicodeDecodeError as e:
            logger.error(f"文件编码错误 {self.file_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"加载 CSV 文件失败 {self.file_path}: {e}")
            raise


class JSONLoader(DocumentLoader):
    """
    JSON 文件加载器

    使用 jq schema 提取 JSON 中的文本内容。
    """

    def __init__(self, file_path: str, jq_schema: str = ". "):
        """
        初始化 JSON 加载器

        Args:
            file_path: 文件路径
            jq_schema: jq 查询表达式，用于提取内容，默认 ". " 提取整个 JSON
        """
        self.file_path = Path(file_path)
        self.jq_schema = jq_schema

    def load(self) -> list[Document]:
        """
        加载 JSON 文件

        Returns:
            包含提取内容的 Document 列表

        Raises:
            FileNotFoundError: 文件不存在
            json.JSONDecodeError: JSON 格式错误
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"文件不存在: {self.file_path}")

        if not self.file_path.is_file():
            raise ValueError(f"路径不是文件: {self.file_path}")

        try:
            loader = LCJSONLoader(
                file_path=str(self.file_path),
                jq_schema=self.jq_schema,
                text_content=False,
            )
            lc_docs = loader.load()
            return [Document.from_langchain_document(doc) for doc in lc_docs]
        except json.JSONDecodeError as e:
            logger.error(f"JSON 格式错误 {self.file_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"加载 JSON 文件失败 {self.file_path}: {e}")
            raise


__all__ = [
    "Document",
    "DocumentLoader",
    "TextDocumentLoader",
    "CSVLoader",
    "JSONLoader",
]
