# -*- coding: utf-8 -*-
"""
文档抽象

提供统一的 Document 数据结构和多种文档加载器。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from langchain_core.documents import Document as LCDocument


# =============================================================================
# Document
# =============================================================================


@dataclass
class Document:
    """
    文档数据模型

    对标 langchain_core.documents.Document，提供更简洁的接口。

    Attributes:
        page_content: 文档内容
        metadata: 元数据（来源、页码、时间等）
    """

    page_content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None

    def to_langchain(self) -> LCDocument:
        """转换为 langchain Document"""
        return LCDocument(
            page_content=self.page_content,
            metadata=self.metadata,
            id=self.id,
        )

    @classmethod
    def from_langchain(cls, doc: LCDocument) -> "Document":
        """从 langchain Document 创建"""
        return cls(
            page_content=doc.page_content,
            metadata=doc.metadata,
            id=doc.id,
        )

    def __str__(self) -> str:
        return self.page_content[:200] + ("..." if len(self.page_content) > 200 else "")


# =============================================================================
# Document Loader
# =============================================================================


class DocumentLoader:
    """
    文档加载器

    支持多种格式的文档加载：
    - Text: 纯文本 (.txt)
    - Markdown: Markdown 文档 (.md)
    - CSV: CSV 文件
    - JSON: JSON 文件（每行为一个文档）
    - PDF: PDF 文件（需要 pdfplumber）
    """

    @staticmethod
    def load_text(file_path: str | Path, encoding: str = "utf-8") -> list[Document]:
        """
        加载纯文本文件

        Args:
            file_path: 文件路径
            encoding: 文件编码

        Returns:
            文档列表（单文档）
        """
        path = Path(file_path)
        content = path.read_text(encoding=encoding)
        return [
            Document(
                page_content=content,
                metadata={"source": str(path), "type": "text"},
            )
        ]

    @staticmethod
    def load_markdown(
        file_path: str | Path, encoding: str = "utf-8"
    ) -> list[Document]:
        """
        加载 Markdown 文件，按标题分块

        Args:
            file_path: 文件路径
            encoding: 文件编码

        Returns:
            文档列表
        """
        import re

        path = Path(file_path)
        content = path.read_text(encoding=encoding)

        # 按 ## 标题分块
        sections = re.split(r"(?=^##\s+)", content, flags=re.MULTILINE)
        documents = []
        for i, section in enumerate(sections):
            section = section.strip()
            if not section:
                continue
            # 提取标题作为元数据
            heading = ""
            match = re.match(r"^##\s+(.+)$", section, re.MULTILINE)
            if match:
                heading = match.group(1).strip()

            documents.append(
                Document(
                    page_content=section,
                    metadata={
                        "source": str(path),
                        "type": "markdown",
                        "section": heading or f"section_{i}",
                    },
                )
            )

        return documents

    @staticmethod
    def load_csv(
        file_path: str | Path,
        encoding: str = "utf-8",
        separator: str = ",",
    ) -> list[Document]:
        """
        加载 CSV 文件，每行作为一个文档

        Args:
            file_path: 文件路径
            encoding: 文件编码
            separator: 分隔符

        Returns:
            文档列表
        """
        import csv

        path = Path(file_path)
        documents = []

        with open(path, "r", encoding=encoding) as f:
            reader = csv.DictReader(f, delimiter=separator)
            for i, row in enumerate(reader):
                content = "\n".join(f"{k}: {v}" for k, v in row.items() if v)
                documents.append(
                    Document(
                        page_content=content,
                        metadata={"source": str(path), "type": "csv", "row": i},
                    )
                )

        return documents

    @staticmethod
    def load_json(
        file_path: str | Path, encoding: str = "utf-8"
    ) -> list[Document]:
        """
        加载 JSON 文件，每行（JSONL）或每个对象作为一个文档

        Args:
            file_path: 文件路径
            encoding: 文件编码

        Returns:
            文档列表
        """
        import json

        path = Path(file_path)
        content = path.read_text(encoding=encoding)
        content = content.strip()
        documents = []

        # 尝试 JSONL 格式（每行一个 JSON）
        if "\n" in content and all(line.strip().startswith("{") for line in content.split("\n") if line.strip()):
            for i, line in enumerate(content.split("\n")):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    documents.append(
                        Document(
                            page_content=json.dumps(obj, ensure_ascii=False),
                            metadata={"source": str(path), "type": "jsonl", "line": i},
                        )
                    )
                except json.JSONDecodeError:
                    continue
        else:
            # 尝试标准 JSON 数组
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    for i, item in enumerate(data):
                        if isinstance(item, dict):
                            content_str = json.dumps(item, ensure_ascii=False)
                            documents.append(
                                Document(
                                    page_content=content_str,
                                    metadata={"source": str(path), "type": "json", "index": i},
                                )
                            )
                elif isinstance(data, dict):
                    documents.append(
                        Document(
                            page_content=json.dumps(data, ensure_ascii=False),
                            metadata={"source": str(path), "type": "json"},
                        )
                    )
            except json.JSONDecodeError:
                pass

        return documents

    @staticmethod
    def load(
        file_path: str | Path,
        encoding: str = "utf-8",
        separator: str = ",",
    ) -> list[Document]:
        """
        自动识别格式并加载文档

        Args:
            file_path: 文件路径
            encoding: 文件编码
            separator: CSV 分隔符

        Returns:
            文档列表
        """
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix in {".txt", ".text"}:
            return DocumentLoader.load_text(path, encoding)
        elif suffix in {".md", ".markdown"}:
            return DocumentLoader.load_markdown(path, encoding)
        elif suffix == ".csv":
            return DocumentLoader.load_csv(path, encoding, separator)
        elif suffix == ".json":
            return DocumentLoader.load_json(path, encoding)
        else:
            raise ValueError(f"不支持的文件格式: {suffix}，支持的格式: .txt/.md/.csv/.json")


# =============================================================================
# Directory Loader
# =============================================================================


class DirectoryLoader:
    """
    目录加载器

    递归加载目录下所有支持的文档。
    """

    def __init__(
        self,
        path: str | Path,
        glob: str = "**/*",
        suffixes: Optional[set[str]] = None,
        encoding: str = "utf-8",
        exclude_patterns: Optional[list[str]] = None,
    ):
        """
        初始化目录加载器

        Args:
            path: 目录路径
            glob: 文件匹配模式（默认 recursive）
            suffixes: 允许的后缀（None = 所有支持的格式）
            encoding: 文件编码
            exclude_patterns: 排除的文件名模式
        """
        self._path = Path(path)
        self._glob = glob
        self._suffixes = suffixes or {".txt", ".md", ".markdown", ".csv", ".json"}
        self._encoding = encoding
        self._exclude = set(exclude_patterns or [])

    def load(self) -> list[Document]:
        """加载目录下所有文档"""
        all_docs = []
        for file_path in self._path.glob(self._glob):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in self._suffixes:
                continue
            if file_path.name in self._exclude:
                continue

            try:
                docs = DocumentLoader.load(file_path, self._encoding)
                all_docs.extend(docs)
            except Exception as e:
                from core.logger import logger
                logger.warning(f"加载文件失败 {file_path}: {e}")

        return all_docs


__all__ = ["Document", "DocumentLoader", "DirectoryLoader"]
