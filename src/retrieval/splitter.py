# -*- coding: utf-8 -*-
"""
文档分块器

提供多种文档分块策略：
- RecursiveCharacterTextSplitter: 按字符递归分块（推荐）
- TokenTextSplitter: 按 Token 数分块（需 tiktoken）
- SemanticChunker: 按语义分块（需 NLTK）
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from langchain_core.documents import Document as LCDocument

from core.logger import logger


# =============================================================================
# 文本分块器基类
# =============================================================================


class TextSplitter:
    """
    文本分块器基类

    提供统一的分块接口，将长文本切分为小块。
    """

    def split_text(self, text: str) -> list[str]:
        """
        将单条文本分块

        Args:
            text: 输入文本

        Returns:
            文本块列表
        """
        raise NotImplementedError

    def split_documents(
        self, documents: list[Any]
    ) -> list[LCDocument]:
        """
        将文档列表分块

        Args:
            documents: Document 列表

        Returns:
            分块后的文档列表
        """
        from langchain_core.documents import Document as LCDoc

        chunks = []
        for doc in documents:
            # 统一转换为 langchain Document
            if hasattr(doc, "page_content"):
                lc_doc = doc if isinstance(doc, LCDoc) else LCDoc(
                    page_content=doc.page_content,
                    metadata=doc.metadata if hasattr(doc, "metadata") else {},
                )
            else:
                lc_doc = LCDoc(page_content=str(doc))

            texts = self.split_text(lc_doc.page_content)
            for i, chunk_text in enumerate(texts):
                chunks.append(
                    LCDoc(
                        page_content=chunk_text,
                        metadata={
                            **lc_doc.metadata,
                            "chunk_index": i,
                            "chunk_count": len(texts),
                        },
                    )
                )
        return chunks


# =============================================================================
# 递归字符分块器（推荐）
# =============================================================================


class RecursiveTextSplitter(TextSplitter):
    """
    递归字符分块器

    按字符数递归分块，保持语义完整性。
    支持多种分隔符，按优先级尝试切分。

    特性：
    - 按段落 -> 句子 -> 单词 逐级尝试
    - 保持单块最小长度，避免碎片化
    - 保留元数据
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: Optional[list[str]] = None,
        length_function: Callable[[str], int] = len,
        keep_separator: bool = True,
        is_separator_regex: bool = False,
    ):
        """
        初始化分块器

        Args:
            chunk_size: 每块最大字符数
            chunk_overlap: 块之间的重叠字符数
            separators: 分隔符列表（按优先级）
            length_function: 计算文本长度的函数
            keep_separator: 保留分隔符
            is_separator_regex: 分隔符是否为正则表达式
        """
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._separators = separators or ["\n\n", "\n", "。", "！", "？", " ", ""]
        self._length_function = length_function
        self._keep_separator = keep_separator
        self._is_separator_regex = is_separator_regex

    def split_text(self, text: str) -> list[str]:
        """分块核心逻辑"""
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter as LCSplitter
            splitter = LCSplitter(
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
                length_function=self._length_function,
                separators=self._separators,
                keep_separator=self._keep_separator,
                is_separator_regex=self._is_separator_regex,
            )
            return splitter.split_text(text)
        except ImportError:
            logger.warning("langchain-text-splitters 未安装，使用内置分块逻辑")
            return self._split_fallback(text)

    def _split_fallback(self, text: str) -> list[str]:
        """内置分块逻辑（无依赖）"""
        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + self._chunk_size, text_len)

            if end < text_len and start > 0:
                # 尝试在重叠区域内找分隔符
                search_start = max(start, end - self._chunk_overlap)
                found = False
                for sep in self._separators:
                    if not sep:
                        break
                    pos = text.rfind(sep, search_start, end)
                    if pos != -1:
                        end = pos + len(sep)
                        found = True
                        break
                if not found:
                    # 尝试在块末尾找空格
                    pos = text.rfind(" ", start, end)
                    if pos != -1:
                        end = pos

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - self._chunk_overlap
            if start >= text_len:
                break
            start = max(start, end - self._chunk_size)

        return chunks

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def chunk_overlap(self) -> int:
        return self._chunk_overlap


# =============================================================================
# Token 分块器
# =============================================================================


class TokenTextSplitter(TextSplitter):
    """
    Token 分块器

    按 Token 数分块，适用于需要精确控制 token 消耗的场景。
    需要 tiktoken 库。
    """

    def __init__(
        self,
        encoding_name: str = "cl100k_base",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        """
        初始化 Token 分块器

        Args:
            encoding_name: tiktoken 编码名称（cl100k_base / p50k_base / r50k_base）
            chunk_size: 每块最大 Token 数
            chunk_overlap: 块之间的重叠 Token 数
        """
        self._encoding_name = encoding_name
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

        try:
            import tiktoken

            self._enc = tiktoken.get_encoding(encoding_name)
        except ImportError:
            logger.warning("tiktoken 未安装，使用字符数近似")
            self._enc = None

    def split_text(self, text: str) -> list[str]:
        if self._enc is None:
            # 回退到字符数近似
            return self._split_by_chars(text)

        tokens = self._enc.encode(text)
        chunks = []
        start = 0

        while start < len(tokens):
            end = min(start + self._chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self._enc.decode(chunk_tokens)
            if chunk_text.strip():
                chunks.append(chunk_text)
            start = end - self._chunk_overlap
            if start >= len(tokens):
                break

        return chunks

    def _split_by_chars(self, text: str) -> list[str]:
        """按字符数近似分块（1 token ≈ 4 字符）"""
        approx_chunk_size = self._chunk_size * 4
        overlap = self._chunk_overlap * 4
        return RecursiveTextSplitter(
            chunk_size=approx_chunk_size, chunk_overlap=overlap
        ).split_text(text)


# =============================================================================
# 工厂
# =============================================================================


def create_text_splitter(
    splitter_type: str = "recursive",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    **kwargs: Any,
) -> TextSplitter:
    """
    工厂函数：创建文本分块器

    Args:
        splitter_type: 分块器类型（recursive / token）
        chunk_size: 块大小
        chunk_overlap: 重叠大小
        **kwargs: 其他参数

    Returns:
        TextSplitter 实例
    """
    if splitter_type == "recursive":
        return RecursiveTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            **kwargs,
        )
    elif splitter_type == "token":
        return TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            **kwargs,
        )
    else:
        raise ValueError(f"不支持的分块器类型: {splitter_type}")


__all__ = [
    "TextSplitter",
    "RecursiveTextSplitter",
    "TokenTextSplitter",
    "create_text_splitter",
]
