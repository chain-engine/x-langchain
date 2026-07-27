# -*- coding: utf-8 -*-
"""
上下文压缩检索器

基于 LangChain 的 ContextualCompressionRetriever 思想实现，
通过 LLM 对检索结果进行压缩，去除无关上下文，保留关键信息。

压缩器类型：
- LLMCompactor: 使用 LLM 将每个文档压缩为摘要
- DocumentCompressor: 基础文档压缩接口
- ChainFilter: 使用 Chain 选择是否保留文档
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document as LCDocument
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever

from core.logger import logger


# =============================================================================
# 文档压缩器基类
# =============================================================================


class DocumentCompressor(ABC):
    """
    文档压缩器抽象基类

    所有压缩器实现需继承此类并实现 compress_documents 方法。
    """

    @abstractmethod
    def compress_documents(
        self,
        documents: list[LCDocument],
        query: str,
        **kwargs: Any,
    ) -> list[LCDocument]:
        """
        压缩文档列表

        Args:
            documents: 原始文档列表
            query: 原始查询（用于判断相关性）
            **kwargs: 其他参数

        Returns:
            压缩后的文档列表
        """
        raise NotImplementedError


# =============================================================================
# LLM 压缩器
# =============================================================================


class LLMCompactor(DocumentCompressor):
    """
    基于 LLM 的文档压缩器

    使用 LLM 将每个文档压缩为简洁的摘要，同时保留与查询相关的信息。
    适用于长文档场景，减少 token 消耗同时保留关键信息。
    """

    DEFAULT_COMPRESSION_TEMPLATE = """Given the following query and document, extract the parts 
that are relevant to the query. Do not elaborate or add any additional information.

Query: {query}

Document: {document}

Relevant information (简短摘要，保留与查询相关的内容):"""

    def __init__(
        self,
        llm: BaseLanguageModel,
        prompt_template: Optional[str] = None,
        strip_documents: bool = True,
        max_chars: Optional[int] = None,
    ):
        """
        初始化 LLM 压缩器

        Args:
            llm: 用于压缩的 LLM
            prompt_template: 压缩提示模板，None 则使用默认模板
            strip_documents: 是否去除文档中的多余空白
            max_chars: 压缩后最大字符数（None 表示不限制）
        """
        self._llm = llm
        self._prompt_template = prompt_template or self.DEFAULT_COMPRESSION_TEMPLATE
        self._strip_documents = strip_documents
        self._max_chars = max_chars

    def compress_documents(
        self,
        documents: list[LCDocument],
        query: str,
        **kwargs: Any,
    ) -> list[LCDocument]:
        """使用 LLM 压缩每个文档"""
        if not documents:
            return []

        compressed = []
        for doc in documents:
            try:
                compressed_text = self._compress_single(doc, query)
                if compressed_text and compressed_text.strip():
                    # 保留原始元数据
                    metadata = dict(doc.metadata) if doc.metadata else {}
                    metadata["original_length"] = len(doc.page_content)
                    metadata["compressed_length"] = len(compressed_text)
                    metadata["compression_source"] = "llm_compactor"

                    compressed.append(
                        LCDocument(
                            page_content=compressed_text,
                            metadata=metadata,
                        )
                    )
            except Exception:
                logger.warning(f"文档压缩失败，保留原始文档")
                compressed.append(doc)

        logger.debug(f"LLMCompactor: {len(documents)} -> {len(compressed)} 文档")
        return compressed

    def _compress_single(self, doc: LCDocument, query: str) -> str:
        """压缩单个文档"""
        text = doc.page_content
        if self._strip_documents:
            text = " ".join(text.split())

        prompt = PromptTemplate.from_template(self._prompt_template)
        chain = prompt | self._llm | StrOutputParser()

        result = chain.invoke({"query": query, "document": text})

        if self._max_chars and len(result) > self._max_chars:
            result = result[: self._max_chars] + "..."

        return result


# =============================================================================
# Chain 过滤器压缩器
# =============================================================================


class ChainFilter(DocumentCompressor):
    """
    基于 Chain 的文档过滤器

    使用 LLM Chain 判断每个文档是否与查询相关，只保留相关文档。
    不会修改文档内容，只做相关性过滤。
    """

    DEFAULT_FILTER_TEMPLATE = """Given the following query and document, determine if the 
document contains information that is relevant to answering the query.

Query: {query}

Document: {document}

Respond with only YES or NO."""

    def __init__(
        self,
        llm: BaseLanguageModel,
        prompt_template: Optional[str] = None,
        filter_type: str = "include",  # include | exclude
    ):
        """
        初始化 Chain 过滤器

        Args:
            llm: 用于判断相关性的 LLM
            prompt_template: 过滤提示模板
            filter_type: 过滤类型
                - "include": 只保留相关文档（默认）
                - "exclude": 排除相关文档
        """
        self._llm = llm
        self._prompt_template = prompt_template or self.DEFAULT_FILTER_TEMPLATE
        if filter_type not in ("include", "exclude"):
            raise ValueError(f"filter_type 必须为 'include' 或 'exclude'，实际: {filter_type}")
        self._filter_type = filter_type

    def compress_documents(
        self,
        documents: list[LCDocument],
        query: str,
        **kwargs: Any,
    ) -> list[LCDocument]:
        """判断并过滤文档"""
        if not documents:
            return []

        prompt = PromptTemplate.from_template(self._prompt_template)
        chain = prompt | self._llm

        filtered = []
        for doc in documents:
            try:
                response = chain.invoke({"query": query, "document": doc.page_content})
                content = getattr(response, "content", str(response)).strip().lower()

                is_relevant = "yes" in content

                if self._filter_type == "include" and is_relevant:
                    filtered.append(doc)
                elif self._filter_type == "exclude" and not is_relevant:
                    filtered.append(doc)

            except Exception:
                logger.warning(f"文档过滤判断失败")
                # 判断失败时默认保留
                filtered.append(doc)

        logger.debug(
            f"ChainFilter: {len(documents)} -> {len(filtered)} 文档 "
            f"(filter_type={self._filter_type})"
        )
        return filtered


# =============================================================================
# 简单文本压缩器（无需 LLM）
# =============================================================================


class SimpleCompressor(DocumentCompressor):
    """
    基于规则的简单文档压缩器

    不需要 LLM，通过截断句子或按段落选择来压缩文档。
    适用于资源受限场景或快速原型。
    """

    def __init__(
        self,
        max_chars: int = 500,
        max_lines: Optional[int] = None,
        sentence_boundary: bool = True,
    ):
        """
        初始化简单压缩器

        Args:
            max_chars: 压缩后最大字符数
            max_lines: 压缩后最大行数（可选）
            sentence_boundary: 是否在句子的边界处截断
        """
        self._max_chars = max_chars
        self._max_lines = max_lines
        self._sentence_boundary = sentence_boundary

    def compress_documents(
        self,
        documents: list[LCDocument],
        query: str,
        **kwargs: Any,
    ) -> list[LCDocument]:
        """简单截断压缩"""
        if not documents:
            return []

        compressed = []
        for doc in documents:
            text = self._compress_single(doc.page_content)
            metadata = dict(doc.metadata) if doc.metadata else {}
            metadata["original_length"] = len(doc.page_content)
            metadata["compressed_length"] = len(text)
            metadata["compression_source"] = "simple_compressor"

            compressed.append(LCDocument(page_content=text, metadata=metadata))

        return compressed

    def _compress_single(self, text: str) -> str:
        """截断单个文档"""
        if self._max_lines:
            lines = text.split("\n")
            text = "\n".join(lines[: self._max_lines])

        if self._max_chars and len(text) > self._max_chars:
            if self._sentence_boundary:
                # 在最后一个句号处截断
                truncated = text[: self._max_chars]
                last_punct = max(
                    truncated.rfind("。"),
                    truncated.rfind("."),
                    truncated.rfind("！"),
                    truncated.rfind("!"),
                    truncated.rfind("？"),
                    truncated.rfind("?"),
                )
                if last_punct > self._max_chars * 0.5:
                    text = truncated[: last_punct + 1]
                else:
                    text = truncated + "..."
            else:
                text = text[: self._max_chars] + "..."

        return text


# =============================================================================
# 上下文压缩检索器
# =============================================================================


class ContextualCompressionRetriever(BaseRetriever):
    """
    上下文压缩检索器

    在底层检索器的基础上增加压缩步骤，使用压缩器去除无关上下文。
    支持多种压缩策略：LLM 压缩、Chain 过滤、规则截断。

    使用方式：
        ```python
        from retrieval import VectorRetriever, EmbeddingFactory
        from retrieval.compression import ContextualCompressionRetriever, LLMCompactor

        base_retriever = VectorRetriever(vectorstore, k=10)
        compressor = LLMCompactor(llm=chat_model)
        compressor_retriever = ContextualCompressionRetriever(
            base_retriever=base_retriever,
            compressor=compressor,
            top_k=5,
        )
        results = compressor_retriever.invoke("查询内容")
        ```
    """

    def __init__(
        self,
        base_retriever: Any,
        compressor: Optional[DocumentCompressor] = None,
        top_k: int = 4,
        search_kwargs: Optional[dict[str, Any]] = None,
    ):
        """
        初始化上下文压缩检索器

        Args:
            base_retriever: 底层检索器（实现 BaseRetriever 或有 get_relevant_documents 方法）
            compressor: 文档压缩器，None 则不压缩（退化为普通检索器）
            top_k: 压缩后返回的文档数量
            search_kwargs: 透传给底层检索器的额外参数
        """
        super().__init__()
        self._base_retriever = base_retriever
        self._compressor = compressor
        self._top_k = top_k
        self._search_kwargs = search_kwargs or {}

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> list[LCDocument]:
        """获取并压缩相关文档"""
        # Step 1: 从底层检索器获取较多文档（留出压缩空间）
        k = self._search_kwargs.get("k", self._top_k * 3)
        try:
            docs = self._base_retriever.get_relevant_documents(query, k=k)
        except TypeError:
            # 底层检索器不支持 k 参数
            docs = self._base_retriever.get_relevant_documents(query)

        if not docs:
            return []

        logger.debug(f"ContextualCompressionRetriever: 检索到 {len(docs)} 个候选文档")

        # Step 2: 压缩文档
        if self._compressor is not None:
            docs = self._compressor.compress_documents(docs, query)

        # Step 3: 限制返回数量
        return docs[: self._top_k]

    def set_compressor(self, compressor: DocumentCompressor) -> None:
        """动态切换压缩器"""
        self._compressor = compressor
        logger.debug(f"切换压缩器: {type(compressor).__name__}")

    def set_top_k(self, top_k: int) -> None:
        """动态调整返回数量"""
        self._top_k = top_k


# =============================================================================
# 工厂函数
# =============================================================================


def create_compression_retriever(
    base_retriever: Any,
    llm: Optional[BaseLanguageModel] = None,
    compression_type: str = "llm",
    top_k: int = 4,
    **kwargs: Any,
) -> ContextualCompressionRetriever:
    """
    工厂函数：创建上下文压缩检索器

    Args:
        base_retriever: 底层检索器
        llm: 用于压缩的 LLM（llm 模式必需）
        compression_type: 压缩类型
            - "llm": 使用 LLM 压缩（默认）
            - "filter": 使用 Chain 过滤
            - "simple": 简单截断压缩
        top_k: 返回文档数
        **kwargs: 透传给压缩器

    Returns:
        ContextualCompressionRetriever 实例
    """
    compressor: Optional[DocumentCompressor] = None

    if compression_type == "llm":
        if llm is None:
            raise ValueError("compression_type='llm' 时必须提供 llm 参数")
        compressor = LLMCompactor(llm=llm, **kwargs)
    elif compression_type == "filter":
        if llm is None:
            raise ValueError("compression_type='filter' 时必须提供 llm 参数")
        compressor = ChainFilter(llm=llm, **kwargs)
    elif compression_type == "simple":
        compressor = SimpleCompressor(**kwargs)
    elif compression_type != "none":
        raise ValueError(
            f"不支持的 compression_type: {compression_type}，"
            f"支持的: llm / filter / simple / none"
        )

    return ContextualCompressionRetriever(
        base_retriever=base_retriever,
        compressor=compressor,
        top_k=top_k,
    )


__all__ = [
    "DocumentCompressor",
    "LLMCompactor",
    "ChainFilter",
    "SimpleCompressor",
    "ContextualCompressionRetriever",
    "create_compression_retriever",
]
