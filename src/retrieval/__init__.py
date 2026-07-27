# -*- coding: utf-8 -*-
"""
Retrieval 模块 - 检索增强生成（RAG）基础设施

提供语义检索的完整工具链：
|- Embedding: 嵌入模型抽象与多后端支持
|- VectorStore: 向量存储抽象与多后端实现
|- DocumentLoader: 文档加载（Text / Markdown / CSV / JSON）
|- TextSplitter: 文档分块（按字符 / 语义）
|- Retriever: 检索器抽象
|- ContextualCompressionRetriever: 上下文压缩检索器
|- SemanticMemory: Agent 语义记忆（基于向量检索）

使用方式：
    ```python
    from retrieval import SemanticMemory, TextSplitter, EmbeddingFactory

    # 创建语义记忆
    memory = SemanticMemory(embedding=EmbeddingFactory.create("openai"))
    memory.add_documents(["项目文档内容..."])

    # 检索相关记忆
    results = memory.search("项目配置方法", k=3)
    ```
"""

from .embedding import (
    EmbeddingFactory,
    EmbeddingConfig,
    BaseEmbedding,
    OpenAIEmbedding,
    DashScopeEmbedding,
    LocalEmbedding,
    MockEmbedding,
)
from .vectorstore import (
    VectorStoreFactory,
    VectorStoreConfig,
    BaseVectorStore,
    ChromaVectorStore,
    FAISSVectorStore,
    InMemoryVectorStore,
)
from .document import Document, DocumentLoader, DirectoryLoader
from .splitter import TextSplitter, RecursiveTextSplitter, TokenTextSplitter, create_text_splitter
from .retriever import Retriever, VectorRetriever, EnsembleRetriever, MultiQueryRetriever
from .semantic_memory import SemanticMemory, SemanticMemoryConfig
from .compression import (
    DocumentCompressor,
    LLMCompactor,
    ChainFilter,
    SimpleCompressor,
    ContextualCompressionRetriever,
    create_compression_retriever,
)

__all__ = [
    # Embedding
    "EmbeddingFactory",
    "EmbeddingConfig",
    # VectorStore
    "VectorStoreFactory",
    "VectorStoreConfig",
    # Document
    "Document",
    "DocumentLoader",
    # Splitter
    "TextSplitter",
    "RecursiveTextSplitter",
    # Retriever
    "Retriever",
    "VectorRetriever",
    "EnsembleRetriever",
    # Compression
    "DocumentCompressor",
    "LLMCompactor",
    "ChainFilter",
    "SimpleCompressor",
    "ContextualCompressionRetriever",
    "create_compression_retriever",
    # Semantic Memory
    "SemanticMemory",
    "SemanticMemoryConfig",
]
