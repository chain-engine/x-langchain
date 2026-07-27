# -*- coding: utf-8 -*-
"""
LCEL Chain - 基于 LangChain Expression Language 的流水线

提供标准化的 LCEL Chain 构建方式：
- create_simple_chain: Prompt + LLM 简单链
- create_rag_chain: RAG 检索问答链
- LCELChain: 可配置的通用链
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.vectorstores import VectorStore

from core.logger import logger


# =============================================================================
# 通用 LCEL Chain
# =============================================================================


class LCELChain:
    """
    LCEL 流水线链

    封装一个可配置的 LCEL 流水线。
    支持任意 Runnable 组合，包括 Prompt、LLM、Parser、Retrieval 等。

    使用方式：
        ```python
        chain = LCELChain()
        chain.add_node(prompt_template, name="prompt")
        chain.add_node(llm, name="llm")
        chain.add_edge("prompt", "llm")
        result = chain.invoke({"question": "..."})
        ```
    """

    def __init__(self, name: str = "LCELChain"):
        """
        初始化 LCEL 链

        Args:
            name: 链名称（用于日志和追踪）
        """
        self._name = name
        self._nodes: dict[str, Runnable] = {}
        self._edges: list[tuple[str, str]] = []
        self._entry_point: Optional[str] = None
        self._output_key: str = "text"
        self._final_runnable: Optional[Runnable] = None

    def add_node(self, runnable: Runnable, name: str) -> "LCELChain":
        """
        添加节点

        Args:
            runnable: Runnable 实例
            name: 节点名称

        Returns:
            self（支持链式调用）
        """
        self._nodes[name] = runnable
        self._final_runnable = None  # 需要重建
        logger.debug(f"Chain [{self._name}] 添加节点: {name}")
        return self

    def add_edge(self, from_node: str, to_node: str) -> "LCELChain":
        """
        添加边（节点之间的连接）

        Args:
            from_node: 起始节点
            to_node: 目标节点

        Returns:
            self
        """
        if from_node not in self._nodes:
            raise ValueError(f"起始节点不存在: {from_node}")
        if to_node not in self._nodes:
            raise ValueError(f"目标节点不存在: {to_node}")
        self._edges.append((from_node, to_node))
        self._final_runnable = None
        return self

    def set_entry_point(self, node_name: str) -> "LCELChain":
        """设置入口节点"""
        if node_name not in self._nodes:
            raise ValueError(f"节点不存在: {node_name}")
        self._entry_point = node_name
        return self

    def set_output_key(self, key: str) -> "LCELChain":
        """设置输出键名"""
        self._output_key = key
        return self

    def _build(self) -> Runnable:
        """根据节点和边构建最终的 Runnable"""
        if not self._nodes:
            raise ValueError("Chain 没有节点")

        # 简单情况：单节点直接返回
        if len(self._nodes) == 1:
            return list(self._nodes.values())[0]

        # 线性链：按顺序 pipe
        if self._entry_point:
            order = self._get_node_order()
            runnable = self._nodes[order[0]]
            for node_name in order[1:]:
                runnable = runnable | self._nodes[node_name]
            return runnable

        # 多入口：查找入度为 0 的节点作为入口
        entry = self._find_entry_nodes()
        if len(entry) == 1:
            self.set_entry_point(entry[0])
            return self._build()

        # 无法自动推断，使用第一个节点
        self.set_entry_point(list(self._nodes.keys())[0])
        return self._build()

    def _get_node_order(self) -> list[str]:
        """拓扑排序获取节点顺序"""
        in_degree = {name: 0 for name in self._nodes}
        for f, t in self._edges:
            in_degree[t] += 1

        order = []
        queue = [name for name, deg in in_degree.items() if deg == 0]

        while queue:
            node = queue.pop(0)
            order.append(node)
            for f, t in self._edges:
                if f == node:
                    in_degree[t] -= 1
                    if in_degree[t] == 0:
                        queue.append(t)

        return order

    def _find_entry_nodes(self) -> list[str]:
        """查找入度为 0 的节点"""
        in_degree = {name: 0 for name in self._nodes}
        for _, t in self._edges:
            in_degree[t] += 1
        return [name for name, deg in in_degree.items() if deg == 0]

    @property
    def runnable(self) -> Runnable:
        """获取最终的 Runnable"""
        if self._final_runnable is None:
            self._final_runnable = self._build()
        return self._final_runnable

    def invoke(self, input: Any, **kwargs: Any) -> Any:
        """同步调用"""
        return self.runnable.invoke(input, **kwargs)

    def ainvoke(self, input: Any, **kwargs: Any) -> Any:
        """异步调用"""
        import asyncio

        async def _ainvoke():
            return await self.runnable.ainvoke(input, **kwargs)

        return _ainvoke()

    def stream(self, input: Any, **kwargs: Any):
        """流式调用"""
        return self.runnable.stream(input, **kwargs)

    def astream(self, input: Any, **kwargs: Any):
        """异步流式调用"""
        return self.runnable.astream(input, **kwargs)

    def batch(self, inputs: list[Any], **kwargs: Any) -> list[Any]:
        """批量调用"""
        return self.runnable.batch(inputs, **kwargs)

    def __repr__(self) -> str:
        return f"<LCELChain: {self._name}, nodes={list(self._nodes.keys())}>"


# =============================================================================
# 简单 Chain 工厂
# =============================================================================


def create_simple_chain(
    prompt: ChatPromptTemplate | str | list[tuple[str, str]],
    llm: BaseChatModel,
    output_parser: Optional[Runnable] = None,
    *,
    system_message: Optional[str] = None,
    output_key: str = "text",
) -> Runnable:
    """
    创建简单的 Prompt + LLM Chain

    支持三种 prompt 格式：
    1. ChatPromptTemplate 实例
    2. 模板字符串（自动构建）
    3. 消息列表 [("system", "..."), ("user", "{input}")]

    Args:
        prompt: 提示模板
        llm: 聊天模型
        output_parser: 输出解析器（可选）
        system_message: 系统消息（与 prompt 合并）
        output_key: 输出键名

    Returns:
        Runnable 链

    使用示例：
        ```python
        from llms import create_chat_model

        llm = create_chat_model()
        chain = create_simple_chain(
            prompt="请回答: {question}",
            llm=llm,
        )
        result = chain.invoke({"question": "你好"})
        ```
    """
    # 构建 ChatPromptTemplate
    if isinstance(prompt, ChatPromptTemplate):
        chat_prompt = prompt
    elif isinstance(prompt, str):
        if system_message:
            chat_prompt = ChatPromptTemplate.from_messages([
                ("system", system_message),
                ("user", prompt),
            ])
        else:
            chat_prompt = ChatPromptTemplate.from_template(prompt)
    elif isinstance(prompt, list):
        chat_prompt = ChatPromptTemplate.from_messages(prompt)
    else:
        raise TypeError(f"prompt 类型不支持: {type(prompt)}")

    # 构建链
    chain: Runnable = chat_prompt | llm

    if output_parser:
        chain = chain | output_parser
    elif output_key:
        # 默认使用 StrOutputParser
        chain = chain | StrOutputParser()

    return chain


# =============================================================================
# RAG Chain 工厂
# =============================================================================


def create_rag_chain(
    llm: BaseChatModel,
    retriever: BaseRetriever | VectorStore,
    *,
    system_message: Optional[str] = (
        "你是一个助手。根据以下上下文信息回答用户问题。"
        "如果上下文中没有相关信息，请明确说明。"
    ),
    question_key: str = "question",
    context_key: str = "context",
    output_parser: Optional[Runnable] = None,
) -> Runnable:
    """
    创建 RAG 检索问答 Chain

    Chain 结构：
        {question}
          |
          v
    [检索上下文] --> [组合上下文]
          |
          v
        [LLM]
          |
          v
    [输出解析] (可选)

    Args:
        llm: 聊天模型
        retriever: 检索器（BaseRetriever 或 VectorStore）
        system_message: 系统提示词
        question_key: 问题输入键名
        context_key: 上下文字典键名
        output_parser: 输出解析器

    Returns:
        RAG Runnable Chain

    使用示例：
        ```python
        from retrieval import VectorRetriever, ChromaVectorStore
        from llms import create_chat_model

        vs = ChromaVectorStore(...)
        retriever = VectorRetriever(vectorstore=vs, k=3)
        llm = create_chat_model()

        rag_chain = create_rag_chain(llm, retriever)
        result = rag_chain.invoke({"question": "公司的政策是什么？"})
        ```
    """
    # 从 VectorStore 获取 Retriever
    if hasattr(retriever, "as_retriever"):
        actual_retriever: BaseRetriever = retriever.as_retriever()
    else:
        actual_retriever = retriever

    # 构建 RAG prompt
    template = (
        f"{system_message}\n\n"
        f"上下文信息:\n{{{context_key}}}\n\n"
        f"问题: {{{question_key}}}"
    )
    rag_prompt = ChatPromptTemplate.from_template(template)

    # 构建 Chain
    def format_docs(docs: list[Any]) -> str:
        """格式化检索结果为上下文字符串"""
        return "\n\n".join(
            f"[文档 {i+1}]: {getattr(doc, 'page_content', str(doc))}"
            for i, doc in enumerate(docs)
        )

    chain: Runnable = (
        {
            context_key: actual_retriever | RunnablePassthrough.assign(
                original_input=lambda _: RunnablePassthrough()
            ),
            question_key: RunnablePassthrough(),
        }
        | {
            context_key: lambda x: format_docs(x.get(context_key, [])),
            question_key: lambda x: x.get(question_key, ""),
        }
        | rag_prompt
        | llm
    )

    if output_parser:
        chain = chain | output_parser

    return chain


# =============================================================================
# 可配置的 RAG Chain
# =============================================================================


class RAGChain:
    """
    可配置的 RAG Chain

    提供更灵活的 RAG 链构建方式：
    - 自定义检索参数
    - 自定义 Prompt
    - 支持多跳推理

    使用方式：
        ```python
        from retrieval import VectorRetriever

        retriever = VectorRetriever(vectorstore=vs, k=3)
        rag = RAGChain(llm, retriever)
        rag.set_reranker(lambda docs, query: sorted(docs, key=lambda d: len(d.page_content), reverse=True))
        result = rag.invoke({"question": "..."})
        ```
    """

    def __init__(
        self,
        llm: BaseChatModel,
        retriever: BaseRetriever,
        *,
        k: int = 3,
        score_threshold: Optional[float] = None,
        output_parser: Optional[Runnable] = None,
    ):
        self._llm = llm
        self._retriever = retriever
        self._k = k
        self._score_threshold = score_threshold
        self._output_parser = output_parser
        self._system_message = (
            "你是一个助手。根据以下上下文信息回答用户问题。"
            "如果上下文中没有相关信息，请明确说明。"
        )
        self._reranker: Optional[Callable[[list[Any], str], list[Any]]] = None
        self._query_rewriter: Optional[Runnable] = None
        self._chain: Optional[Runnable] = None

    def set_system_message(self, message: str) -> "RAGChain":
        """设置系统消息"""
        self._system_message = message
        self._chain = None
        return self

    def set_reranker(
        self, reranker: Callable[[list[Any], str], list[Any]]
    ) -> "RAGChain":
        """设置重排序函数"""
        self._reranker = reranker
        return self

    def set_query_rewriter(self, chain: Runnable) -> "RAGChain":
        """设置查询重写 Chain（用于多跳推理）"""
        self._query_rewriter = chain
        return self

    @property
    def chain(self) -> Runnable:
        """获取构建好的 Chain"""
        if self._chain is None:
            self._chain = self._build()
        return self._chain

    def _build(self) -> Runnable:
        """构建 Chain"""
        from langchain_core.output_parsers import StrOutputParser

        template = (
            f"{self._system_message}\n\n"
            "上下文信息:\n{context}\n\n"
            "问题: {question}"
        )
        prompt = ChatPromptTemplate.from_template(template)

        def format_docs(docs: list[Any]) -> str:
            return "\n\n".join(
                f"[文档 {i+1}]: {getattr(doc, 'page_content', str(doc))}"
                for i, doc in enumerate(docs)
            )

        chain: Runnable = (
            {
                "context": self._retriever | RunnablePassthrough(),
                "question": RunnablePassthrough(),
            }
            | {
                "context": lambda x: format_docs(x.get("context", [])),
                "question": lambda x: x.get("question", ""),
            }
            | prompt
            | self._llm
            | (self._output_parser or StrOutputParser())
        )
        return chain

    def invoke(self, input: dict[str, Any], **kwargs: Any) -> Any:
        return self.chain.invoke(input, **kwargs)

    def ainvoke(self, input: dict[str, Any], **kwargs: Any) -> Any:
        return self.chain.ainvoke(input, **kwargs)

    def stream(self, input: dict[str, Any], **kwargs: Any):
        return self.chain.stream(input, **kwargs)

    def astream(self, input: dict[str, Any], **kwargs: Any):
        return self.chain.astream(input, **kwargs)


__all__ = [
    "LCELChain",
    "RAGChain",
    "create_simple_chain",
    "create_rag_chain",
]
