# -*- coding: utf-8 -*-
"""
RetrievalQAChain - RAG 检索问答链

封装 LangChain 的 RetrievalQA 模式。
"""

from __future__ import annotations

from typing import Any, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable
from langchain_core.vectorstores import VectorStore

from core.logger import logger


class RetrievalQAChain:
    """
    RetrievalQAChain - RAG 检索问答链

    封装 "检索 + 问答" 模式：
    Question -> [Retriever] -> [Context] -> [LLM] -> Answer

    使用方式：
        ```python
        from llms import create_chat_model
        from retrieval import VectorRetriever
        from chains import RetrievalQAChain

        llm = create_chat_model()
        retriever = VectorRetriever(vectorstore=vs, k=3)
        chain = RetrievalQAChain(llm=llm, retriever=retriever)

        result = chain.invoke({"query": "项目配置在哪里？"})
        print(result["answer"])
        ```
    """

    def __init__(
        self,
        llm: BaseChatModel,
        retriever: BaseRetriever | VectorStore,
        *,
        system_message: Optional[str] = None,
        output_key: str = "answer",
        input_key: str = "query",
        return_source_documents: bool = False,
        output_parser: Optional[Runnable] = None,
        combine_docs_chain_config: Optional[dict[str, Any]] = None,
    ):
        """
        初始化 RetrievalQAChain

        Args:
            llm: 聊天模型
            retriever: 检索器或向量存储
            system_message: 系统提示词
            output_key: 输出结果键名
            input_key: 输入问题键名
            return_source_documents: 是否返回源文档
            output_parser: 输出解析器
            combine_docs_chain_config: 文档组合链配置
        """
        self._llm = llm
        self._input_key = input_key
        self._output_key = output_key
        self._return_source_documents = return_source_documents

        # 获取 retriever
        if hasattr(retriever, "as_retriever"):
            self._retriever: BaseRetriever = retriever.as_retriever()
        else:
            self._retriever = retriever

        # 构建 system_message
        self._system_message = system_message or (
            "你是一个助手。请根据以下上下文信息回答用户问题。"
            "如果上下文中没有相关信息，请如实说明。"
        )

        # 构建 combine_docs_chain
        self._combine_docs_chain = self._build_combine_docs_chain(output_parser)

        # 构建最终链
        self._chain: Runnable = self._build_chain()

        logger.info(
            f"RetrievalQAChain 初始化: input_key={input_key}, "
            f"output_key={output_key}, return_docs={return_source_documents}"
        )

    def _build_combine_docs_chain(
        self, output_parser: Optional[Runnable]
    ) -> Runnable:
        """构建文档组合链"""
        template = (
            f"{self._system_message}\n\n"
            "上下文信息:\n{context}\n\n"
            f"问题: {{{self._input_key}}}"
        )
        prompt = ChatPromptTemplate.from_template(template)

        chain: Runnable = prompt | self._llm
        if output_parser:
            chain = chain | output_parser
        else:
            chain = chain | StrOutputParser()

        return chain

    def _build_chain(self) -> Runnable:
        """构建完整的 RAG 链"""
        from langchain_core.runnables import RunnablePassthrough

        def format_docs(docs: list[Any]) -> str:
            return "\n\n".join(
                f"[文档 {i+1}]: {getattr(doc, 'page_content', str(doc))}"
                for i, doc in enumerate(docs)
            )

        def combine(input_dict: dict[str, Any]) -> dict[str, Any]:
            docs = input_dict.get("source_documents", [])
            return {
                self._input_key: input_dict.get(self._input_key, ""),
                "context": format_docs(docs),
            }

        def select_output(input_dict: dict[str, Any]) -> dict[str, Any]:
            result = {self._output_key: input_dict.get("text", "")}
            if self._return_source_documents:
                result["source_documents"] = input_dict.get("source_documents", [])
            return result

        # Chain: retrieve -> combine docs -> LLM -> select output
        from langchain_core.runnables import RunnableBranch

        # 使用 RunnablePassthrough 确保输入传递
        retrieve_chain = {
            "source_documents": self._retriever | RunnablePassthrough(),
            self._input_key: RunnablePassthrough(),
        }

        chain: Runnable = (
            retrieve_chain
            | (RunnableBranch.from_edged(
                [(lambda x: True, lambda x: combine(x))]
            ) if False else lambda x: combine(x))
        )
        # 简化：直接用函数组合
        chain = (
            {
                "source_documents": self._retriever,
                self._input_key: RunnablePassthrough(),
            }
            | (
                lambda x: {
                    "source_documents": x["source_documents"],
                    self._input_key: x.get(self._input_key, ""),
                    "context": format_docs(x.get("source_documents", [])),
                }
            )
            | self._combine_docs_chain
            | (lambda x: {self._output_key: x, "source_documents": []})
        )

        return chain

    @property
    def chain(self) -> Runnable:
        return self._chain

    def invoke(self, input: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        """
        调用 RAG 链

        Args:
            input: 输入字典，包含 query 键

        Returns:
            结果字典，包含 answer 键（和可选的 source_documents）
        """
        return self._chain.invoke(input, **kwargs)

    async def ainvoke(self, input: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        """异步调用"""
        return await self._chain.ainvoke(input, **kwargs)

    def stream(self, input: dict[str, Any], **kwargs: Any):
        """流式调用"""
        return self._chain.stream(input, **kwargs)

    def __repr__(self) -> str:
        return f"<RetrievalQAChain: input={self._input_key}, output={self._output_key}>"


# =============================================================================
# 工厂函数
# =============================================================================


def create_retrieval_qa_chain(
    llm: BaseChatModel,
    retriever: BaseRetriever | VectorStore,
    *,
    system_message: Optional[str] = None,
    input_key: str = "query",
    output_key: str = "answer",
    return_source_documents: bool = False,
) -> RetrievalQAChain:
    """
    工厂函数：创建 RAG 问答链

    Args:
        llm: 聊天模型
        retriever: 检索器或向量存储
        system_message: 系统提示词
        input_key: 输入键名
        output_key: 输出键名
        return_source_documents: 是否返回源文档

    Returns:
        RetrievalQAChain 实例
    """
    return RetrievalQAChain(
        llm=llm,
        retriever=retriever,
        system_message=system_message,
        input_key=input_key,
        output_key=output_key,
        return_source_documents=return_source_documents,
    )


__all__ = ["RetrievalQAChain", "create_retrieval_qa_chain"]
