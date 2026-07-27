# -*- coding: utf-8 -*-
"""基于检索增强生成的 LCEL 问答链。"""

from typing import Any

from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda

_DEFAULT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "请根据以下上下文回答问题。如果上下文中没有答案，请明确说明。\n\n上下文：\n{context}",
        ),
        ("human", "{question}"),
    ]
)


def _format_docs(documents: Any) -> str:
    """将检索结果转换为提示词使用的上下文字符串。"""
    if isinstance(documents, str):
        return documents
    return "\n\n".join(
        doc.page_content if hasattr(doc, "page_content") else str(doc)
        for doc in documents
    )


class RetrievalQAChain:
    """将检索器、文档格式化器、提示词和 LLM 组合成 RAG 链。"""

    def __init__(
        self,
        llm: BaseLanguageModel | Runnable,
        retriever: Runnable,
        prompt_template: Any = None,
        output_parser: Runnable | None = None,
    ) -> None:
        """初始化检索问答链。"""
        self.llm = llm
        self.retriever = retriever
        self.prompt = prompt_template or _DEFAULT_PROMPT
        self.output_parser = output_parser or StrOutputParser()
        format_docs = RunnableLambda(_format_docs)
        self._chain: Runnable = (
            {"context": retriever | format_docs, "question": RunnableLambda(lambda x: x)}
            | self.prompt
            | llm
            | self.output_parser
        )

    def invoke(self, query: str, **kwargs: Any) -> str:
        """同步执行检索问答。"""
        return self._chain.invoke(query, config=kwargs.pop("config", None))

    async def ainvoke(self, query: str, **kwargs: Any) -> str:
        """异步执行检索问答。"""
        return await self._chain.ainvoke(query, config=kwargs.pop("config", None))

    def stream(self, query: str, **kwargs: Any):
        """流式执行检索问答。"""
        yield from self._chain.stream(query, config=kwargs.pop("config", None))


def create_rag_chain(
    llm: BaseLanguageModel | Runnable,
    vectorstore: Any,
    prompt_template: Any = None,
) -> RetrievalQAChain:
    """从向量存储创建检索问答链。"""
    if not hasattr(vectorstore, "as_retriever"):
        raise TypeError("vectorstore 必须提供 as_retriever 方法")
    return RetrievalQAChain(llm, vectorstore.as_retriever(), prompt_template)


__all__ = ["RetrievalQAChain", "create_rag_chain"]
