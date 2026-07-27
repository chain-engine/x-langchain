# -*- coding: utf-8 -*-
"""
Chains & LCEL 示例

演示如何使用 chains 模块和 LCEL 组合。
"""

from llms import create_chat_model
from chains import LLMChain, create_llm_chain, create_retrieval_qa_chain
from retrieval import MockEmbedding, InMemoryVectorStore, VectorRetriever
from output_parsers import PydanticOutputParser
from pydantic import BaseModel


class Answer(BaseModel):
    answer: str
    confidence: float


def demo_llm_chain():
    """简单 LLMChain"""
    print("=" * 60)
    print("[LLMChain] 简单问答链")

    llm = create_chat_model("mock")
    chain = create_llm_chain(
        llm=llm,
        system_message="你是一个友好的助手。",
        user_template="请回答: {question}",
    )

    result = chain.invoke({"question": "你好！"})
    print(f"[输入] question=你好！")
    print(f"[输出] {result}")


def demo_llm_chain_with_parser():
    """带结构化输出的 LLMChain"""
    print("\n" + "=" * 60)
    print("[LLMChain + Parser] 结构化输出")

    llm = create_chat_model("mock")
    parser = PydanticOutputParser(pydantic_model=Answer)

    chain = create_llm_chain(
        llm=llm,
        system_message="请给出简洁回答。",
        user_template="问题: {question}",
        output_parser=parser,
    )

    # 注意：这里 parser 需要配合 LLM 的 with_structured_output 使用
    print("[提示] 完整结构化输出需配合 bind_to_model")


def demo_rag_chain():
    """RAG 问答链"""
    print("\n" + "=" * 60)
    print("[RAG Chain] 检索增强问答")

    # 准备向量数据
    embed = MockEmbedding(dimensions=8)
    vs = InMemoryVectorStore(embedding=embed)
    docs = [
        "LangChain 是一个用于构建 LLM 应用的框架",
        "支持工具调用、记忆管理和 Agent 开发",
        "核心模块包括 Model I/O、Chains、Tools、Memory 和 Agents",
        "可以与多种 LLM 集成，如 OpenAI、DeepSeek 等",
    ]
    vs.add_texts(docs)

    # 构建 RAG 链
    retriever = VectorRetriever(vectorstore=vs, k=2)
    llm = create_chat_model("mock")

    chain = create_retrieval_qa_chain(
        llm=llm,
        retriever=retriever,
        system_message=(
            "你是一个知识助手。根据以下上下文信息回答问题。"
            "如果不知道，就说不知道。"
        ),
        return_source_documents=True,
    )

    print("[知识库已加载] 4 条文档")
    questions = [
        "LangChain 是什么？",
        "核心模块有哪些？",
    ]
    for q in questions:
        print(f"\n[问题] {q}")
        result = chain.invoke({"query": q})
        print(f"[回答] {result}")


def demo_lcel_manual():
    """手动 LCEL 组合"""
    print("\n" + "=" * 60)
    print("[LCEL] 手动管道组合")

    from lcel import RunnableLambda, RunnablePassthrough
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    llm = create_chat_model("mock")
    prompt = ChatPromptTemplate.from_template("请把以下文字翻译成英文: {text}")

    def add_exclaim(text: str) -> str:
        return text + "!!!"

    # 构建链: prompt | llm | add_exclaim
    chain = prompt | llm | StrOutputParser() | RunnableLambda(add_exclaim)

    result = chain.invoke({"text": "Hello"})
    print(f"[LCEL 链] 输入: Hello, 输出: {result}")


if __name__ == "__main__":
    demo_llm_chain()
    demo_llm_chain_with_parser()
    demo_rag_chain()
    demo_lcel_manual()
