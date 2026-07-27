# -*- coding: utf-8 -*-
"""
Chains 模块 - LangChain Chain 封装

提供 LangChain 经典 Chain 的工程化封装：
- LLMChain: Prompt + LLM 最简链
- RetrievalQAChain: RAG 检索问答链
- load_qa_chain: 文档问答链
"""

from .llm_chain import LLMChain, create_llm_chain
from .retrieval_qa_chain import RetrievalQAChain, create_retrieval_qa_chain

__all__ = [
    "LLMChain",
    "create_llm_chain",
    "RetrievalQAChain",
    "create_retrieval_qa_chain",
]
