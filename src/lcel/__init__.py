# -*- coding: utf-8 -*-
"""
LCEL 模块 - LangChain Expression Language 组件

提供基于 LCEL 流水线的组件组合能力：
- RunnableLambda: Python 函数包装为 Runnable
- RunnablePassthrough: 输入透传
- LCELChain: 通用 LCEL 流水线构建器
- create_simple_chain: 快速创建简单 Chain
- create_rag_chain: RAG 问答 Chain
"""

from .chain import LCELChain, create_simple_chain, create_rag_chain
from .lcel_utils import RunnableLambda, RunnablePassthrough

__all__ = [
    "LCELChain",
    "create_simple_chain",
    "create_rag_chain",
    "RunnableLambda",
    "RunnablePassthrough",
]
