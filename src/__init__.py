# -*- coding: utf-8 -*-
"""
x-langchain - LangChain 学习与实践项目

基于 LangChain 五大核心能力的 Agent 开发框架：
- Model（模型）: langchain-core / langchain-openai
- Plan（规划）: LangGraph create_react_agent
- Action（行动）: LangGraph Runtime
- Tools（工具）: langchain_core.tools.@tool
- Memory（记忆）: langchain.memory
- Chains: LLMChain / RetrievalQAChain
- Retrieval: 向量存储 + 语义记忆 (RAG)
- LCEL: Runnable 表达式语言
- Output Parsers: 结构化输出解析
- Callbacks: 可观测性与追踪
"""

__version__ = "0.4.0"

import agent
import llms
import memories
import tools
import prompts
import chains
import retrieval
import output_parsers
import callbacks
import lcel

__all__ = [
    "agent",
    "llms",
    "memories",
    "tools",
    "prompts",
    "chains",
    "retrieval",
    "output_parsers",
    "callbacks",
    "lcel",
]
