# -*- coding: utf-8 -*-
"""
x-langchain - LangChain 学习与实践项目

基于 LangChain 五大核心能力的 Agent 开发框架：
- Model（模型）: langchain-core / langchain-openai
- Plan（规划）: LangGraph create_react_agent
- Action（行动）: LangGraph Runtime
- Tools（工具）: langchain_core.tools.@tool
- Memory（记忆）: langchain.memory
"""

__version__ = "0.3.0"

import agent
import llms
import memories
import tools

__all__ = [
    "agent",
    "llms",
    "memories",
    "tools",
]
