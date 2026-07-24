# -*- coding: utf-8 -*-
"""
x-langchain - LangChain 学习与实践项目

这是一个完整的 LangChain（LLM应用开发框架）学习与实践项目。

核心模块：
|- llms: 大语言模型提供者
|- memories: 记忆管理
|- planning: 任务规划
|- actions: 行动调度
|- agent: Agent 核心
|- tools: 工具系统
"""

__version__ = "0.2.0"

from . import agent
from . import llms
from . import memories
from . import planning
from . import actions
from . import tools

__all__ = [
    "agent",
    "llms",
    "memories",
    "planning",
    "actions",
    "tools",
]
