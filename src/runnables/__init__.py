# -*- coding: utf-8 -*-
"""
Runnables 模块 - LCEL 表达式语言

提供 LangChain LCEL（LangChain Expression Language）封装：
- RunnableLambda: Python 函数 → Runnable
- RunnablePassthrough: 透传输入
- ConfigurableLLM: 动态切换 LLM
- create_pipe_chain: 管道组合链
- AsyncAgent: 异步 Agent 封装
"""

from .async_agent import AsyncLCAgent, create_async_agent
from .configurable import ConfigurableLLM, configurable_llm
from .routines import (
    RunnableLambda,
    RunnablePassthrough,
    create_pipe_chain,
    pipe_chain,
)

__all__ = [
    "RunnableLambda",
    "RunnablePassthrough",
    "create_pipe_chain",
    "pipe_chain",
    "ConfigurableLLM",
    "configurable_llm",
    "AsyncLCAgent",
    "create_async_agent",
]
