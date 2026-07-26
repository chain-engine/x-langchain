# -*- coding: utf-8 -*-
"""
Agent 模块 - 基于 LangChain 的 Agent 组件

提供基于 LangGraph create_react_agent 的单 Agent 实现（ReAct 推理模式）。
"""

from agent.chat_history_service import (
    ChatHistoryService,
    chat_history_context,
    create_chat_history_service,
    generate_session_id,
)
from agent.lc_agent import LCAgent, AgentResponse

__all__ = [
    "LCAgent",
    "AgentResponse",
    "ChatHistoryService",
    "create_chat_history_service",
    "chat_history_context",
    "generate_session_id",
]
