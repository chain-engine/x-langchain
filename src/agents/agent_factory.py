# -*- coding: utf-8 -*-
"""
Agent 工厂模块

基于 LangGraph 的 create_react_agent API 创建 Agent。
"""

from typing import Any, List, Optional, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent

from core.logger import logger


class AgentFactory:
    """
    Agent 工厂

    基于 LangGraph 的 create_react_agent API 创建各种类型的 Agent。
    """

    @staticmethod
    def create_react_agent(
        llm: BaseChatModel,
        tools: Sequence[BaseTool],
        system_message: Optional[str] = None,
        state_schema: Optional[type] = None,
    ) -> Runnable:
        """
        创建 ReAct Agent (LangGraph 标准实现)

        Args:
            llm: 语言模型
            tools: 工具列表
            system_message: 系统消息
            state_schema: 状态模式（可选）

        Returns:
            CompiledStateGraph (Agent)
        """
        system_prompt = system_message or """你是一个智能助手，可以帮助用户完成各种任务。
当需要外部信息时，使用工具来获取。不要编造信息。"""

        # 使用 LangGraph 的 create_react_agent
        agent = create_react_agent(
            model=llm,
            tools=tools,
            state_schema=state_schema,
            prompt=system_prompt,
            checkpointer=None,  # 如需持久化，可传入 MemorySaver
        )

        logger.info(f"创建 ReAct Agent (LangGraph)，工具数量: {len(tools)}")
        return agent

    @staticmethod
    def create_tool_calling_agent(
        llm: BaseChatModel,
        tools: Sequence[BaseTool],
        system_message: Optional[str] = None,
    ) -> Runnable:
        """
        创建工具调用 Agent (LangGraph 标准实现)

        Args:
            llm: 语言模型
            tools: 工具列表
            system_message: 系统消息

        Returns:
            CompiledStateGraph (Agent)
        """
        system_prompt = system_message or """你是一个智能助手，可以使用工具来完成用户请求。
请根据用户需求选择合适的工具调用。"""

        # 使用 LangGraph 的 create_react_agent（ReAct 本身就支持工具调用）
        agent = create_react_agent(
            model=llm,
            tools=tools,
            prompt=system_prompt,
            checkpointer=None,
        )

        logger.info(f"创建工具调用 Agent (LangGraph)，工具数量: {len(tools)}")
        return agent


__all__ = ["AgentFactory"]
