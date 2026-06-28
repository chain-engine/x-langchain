# -*- coding: utf-8 -*-
"""
Agent 工厂模块

用于创建和配置 LangChain Agent 实例。
"""

import os
from typing import Any, List, Literal

from core.config import settings
from models import create_chat_model
from tools import get_all_tools, get_all_tools_async

ProviderName = Literal["deepseek", "doubao", "aliyun", "tongyi", "mock"]


class AgentFactory:
    """
    Agent 工厂类，用于创建不同的 Agent 实例
    """

    @staticmethod
    def get_default_tools() -> List[Any]:
        """
        获取默认工具列表（从 ToolRegistry 动态加载）

        Returns:
            工具列表
        """
        return get_all_tools()

    @staticmethod
    async def get_default_tools_async() -> List[Any]:
        """
        异步获取默认工具列表

        Returns:
            工具列表
        """
        return await get_all_tools_async()

    @staticmethod
    def create_agent(
        model_name: ProviderName,
        tools: list[Any] | None = None,
    ) -> Any:
        """
        根据提供者名称创建对应的 Agent 实例

        Args:
            model_name: 提供者名称，支持 'deepseek', 'doubao', 'aliyun', 'tongyi', 'mock'
            tools: 工具列表（如果为 None，则使用 ToolRegistry 中的所有工具）

        Returns:
            创建的 Agent 实例
        """

        # 创建模型
        model: Any = create_chat_model(model_name)

        # 创建 Agent
        return AgentFactory._create_agent_instance(model, tools)

    @staticmethod
    def _create_agent_instance(
        model: Any,
        tools: list[Any] | None = None,
    ) -> Any:
        """
        创建 Agent 实例

        Args:
            model: 模型实例
            tools: 工具列表（如果为 None，则使用 ToolRegistry 中的所有工具）

        Returns:
            Agent 实例
        """
        from langchain.agents import create_agent

        # 使用传入的工具列表或从 ToolRegistry 获取所有工具
        tools_list: list[Any] = tools if tools is not None else AgentFactory.get_default_tools()

        # 根据配置中的结构化输出设置，决定是否使用结构化输出
        structured: bool = os.getenv("STRUCTURED", str(settings.STRUCTURED)).lower() == "true"

        # 根据是否结构化输出，设置不同的系统提示
        system_prompt: str
        if structured:
            system_prompt = """你是一个可以使用工具的智能助手。

当用户需要实时信息或外部数据时，优先调用工具。使用工具后，请清晰总结工具结果，
不要编造事实。遇到数据库问题时，请遵循 TextToSQL 流程：改写问题、查看表结构、
生成 SQL、校验 SQL、执行 SQL，然后用自然语言解释结果。
"""
        else:
            system_prompt = """你是一个可以使用工具的智能助手。

当用户需要实时信息或外部数据时，优先调用工具。使用工具后，请清晰总结工具结果，
不要编造事实。遇到数据库问题时，请遵循 TextToSQL 流程：改写问题、查看表结构、
生成 SQL、校验 SQL、执行 SQL，然后用自然语言解释结果。
"""

        # LangChain 官方推荐：直接传入 LLM / ChatModel 实例和工具列表创建 Agent
        return create_agent(
            model=model,
            tools=tools_list,
            system_prompt=system_prompt,
            debug=False,  # 禁用详细输出
        )


# 创建全局 Agent 工厂实例
agent_factory: AgentFactory = AgentFactory()