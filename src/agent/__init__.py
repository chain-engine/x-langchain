# -*- coding: utf-8 -*-
"""
Agent 模块

整合 LLM、Memory、Planning、Action、Tools 五大核心子系统。
"""

from core.config import AgentConfig

from .agent import Agent, AgentResponse


def create_agent_config(**kwargs) -> AgentConfig:
    """创建 Agent 配置的便捷函数"""
    return AgentConfig(**kwargs)


__all__ = [
    "Agent",
    "AgentConfig",
    "AgentResponse",
    "create_agent_config",
]
