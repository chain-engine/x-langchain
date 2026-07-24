# -*- coding: utf-8 -*-
"""
Agent 相关常量
"""

from .base import BaseEnum


class AgentMode(BaseEnum):
    """Agent 运行模式"""
    CHAIN = ("chain", "简单链式调用")
    REACT = ("react", "ReAct 推理模式")
    PLAN = ("plan", "规划模式")


__all__ = ["AgentMode"]
