# -*- coding: utf-8 -*-
"""
Agent 相关常量
"""

from __future__ import annotations

from .base import BaseEnum


class AgentMode(BaseEnum):
    """
    Agent 运行模式

    - CHAIN: 简单链式调用，直接调用 LLM
    - REACT: ReAct 推理模式（Thought → Action → Observation 循环）
    - PLAN:  规划模式（预留扩展）
    """
    CHAIN = ("chain", "简单链式调用")
    REACT = ("react", "ReAct 推理模式（工具调用循环）")
    PLAN = ("plan", "规划模式（预留扩展）")


__all__ = ["AgentMode"]
