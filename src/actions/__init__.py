# -*- coding: utf-8 -*-
"""
行动（Action）模块

提供行动调度功能：
- 行动基类
- 行动执行器
- 工具调用行动
- 直接回复行动

设计原则：
- 策略模式：支持多种行动类型
- 工厂模式：灵活创建行动
- 组合模式：行动可以组合
"""

from .base import BaseAction, ActionResult, ActionType
from .dispatcher import ActionDispatcher
from .executors import (
    DirectResponseAction,
    ToolCallAction,
    CompoundAction,
    ActionExecutor,
)

__all__ = [
    "BaseAction",
    "ActionResult",
    "ActionType",
    "ActionDispatcher",
    "DirectResponseAction",
    "ToolCallAction",
    "CompoundAction",
    "ActionExecutor",
]
