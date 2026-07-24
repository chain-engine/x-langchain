# -*- coding: utf-8 -*-
"""
规划（Planning）模块

提供任务规划和分解功能：
- 基础规划接口
- 子任务定义
- 任务执行器
- 规划管理器

设计原则：
- 面向对象：使用抽象基类定义接口
- 策略模式：支持多种规划策略
- 链式调用：支持任务依赖关系
"""

from .base import BasePlanner, Task, TaskResult, TaskStatus
from .executor import TaskExecutor, SequentialExecutor, ParallelExecutor
from .manager import PlanningManager
from .strategies import LLMPlanner, SimplePlanner

__all__ = [
    "BasePlanner",
    "Task",
    "TaskResult",
    "TaskStatus",
    "TaskExecutor",
    "SequentialExecutor",
    "ParallelExecutor",
    "PlanningManager",
    "LLMPlanner",
    "SimplePlanner",
]
