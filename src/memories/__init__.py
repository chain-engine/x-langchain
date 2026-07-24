# -*- coding: utf-8 -*-
"""
记忆（Memory）模块

提供多层次的记忆管理功能：
- 基础记忆接口
- 对话历史记忆
- 持久化记忆存储
- 记忆管理器

设计原则：
- 面向对象：使用抽象基类定义接口
- 组合优于继承：核心功能通过组合实现
- 单一职责：每个类只负责一种记忆类型
"""

from .base import BaseMemory, MemoryMessage
from .history import ConversationHistoryMemory
from .persistence import PersistentMemory, SQLiteMemoryStore
from .manager import MemoryManager

__all__ = [
    "BaseMemory",
    "MemoryMessage",
    "ConversationHistoryMemory",
    "PersistentMemory",
    "SQLiteMemoryStore",
    "MemoryManager",
]
