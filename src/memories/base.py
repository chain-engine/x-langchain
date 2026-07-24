# -*- coding: utf-8 -*-
"""
记忆基类定义

定义记忆系统的抽象接口和数据结构。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class MessageRole(str, Enum):
    """消息角色枚举"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class MemoryMessage:
    """
    记忆消息

    表示记忆中的一个消息单元。
    """
    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    def to_langchain_format(self) -> dict:
        """转换为 LangChain 消息格式"""
        return {
            "role": self.role.value,
            "content": self.content,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryMessage":
        """从字典创建"""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = datetime.now()

        return cls(
            role=MessageRole(data.get("role", "user")),
            content=data.get("content", ""),
            timestamp=timestamp,
            metadata=data.get("metadata", {}),
        )


class BaseMemory(ABC):
    """
    记忆基类

    定义记忆系统的抽象接口。
    所有具体记忆实现都需要继承此类。
    """

    @abstractmethod
    def add_message(self, message: MemoryMessage) -> None:
        """
        添加消息到记忆

        Args:
            message: 记忆消息
        """
        pass

    @abstractmethod
    def get_messages(self, limit: Optional[int] = None) -> list[MemoryMessage]:
        """
        获取记忆中的消息

        Args:
            limit: 限制返回的消息数量，None 表示全部返回

        Returns:
            消息列表
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """清空记忆"""
        pass

    @abstractmethod
    def get_messages_for_llm(self, limit: Optional[int] = None) -> list[dict]:
        """
        获取适合发送给 LLM 的消息格式

        Args:
            limit: 限制返回的消息数量

        Returns:
            LangChain 格式的消息列表
        """
        pass

    def __len__(self) -> int:
        """返回记忆中的消息数量"""
        return len(self.get_messages())

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {len(self)} messages>"
