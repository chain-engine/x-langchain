# -*- coding: utf-8 -*-
"""
记忆基类定义

定义记忆系统的抽象接口和数据结构。
兼容 LangChain 标准。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

# LangChain 标准消息类型
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage


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
    兼容 LangChain 消息格式。
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

    def to_langchain_message(self) -> BaseMessage:
        """转换为 LangChain 消息格式"""
        role_map = {
            MessageRole.SYSTEM: SystemMessage,
            MessageRole.USER: HumanMessage,
            MessageRole.ASSISTANT: AIMessage,
            MessageRole.TOOL: ToolMessage,
        }
        msg_class = role_map.get(self.role, HumanMessage)
        return msg_class(content=self.content)

    def to_langchain_format(self) -> dict:
        """转换为 LangChain 消息字典格式"""
        return {
            "role": self.role.value,
            "content": self.content,
        }

    @classmethod
    def from_langchain_message(cls, message: BaseMessage) -> "MemoryMessage":
        """从 LangChain 消息创建"""
        role_map = {
            "system": MessageRole.SYSTEM,
            "human": MessageRole.USER,
            "ai": MessageRole.ASSISTANT,
            "tool": MessageRole.TOOL,
        }
        if hasattr(message, "type"):
            role = role_map.get(message.type, MessageRole.USER)
        else:
            role = MessageRole.USER
        return cls(
            role=role,
            content=message.content if hasattr(message, "content") else str(message),
        )

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
    兼容 LangChain Memory 标准接口。
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

    def add_observation(self, tool_name: str, structured_result: Any) -> None:
        """
        添加观测结果（结构化）

        Args:
            tool_name: 工具名称
            structured_result: 结构化结果对象
        """
        content = f"[{tool_name}] " + str(structured_result)
        metadata = {}
        if hasattr(structured_result, "to_dict"):
            metadata = structured_result.to_dict()
        self.add_tool_message(content=content, tool_name=tool_name, metadata=metadata)

    def add_tool_message(
        self,
        content: str,
        tool_name: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """添加工具消息的便捷方法"""
        msg_metadata = metadata or {}
        if tool_name:
            msg_metadata["tool_name"] = tool_name
        self.add_message(
            MemoryMessage(
                role=MessageRole.TOOL,
                content=content,
                metadata=msg_metadata,
            )
        )

    def __len__(self) -> int:
        """返回记忆中的消息数量"""
        return len(self.get_messages())

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {len(self)} messages>"
