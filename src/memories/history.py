# -*- coding: utf-8 -*-
"""
对话历史记忆

实现基于内存的对话历史管理。
"""

from typing import Optional

from core.logger import logger

from .base import BaseMemory, MemoryMessage, MessageRole


class ConversationHistoryMemory(BaseMemory):
    """
    对话历史记忆

    基于内存的对话历史存储，支持：
    - 消息添加和检索
    - 自动摘要（可选）
    - 对话上下文窗口限制
    """

    def __init__(
        self,
        max_messages: Optional[int] = None,
        system_message: Optional[str] = None,
    ):
        """
        初始化对话历史记忆

        Args:
            max_messages: 最大保存的消息数量，None 表示不限制
            system_message: 系统消息内容（如果有）
        """
        self._messages: list[MemoryMessage] = []
        self._max_messages = max_messages

        if system_message:
            self._messages.append(
                MemoryMessage(
                    role=MessageRole.SYSTEM,
                    content=system_message,
                )
            )

        logger.debug(f"初始化对话历史记忆，max_messages={max_messages}")

    def add_message(self, message: MemoryMessage) -> None:
        """
        添加消息到记忆

        Args:
            message: 记忆消息
        """
        self._messages.append(message)
        logger.debug(f"添加消息到记忆，当前共 {len(self._messages)} 条消息")

        if self._max_messages and len(self._messages) > self._max_messages:
            self._trim_old_messages()

    def _trim_old_messages(self) -> None:
        """裁剪旧消息（保留系统消息）"""
        if not self._messages:
            return

        system_messages = [m for m in self._messages if m.role == MessageRole.SYSTEM]
        non_system_messages = [m for m in self._messages if m.role != MessageRole.SYSTEM]

        keep_count = self._max_messages - len(system_messages)
        if keep_count > 0:
            self._messages = system_messages + non_system_messages[-keep_count:]
        else:
            self._messages = system_messages[-self._max_messages:]

        logger.debug(f"裁剪消息后，当前共 {len(self._messages)} 条消息")

    def get_messages(self, limit: Optional[int] = None) -> list[MemoryMessage]:
        """
        获取记忆中的消息

        Args:
            limit: 限制返回的消息数量

        Returns:
            消息列表
        """
        if limit is None:
            return list(self._messages)

        system_messages = [m for m in self._messages if m.role == MessageRole.SYSTEM]
        non_system_messages = [m for m in self._messages if m.role != MessageRole.SYSTEM]

        return system_messages + non_system_messages[-limit:]

    def get_recent_messages(self, count: int) -> list[MemoryMessage]:
        """
        获取最近的消息

        Args:
            count: 要获取的消息数量

        Returns:
            最近的消息列表
        """
        return self.get_messages(limit=count)

    def clear(self) -> None:
        """清空记忆"""
        system_messages = [m for m in self._messages if m.role == MessageRole.SYSTEM]
        self._messages = system_messages
        logger.debug("清空对话历史记忆")

    def get_messages_for_llm(self, limit: Optional[int] = None) -> list[dict]:
        """
        获取适合发送给 LLM 的消息格式

        Args:
            limit: 限制返回的消息数量

        Returns:
            LangChain 格式的消息列表
        """
        messages = self.get_messages(limit=limit)
        return [msg.to_langchain_format() for msg in messages]

    @property
    def is_empty(self) -> bool:
        """检查记忆是否为空（排除系统消息）"""
        return all(m.role == MessageRole.SYSTEM for m in self._messages)

    def add_user_message(self, content: str) -> None:
        """快捷方法：添加用户消息"""
        self.add_message(MemoryMessage(role=MessageRole.USER, content=content))

    def add_assistant_message(self, content: str) -> None:
        """快捷方法：添加助手消息"""
        self.add_message(MemoryMessage(role=MessageRole.ASSISTANT, content=content))

    def add_tool_message(self, content: str, tool_name: Optional[str] = None) -> None:
        """快捷方法：添加工具消息"""
        metadata = {}
        if tool_name:
            metadata["tool_name"] = tool_name
        self.add_message(
            MemoryMessage(
                role=MessageRole.TOOL,
                content=content,
                metadata=metadata,
            )
        )
