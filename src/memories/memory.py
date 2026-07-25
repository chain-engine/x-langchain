# -*- coding: utf-8 -*-
"""
LangChain Memory 模块封装

基于 LangChain Core 的 Memory 组件，提供统一的记忆接口。
使用 langchain_core.chat_history 和 RunnableWithMessageHistory。
"""

from typing import Any, Callable, List, Optional, Sequence

from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import Runnable

from core.logger import logger


class ChatMessageHistory(InMemoryChatMessageHistory):
    """
    聊天消息历史

    基于 LangChain Core 的 InMemoryChatMessageHistory 实现。
    """

    def __init__(self):
        super().__init__()

    def add_user_message(self, message: str) -> None:
        """添加用户消息"""
        self.add_message(HumanMessage(content=message))

    def add_ai_message(self, message: str) -> None:
        """添加 AI 消息"""
        self.add_message(AIMessage(content=message))


class ConversationMemory:
    """
    对话记忆

    基于 LangChain Core 封装的对话记忆组件。
    支持消息历史管理和上下文窗口限制。
    """

    def __init__(
        self,
        max_messages: Optional[int] = None,
        return_messages: bool = True,
    ):
        """
        初始化对话记忆

        Args:
            max_messages: 最大保存消息数，None 表示不限制
            return_messages: 是否返回消息列表
        """
        self._chat_history = ChatMessageHistory()
        self._max_messages = max_messages
        self._return_messages = return_messages

    def add_user_message(self, message: str) -> None:
        """添加用户消息"""
        self._chat_history.add_user_message(message)
        self._trim_history()

    def add_ai_message(self, message: str) -> None:
        """添加 AI 消息"""
        self._chat_history.add_ai_message(message)
        self._trim_history()

    def add_message(self, role: str, content: str) -> None:
        """添加消息（指定角色）"""
        if role.lower() == "user":
            self.add_user_message(content)
        elif role.lower() == "assistant":
            self.add_ai_message(content)
        else:
            self._chat_history.add_message(HumanMessage(content=content))

    def get_messages(self) -> List[BaseMessage]:
        """获取所有消息"""
        return self._chat_history.messages

    def clear(self) -> None:
        """清空记忆"""
        self._chat_history.clear()

    def _trim_history(self) -> None:
        """裁剪历史消息"""
        if self._max_messages is None:
            return

        messages = self._chat_history.messages
        if len(messages) > self._max_messages:
            # 保留最后 N 条消息
            self._chat_history.messages = messages[-self._max_messages:]

    def load_memory_variables(self, inputs: dict) -> dict:
        """加载记忆变量（供 Chain 使用）"""
        return {"chat_history": self.get_messages()}

    def save_context(self, inputs: dict, outputs: dict) -> None:
        """保存上下文"""
        user_input = inputs.get("input", "")
        ai_output = outputs.get("output", "")

        if user_input:
            self.add_user_message(user_input)
        if ai_output:
            self.add_ai_message(ai_output)


class BufferMemory:
    """
    Buffer Memory

    基于 LangChain Core 的缓冲区记忆，自动管理上下文。
    """

    def __init__(
        self,
        chat_memory: Optional[BaseChatMessageHistory] = None,
        return_messages: bool = True,
    ):
        """
        初始化 Buffer Memory

        Args:
            chat_memory: 聊天历史记录
            return_messages: 返回消息列表还是字符串
        """
        self._chat_memory = chat_memory or InMemoryChatMessageHistory()
        self._return_messages = return_messages

    @property
    def chat_memory(self) -> BaseChatMessageHistory:
        """获取聊天历史"""
        return self._chat_memory

    def save_context(self, inputs: dict, outputs: dict) -> None:
        """保存上下文"""
        user_input = inputs.get("input", "")
        ai_output = outputs.get("output", "")

        if user_input:
            self._chat_memory.add_message(HumanMessage(content=user_input))
        if ai_output:
            self._chat_memory.add_message(AIMessage(content=ai_output))

    def load_memory_variables(self, inputs: dict) -> dict:
        """加载记忆变量"""
        messages = self._chat_memory.messages
        if self._return_messages:
            return {"history": messages}
        else:
            return {"history": "\n".join(str(m.content) for m in messages)}

    def clear(self) -> None:
        """清空记忆"""
        self._chat_memory.clear()


def create_conversation_memory(max_messages: int = 50) -> ConversationMemory:
    """创建对话记忆"""
    return ConversationMemory(max_messages=max_messages)


def create_buffer_memory() -> BufferMemory:
    """创建缓冲区记忆"""
    return BufferMemory()


__all__ = [
    "ChatMessageHistory",
    "ConversationMemory",
    "BufferMemory",
    "create_conversation_memory",
    "create_buffer_memory",
]
