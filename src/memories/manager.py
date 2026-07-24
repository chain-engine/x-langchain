# -*- coding: utf-8 -*-
"""
记忆管理器

管理多个会话的记忆，支持记忆的聚合和切换。
"""

from typing import Optional

from core.logger import logger

from .base import BaseMemory, MemoryMessage
from .history import ConversationHistoryMemory
from .persistence import PersistentMemory, SQLiteMemoryStore


class MemoryManager:
    """
    记忆管理器

    管理多个会话的记忆，支持：
    - 多会话管理
    - 活动会话切换
    - 记忆聚合查询
    """

    def __init__(
        self,
        max_messages_per_session: Optional[int] = 100,
        enable_persistence: bool = True,
    ):
        """
        初始化记忆管理器

        Args:
            max_messages_per_session: 每个会话最大消息数
            enable_persistence: 是否启用持久化
        """
        self._max_messages = max_messages_per_session
        self._enable_persistence = enable_persistence
        self._sessions: dict[str, BaseMemory] = {}
        self._active_session_id: Optional[str] = None
        self._default_store: Optional[SQLiteMemoryStore] = None

        if enable_persistence:
            self._default_store = SQLiteMemoryStore()

        logger.debug(
            f"初始化记忆管理器，max_messages={max_messages_per_session}, "
            f"persistence={enable_persistence}"
        )

    def create_session(
        self,
        session_id: str,
        system_message: Optional[str] = None,
    ) -> BaseMemory:
        """
        创建新会话记忆

        Args:
            session_id: 会话 ID
            system_message: 系统消息

        Returns:
            创建的记忆对象
        """
        if session_id in self._sessions:
            logger.warning(f"会话 {session_id} 已存在，返回现有记忆")
            return self._sessions[session_id]

        if self._enable_persistence and self._default_store:
            memory = PersistentMemory(
                session_id=session_id,
                store=self._default_store,
                max_messages=self._max_messages,
                system_message=system_message,
            )
        else:
            memory = ConversationHistoryMemory(
                max_messages=self._max_messages,
                system_message=system_message,
            )

        self._sessions[session_id] = memory
        logger.info(f"创建会话记忆: {session_id}")
        return memory

    def get_session(self, session_id: str) -> Optional[BaseMemory]:
        """
        获取会话记忆

        Args:
            session_id: 会话 ID

        Returns:
            记忆对象，不存在则返回 None
        """
        return self._sessions.get(session_id)

    def delete_session(self, session_id: str) -> bool:
        """
        删除会话

        Args:
            session_id: 会话 ID

        Returns:
            是否成功删除
        """
        if session_id not in self._sessions:
            return False

        del self._sessions[session_id]

        if self._enable_persistence and self._default_store:
            self._default_store.delete_session(session_id)

        if self._active_session_id == session_id:
            self._active_session_id = None

        logger.info(f"删除会话记忆: {session_id}")
        return True

    def set_active_session(self, session_id: str) -> bool:
        """
        设置活动会话

        Args:
            session_id: 会话 ID

        Returns:
            是否成功设置
        """
        if session_id not in self._sessions:
            self.create_session(session_id)

        self._active_session_id = session_id
        logger.debug(f"设置活动会话: {session_id}")
        return True

    def get_active_session(self) -> Optional[BaseMemory]:
        """
        获取活动会话记忆

        Returns:
            当前活动会话的记忆对象
        """
        if self._active_session_id is None:
            return None
        return self._sessions.get(self._active_session_id)

    def get_active_session_id(self) -> Optional[str]:
        """获取活动会话 ID"""
        return self._active_session_id

    @property
    def active_session(self) -> Optional[BaseMemory]:
        """活动会话的便捷属性"""
        return self.get_active_session()

    def add_message_to_active(
        self,
        message: MemoryMessage,
        session_id: Optional[str] = None,
    ) -> None:
        """
        添加消息到活动会话

        Args:
            message: 记忆消息
            session_id: 指定会话 ID，None 则使用活动会话
        """
        target_id = session_id or self._active_session_id
        if target_id is None:
            target_id = "default"
            self.create_session(target_id)
            self._active_session_id = target_id

        memory = self._sessions.get(target_id)
        if memory:
            memory.add_message(message)

    def get_all_messages(self) -> list[MemoryMessage]:
        """
        获取所有会话的消息

        Returns:
            所有消息列表
        """
        all_messages: list[MemoryMessage] = []
        for memory in self._sessions.values():
            all_messages.extend(memory.get_messages())
        return all_messages

    def list_sessions(self) -> list[str]:
        """
        列出所有会话 ID

        Returns:
            会话 ID 列表
        """
        return list(self._sessions.keys())

    def __len__(self) -> int:
        """返回会话数量"""
        return len(self._sessions)

    def __contains__(self, session_id: str) -> bool:
        """检查会话是否存在"""
        return session_id in self._sessions
