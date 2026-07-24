# -*- coding: utf-8 -*-
"""
持久化记忆存储

支持将记忆持久化到 SQLite 数据库。
"""

import json
import sqlite3
from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator, Optional

from core.logger import logger

from .base import BaseMemory, MemoryMessage, MessageRole


class BaseMemoryStore(ABC):
    """记忆存储基类"""

    @abstractmethod
    def save_messages(self, session_id: str, messages: list[MemoryMessage]) -> None:
        """保存消息"""
        pass

    @abstractmethod
    def load_messages(self, session_id: str, limit: Optional[int] = None) -> list[MemoryMessage]:
        """加载消息"""
        pass

    @abstractmethod
    def delete_session(self, session_id: str) -> None:
        """删除会话"""
        pass

    @abstractmethod
    def list_sessions(self) -> list[str]:
        """列出所有会话 ID"""
        pass


class SQLiteMemoryStore(BaseMemoryStore):
    """
    SQLite 记忆存储

    使用 SQLite 数据库持久化记忆。
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        初始化 SQLite 记忆存储

        Args:
            db_path: 数据库路径，None 则使用默认路径
        """
        if db_path is None:
            db_path = str(Path(__file__).parent.parent.parent / "memory_store.db")

        self._db_path = db_path
        self._init_db()
        logger.debug(f"初始化 SQLite 记忆存储: {db_path}")

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """获取数据库连接"""
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        """初始化数据库表"""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    metadata TEXT,
                    FOREIGN KEY (session_id) REFERENCES memory_sessions(session_id)
                        ON DELETE CASCADE
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_session
                    ON memory_messages(session_id, timestamp DESC)
            """)
            conn.commit()

    def save_messages(self, session_id: str, messages: list[MemoryMessage]) -> None:
        """
        保存消息到会话

        Args:
            session_id: 会话 ID
            messages: 消息列表
        """
        now = datetime.now().isoformat()

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO memory_sessions (session_id, created_at, updated_at)
                VALUES (?, COALESCE((SELECT created_at FROM memory_sessions WHERE session_id = ?), ?), ?)
                """,
                (session_id, session_id, now, now),
            )

            conn.execute(
                "DELETE FROM memory_messages WHERE session_id = ?",
                (session_id,),
            )

            for msg in messages:
                conn.execute(
                    """
                    INSERT INTO memory_messages (session_id, role, content, timestamp, metadata)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        msg.role.value,
                        msg.content,
                        msg.timestamp.isoformat(),
                        json.dumps(msg.metadata) if msg.metadata else "",
                    ),
                )

            conn.commit()
        logger.debug(f"保存 {len(messages)} 条消息到会话 {session_id}")

    def load_messages(self, session_id: str, limit: Optional[int] = None) -> list[MemoryMessage]:
        """
        加载会话消息

        Args:
            session_id: 会话 ID
            limit: 限制返回数量

        Returns:
            消息列表
        """
        with self._get_connection() as conn:
            query = "SELECT * FROM memory_messages WHERE session_id = ? ORDER BY timestamp ASC"
            if limit:
                query += f" LIMIT {limit}"

            rows = conn.execute(query, (session_id,)).fetchall()

        messages = []
        for row in rows:
            metadata = {}
            if row["metadata"]:
                try:
                    metadata = json.loads(row["metadata"])
                except json.JSONDecodeError:
                    pass

            messages.append(
                MemoryMessage(
                    role=MessageRole(row["role"]),
                    content=row["content"],
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    metadata=metadata,
                )
            )

        logger.debug(f"从会话 {session_id} 加载了 {len(messages)} 条消息")
        return messages

    def delete_session(self, session_id: str) -> None:
        """删除会话"""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM memory_messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM memory_sessions WHERE session_id = ?", (session_id,))
            conn.commit()
        logger.debug(f"删除会话 {session_id}")

    def list_sessions(self) -> list[str]:
        """列出所有会话 ID"""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT session_id FROM memory_sessions ORDER BY updated_at DESC"
            ).fetchall()
        return [row["session_id"] for row in rows]


class PersistentMemory(BaseMemory):
    """
    持久化记忆

    将内存记忆与持久化存储结合，支持：
    - 自动保存和加载
    - 会话隔离
    - 记忆持久化
    """

    def __init__(
        self,
        session_id: str,
        store: Optional[BaseMemoryStore] = None,
        max_messages: Optional[int] = None,
        system_message: Optional[str] = None,
        auto_save: bool = True,
    ):
        """
        初始化持久化记忆

        Args:
            session_id: 会话 ID
            store: 记忆存储，None 则使用默认 SQLite 存储
            max_messages: 最大保存消息数
            system_message: 系统消息
            auto_save: 是否自动保存
        """
        from .history import ConversationHistoryMemory

        self._session_id = session_id
        self._store = store or SQLiteMemoryStore()
        self._max_messages = max_messages
        self._auto_save = auto_save

        self._memory = ConversationHistoryMemory(
            max_messages=max_messages,
            system_message=system_message,
        )

        self._load_from_store()
        logger.debug(f"初始化持久化记忆会话: {session_id}")

    def _load_from_store(self) -> None:
        """从存储加载记忆"""
        messages = self._store.load_messages(self._session_id)
        for msg in messages:
            self._memory.add_message(msg)

    def _save_to_store(self) -> None:
        """保存记忆到存储"""
        if self._auto_save:
            self._store.save_messages(self._session_id, self._memory.get_messages())

    def add_message(self, message: MemoryMessage) -> None:
        """添加消息并保存"""
        self._memory.add_message(message)
        self._save_to_store()

    def get_messages(self, limit: Optional[int] = None) -> list[MemoryMessage]:
        """获取消息"""
        return self._memory.get_messages(limit=limit)

    def clear(self) -> None:
        """清空记忆"""
        self._memory.clear()
        self._save_to_store()

    def get_messages_for_llm(self, limit: Optional[int] = None) -> list[dict]:
        """获取适合 LLM 的消息格式"""
        return self._memory.get_messages_for_llm(limit=limit)

    def set_system_message(self, content: str) -> None:
        """设置系统消息"""
        self._memory.clear()
        self._memory.add_message(MemoryMessage(role=MessageRole.SYSTEM, content=content))
        self._save_to_store()

    @property
    def session_id(self) -> str:
        """获取会话 ID"""
        return self._session_id

    def save(self) -> None:
        """手动保存"""
        self._save_to_store()
