# -*- coding: utf-8 -*-
"""
Chat History 实现模块

提供 LangChain 原生的多种 Chat History 持久化实现：
- RedisChatHistory: Redis 存储
- FileChatHistory: 本地文件存储
- PostgresChatHistory: PostgreSQL 存储
- MongoDBChatHistory: MongoDB 存储
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from core.logger import logger


# =============================================================================
# 基础接口
# =============================================================================


class ChatHistoryAdapter(BaseChatMessageHistory, ABC):
    """
    Chat History 存储适配器基类

    所有存储后端需继承此类并实现 save/load/clear 方法。
    统一接口，便于在不同存储之间切换。
    """

    @abstractmethod
    def save_messages(self, messages: list[BaseMessage]) -> None:
        """保存消息列表"""
        raise NotImplementedError

    @abstractmethod
    def load_messages(self) -> list[BaseMessage]:
        """加载所有消息"""
        raise NotImplementedError

    @abstractmethod
    def clear_messages(self) -> None:
        """清空所有消息"""
        raise NotImplementedError

    def add_user_message(self, message: str) -> None:
        """添加用户消息"""
        self.add_message(HumanMessage(content=message))

    def add_ai_message(self, message: str) -> None:
        """添加 AI 消息"""
        self.add_message(AIMessage(content=message))

    def add_message(self, message: BaseMessage) -> None:
        """添加单条消息"""
        messages = self.messages
        messages.append(message)
        self.save_messages(messages)

    @property
    def messages(self) -> list[BaseMessage]:
        """获取所有消息"""
        return self.load_messages()

    def clear(self) -> None:
        """清空消息"""
        self.clear_messages()


# =============================================================================
# Redis Chat History
# =============================================================================


class RedisChatHistory(ChatHistoryAdapter):
    """
    Redis Chat History

    基于 Redis 的对话历史存储，支持过期时间和自动清理。
    适用于分布式部署和高并发场景。

    使用方式：
        ```python
        history = RedisChatHistory(
            session_id="user-123",
            redis_url="redis://localhost:6379/0",
            ttl=3600,  # 1小时过期
        )
        history.add_user_message("你好")
        history.add_ai_message("你好，有什么可以帮助你的？")
        messages = history.messages
        ```
    """

    def __init__(
        self,
        session_id: str,
        redis_url: str = "redis://localhost:6379/0",
        ttl: Optional[int] = None,
        key_prefix: str = "chat:",
    ):
        """
        初始化 Redis Chat History

        Args:
            session_id: 会话唯一标识
            redis_url: Redis 连接 URL
            ttl: 过期时间（秒），None 表示不过期
            key_prefix: Redis key 前缀
        """
        self._session_id = session_id
        self._redis_url = redis_url
        self._ttl = ttl
        self._key_prefix = key_prefix
        self._redis: Optional[Any] = None

    @property
    def session_id(self) -> str:
        return self._session_id

    def _get_redis(self) -> Any:
        """获取或创建 Redis 连接"""
        if self._redis is None:
            try:
                import redis
            except ImportError as exc:
                raise ImportError(
                    "请安装 redis: pip install redis"
                ) from exc

            try:
                self._redis = redis.from_url(self._redis_url)
                self._redis.ping()
            except Exception as exc:
                raise ConnectionError(f"无法连接到 Redis: {exc}") from exc
        return self._redis

    def _get_key(self) -> str:
        """获取 Redis key"""
        return f"{self._key_prefix}{self._session_id}"

    def save_messages(self, messages: list[BaseMessage]) -> None:
        """保存消息到 Redis"""
        try:
            r = self._get_redis()
            key = self._get_key()

            # 序列化消息
            serialized = self._serialize_messages(messages)
            r.set(key, json.dumps(serialized, ensure_ascii=False))

            # 设置过期时间
            if self._ttl:
                r.expire(key, self._ttl)

            logger.debug(f"RedisChatHistory: 保存 {len(messages)} 条消息到 {key}")

        except Exception as e:
            logger.error(f"RedisChatHistory 保存失败: {e}")
            raise

    def load_messages(self) -> list[BaseMessage]:
        """从 Redis 加载消息"""
        try:
            r = self._get_redis()
            key = self._get_key()

            data = r.get(key)
            if not data:
                return []

            serialized = json.loads(data)
            messages = self._deserialize_messages(serialized)
            logger.debug(f"RedisChatHistory: 加载 {len(messages)} 条消息")
            return messages

        except Exception as e:
            logger.error(f"RedisChatHistory 加载失败: {e}")
            return []

    def clear_messages(self) -> None:
        """清空 Redis 中的消息"""
        try:
            r = self._get_redis()
            key = self._get_key()
            r.delete(key)
            logger.debug(f"RedisChatHistory: 清空会话 {self._session_id}")
        except Exception as e:
            logger.error(f"RedisChatHistory 清空失败: {e}")
            raise

    @staticmethod
    def _serialize_messages(messages: list[BaseMessage]) -> list[dict]:
        """序列化消息列表"""
        result = []
        for msg in messages:
            msg_dict = {
                "type": msg.type,
                "content": msg.content,
            }
            # 保存额外属性
            if hasattr(msg, "additional_kwargs"):
                msg_dict["additional_kwargs"] = msg.additional_kwargs
            if hasattr(msg, "response_metadata"):
                msg_dict["response_metadata"] = msg.response_metadata
            result.append(msg_dict)
        return result

    @staticmethod
    def _deserialize_messages(data: list[dict]) -> list[BaseMessage]:
        """反序列化消息列表"""
        messages = []
        for msg_dict in data:
            msg_type = msg_dict.get("type", "human")
            content = msg_dict.get("content", "")

            if msg_type == "human" or msg_type == "user":
                msg = HumanMessage(content=content)
            elif msg_type == "ai" or msg_type == "assistant":
                msg = AIMessage(content=content)
            elif msg_type == "system":
                msg = SystemMessage(content=content)
            else:
                msg = HumanMessage(content=content)

            # 恢复额外属性
            if "additional_kwargs" in msg_dict:
                msg.additional_kwargs = msg_dict["additional_kwargs"]
            if "response_metadata" in msg_dict:
                msg.response_metadata = msg_dict["response_metadata"]

            messages.append(msg)

        return messages


# =============================================================================
# 文件存储 Chat History
# =============================================================================


class FileChatHistory(ChatHistoryAdapter):
    """
    文件存储 Chat History

    将对话历史保存到本地 JSON 文件，简单可靠。
    适用于单机部署或测试环境。

    使用方式：
        ```python
        history = FileChatHistory(
            session_id="user-123",
            file_path="./data/chat_history/",
        )
        history.add_user_message("你好")
        ```
    """

    def __init__(
        self,
        session_id: str,
        file_path: str = "./data/chat_history",
        file_ext: str = ".json",
        auto_mkdir: bool = True,
    ):
        """
        初始化文件存储 Chat History

        Args:
            session_id: 会话唯一标识
            file_path: 文件存储目录
            file_ext: 文件扩展名
            auto_mkdir: 是否自动创建目录
        """
        self._session_id = session_id
        self._file_path = Path(file_path)
        self._file_ext = file_ext
        self._auto_mkdir = auto_mkdir

        if self._auto_mkdir:
            self._file_path.mkdir(parents=True, exist_ok=True)

    @property
    def session_id(self) -> str:
        return self._session_id

    def _get_file_path(self) -> Path:
        """获取文件路径"""
        return self._file_path / f"{self._session_id}{self._file_ext}"

    def save_messages(self, messages: list[BaseMessage]) -> None:
        """保存消息到文件"""
        try:
            file_path = self._get_file_path()

            serialized = self._serialize_messages(messages)
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(serialized, f, ensure_ascii=False, indent=2)

            logger.debug(f"FileChatHistory: 保存 {len(messages)} 条消息到 {file_path}")

        except Exception as e:
            logger.error(f"FileChatHistory 保存失败: {e}")
            raise

    def load_messages(self) -> list[BaseMessage]:
        """从文件加载消息"""
        try:
            file_path = self._get_file_path()

            if not file_path.exists():
                return []

            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            messages = self._deserialize_messages(data)
            logger.debug(f"FileChatHistory: 加载 {len(messages)} 条消息")
            return messages

        except Exception as e:
            logger.error(f"FileChatHistory 加载失败: {e}")
            return []

    def clear_messages(self) -> None:
        """清空文件中的消息"""
        try:
            file_path = self._get_file_path()
            if file_path.exists():
                file_path.unlink()
            logger.debug(f"FileChatHistory: 清空会话 {self._session_id}")
        except Exception as e:
            logger.error(f"FileChatHistory 清空失败: {e}")
            raise

    @staticmethod
    def _serialize_messages(messages: list[BaseMessage]) -> list[dict]:
        """序列化消息列表"""
        result = []
        for msg in messages:
            msg_dict = {
                "type": msg.type,
                "content": msg.content,
            }
            if hasattr(msg, "additional_kwargs"):
                msg_dict["additional_kwargs"] = msg.additional_kwargs
            if hasattr(msg, "name"):
                msg_dict["name"] = msg.name
            result.append(msg_dict)
        return result

    @staticmethod
    def _deserialize_messages(data: list[dict]) -> list[BaseMessage]:
        """反序列化消息列表"""
        messages = []
        for msg_dict in data:
            msg_type = msg_dict.get("type", "human")
            content = msg_dict.get("content", "")

            if msg_type == "human" or msg_type == "user":
                msg = HumanMessage(content=content)
            elif msg_type == "ai" or msg_type == "assistant":
                msg = AIMessage(content=content)
            elif msg_type == "system":
                msg = SystemMessage(content=content)
            else:
                msg = HumanMessage(content=content)

            if "additional_kwargs" in msg_dict:
                msg.additional_kwargs = msg_dict["additional_kwargs"]

            messages.append(msg)

        return messages


# =============================================================================
# PostgreSQL Chat History
# =============================================================================


class PostgresChatHistory(ChatHistoryAdapter):
    """
    PostgreSQL Chat History

    基于 PostgreSQL 的对话历史存储，支持异步操作。
    适用于需要强一致性保证的生产环境。

    使用方式：
        ```python
        history = PostgresChatHistory(
            session_id="user-123",
            connection_string="postgresql://user:pass@localhost:5432/chat",
            table_name="chat_messages",
        )
        history.add_user_message("你好")
        ```
    """

    def __init__(
        self,
        session_id: str,
        connection_string: str,
        table_name: str = "chat_messages",
    ):
        """
        初始化 PostgreSQL Chat History

        Args:
            session_id: 会话唯一标识
            connection_string: PostgreSQL 连接字符串
            table_name: 表名
        """
        self._session_id = session_id
        self._connection_string = connection_string
        self._table_name = table_name
        self._engine: Optional[Any] = None

        # 确保表存在
        self._ensure_table()

    @property
    def session_id(self) -> str:
        return self._session_id

    def _get_engine(self) -> Any:
        """获取或创建数据库引擎"""
        if self._engine is None:
            from sqlalchemy import create_engine

            self._engine = create_engine(self._connection_string)
        return self._engine

    def _ensure_table(self) -> None:
        """确保表存在"""
        try:
            engine = self._get_engine()
            from sqlalchemy import Column, String, Text, DateTime, Table, MetaData

            metadata = MetaData()
            table = Table(
                self._table_name,
                metadata,
                Column("id", String(50), primary_key=True),
                Column("session_id", String(100), primary_key=True, index=True),
                Column("message_type", String(20)),
                Column("content", Text),
                Column("created_at", DateTime, default=datetime.now),
            )
            metadata.create_all(engine)
            logger.debug(f"PostgresChatHistory: 确保表 {self._table_name} 存在")

        except Exception as e:
            logger.error(f"PostgresChatHistory 创建表失败: {e}")
            raise

    def save_messages(self, messages: list[BaseMessage]) -> None:
        """保存消息到 PostgreSQL"""
        try:
            from sqlalchemy import delete
            from sqlalchemy.orm import sessionmaker

            engine = self._get_engine()
            Session = sessionmaker(bind=engine)
            session = Session()

            # 清空旧消息
            session.execute(
                delete(self._get_message_table()).where(
                    self._get_message_table().c.session_id == self._session_id
                )
            )

            # 插入新消息
            for i, msg in enumerate(messages):
                session.execute(
                    self._get_message_table().insert().values(
                        id=f"{self._session_id}_{i}",
                        session_id=self._session_id,
                        message_type=msg.type,
                        content=msg.content,
                        created_at=datetime.now(),
                    )
                )

            session.commit()
            session.close()
            logger.debug(f"PostgresChatHistory: 保存 {len(messages)} 条消息")

        except Exception as e:
            logger.error(f"PostgresChatHistory 保存失败: {e}")
            raise

    def load_messages(self) -> list[BaseMessage]:
        """从 PostgreSQL 加载消息"""
        try:
            from sqlalchemy import select

            engine = self._get_engine()
            from sqlalchemy.orm import sessionmaker

            Session = sessionmaker(bind=engine)
            session = Session()

            stmt = select(
                self._get_message_table()
            ).where(
                self._get_message_table().c.session_id == self._session_id
            ).order_by(self._get_message_table().c.created_at.asc())

            rows = session.execute(stmt).fetchall()
            session.close()

            messages = []
            for row in rows:
                if row.message_type in ("human", "user"):
                    msg = HumanMessage(content=row.content)
                elif row.message_type in ("ai", "assistant"):
                    msg = AIMessage(content=row.content)
                elif row.message_type == "system":
                    msg = SystemMessage(content=row.content)
                else:
                    msg = HumanMessage(content=row.content)
                messages.append(msg)

            logger.debug(f"PostgresChatHistory: 加载 {len(messages)} 条消息")
            return messages

        except Exception as e:
            logger.error(f"PostgresChatHistory 加载失败: {e}")
            return []

    def clear_messages(self) -> None:
        """清空 PostgreSQL 中的消息"""
        try:
            from sqlalchemy import delete
            from sqlalchemy.orm import sessionmaker

            engine = self._get_engine()
            Session = sessionmaker(bind=engine)
            session = Session()

            session.execute(
                delete(self._get_message_table()).where(
                    self._get_message_table().c.session_id == self._session_id
                )
            )
            session.commit()
            session.close()
            logger.debug(f"PostgresChatHistory: 清空会话 {self._session_id}")

        except Exception as e:
            logger.error(f"PostgresChatHistory 清空失败: {e}")
            raise

    def _get_message_table(self) -> Any:
        """获取消息表对象"""
        from sqlalchemy import Column, String, Text, DateTime, Table, MetaData

        return Table(
            self._table_name,
            MetaData(),
            Column("id", String(50), primary_key=True),
            Column("session_id", String(100), primary_key=True, index=True),
            Column("message_type", String(20)),
            Column("content", Text),
            Column("created_at", DateTime, default=datetime.now),
        )


# =============================================================================
# MongoDB Chat History
# =============================================================================


class MongoDBChatHistory(ChatHistoryAdapter):
    """
    MongoDB Chat History

    基于 MongoDB 的对话历史存储，支持文档嵌套和灵活查询。
    适用于需要存储额外元数据的场景。

    使用方式：
        ```python
        history = MongoDBChatHistory(
            session_id="user-123",
            connection_string="mongodb://localhost:27017",
            database="chat_db",
            collection="messages",
        )
        history.add_user_message("你好")
        ```
    """

    def __init__(
        self,
        session_id: str,
        connection_string: str = "mongodb://localhost:27017",
        database: str = "chat_db",
        collection: str = "messages",
    ):
        """
        初始化 MongoDB Chat History

        Args:
            session_id: 会话唯一标识
            connection_string: MongoDB 连接字符串
            database: 数据库名
            collection: 集合名
        """
        self._session_id = session_id
        self._connection_string = connection_string
        self._database_name = database
        self._collection_name = collection
        self._client: Optional[Any] = None

    @property
    def session_id(self) -> str:
        return self._session_id

    def _get_collection(self) -> Any:
        """获取 MongoDB 集合"""
        if self._client is None:
            try:
                from pymongo import MongoClient
            except ImportError as exc:
                raise ImportError(
                    "请安装 pymongo: pip install pymongo"
                ) from exc

            try:
                self._client = MongoClient(self._connection_string)
                self._client.admin.command("ping")
            except Exception as exc:
                raise ConnectionError(f"无法连接到 MongoDB: {exc}") from exc

        db = self._client[self._database_name]
        return db[self._collection_name]

    def save_messages(self, messages: list[BaseMessage]) -> None:
        """保存消息到 MongoDB"""
        try:
            collection = self._get_collection()

            # 删除旧消息
            collection.delete_many({"session_id": self._session_id})

            # 插入新消息
            documents = []
            for i, msg in enumerate(messages):
                doc = {
                    "session_id": self._session_id,
                    "index": i,
                    "type": msg.type,
                    "content": msg.content,
                    "created_at": datetime.now(),
                }
                if hasattr(msg, "additional_kwargs"):
                    doc["additional_kwargs"] = msg.additional_kwargs
                documents.append(doc)

            if documents:
                collection.insert_many(documents)

            logger.debug(f"MongoDBChatHistory: 保存 {len(messages)} 条消息")

        except Exception as e:
            logger.error(f"MongoDBChatHistory 保存失败: {e}")
            raise

    def load_messages(self) -> list[BaseMessage]:
        """从 MongoDB 加载消息"""
        try:
            collection = self._get_collection()

            cursor = collection.find(
                {"session_id": self._session_id}
            ).sort("index", 1)

            messages = []
            for doc in cursor:
                msg_type = doc.get("type", "human")
                content = doc.get("content", "")

                if msg_type in ("human", "user"):
                    msg = HumanMessage(content=content)
                elif msg_type in ("ai", "assistant"):
                    msg = AIMessage(content=content)
                elif msg_type == "system":
                    msg = SystemMessage(content=content)
                else:
                    msg = HumanMessage(content=content)

                if "additional_kwargs" in doc:
                    msg.additional_kwargs = doc["additional_kwargs"]

                messages.append(msg)

            logger.debug(f"MongoDBChatHistory: 加载 {len(messages)} 条消息")
            return messages

        except Exception as e:
            logger.error(f"MongoDBChatHistory 加载失败: {e}")
            return []

    def clear_messages(self) -> None:
        """清空 MongoDB 中的消息"""
        try:
            collection = self._get_collection()
            collection.delete_many({"session_id": self._session_id})
            logger.debug(f"MongoDBChatHistory: 清空会话 {self._session_id}")

        except Exception as e:
            logger.error(f"MongoDBChatHistory 清空失败: {e}")
            raise


# =============================================================================
# 工厂函数
# =============================================================================


def create_chat_history(
    session_id: str,
    backend: str = "memory",
    **kwargs: Any,
) -> ChatHistoryAdapter:
    """
    工厂函数：根据后端类型创建 Chat History

    Args:
        session_id: 会话唯一标识
        backend: 存储后端类型
            - "memory": 内存存储（默认）
            - "redis": Redis 存储
            - "file": 文件存储
            - "postgres": PostgreSQL 存储
            - "mongodb": MongoDB 存储
        **kwargs: 透传给对应后端的构造函数

    Returns:
        ChatHistoryAdapter 实例

    Example:
        ```python
        # Redis 存储
        history = create_chat_history("user-123", backend="redis", redis_url="...")

        # 文件存储
        history = create_chat_history("user-123", backend="file", file_path="./data")
        ```
    """
    backends = {
        "memory": None,  # 需要特殊处理
        "redis": RedisChatHistory,
        "file": FileChatHistory,
        "postgres": PostgresChatHistory,
        "mongodb": MongoDBChatHistory,
    }

    if backend not in backends:
        raise ValueError(
            f"不支持的后端类型: {backend}，支持的: {list(backends.keys())}"
        )

    if backend == "memory":
        from langchain_core.chat_history import InMemoryChatMessageHistory

        return InMemoryChatMessageHistory()

    cls = backends[backend]
    return cls(session_id=session_id, **kwargs)


def create_redis_history(
    session_id: str,
    redis_url: str = "redis://localhost:6379/0",
    ttl: Optional[int] = None,
) -> RedisChatHistory:
    """
    工厂函数：创建 Redis Chat History

    Args:
        session_id: 会话 ID
        redis_url: Redis 连接 URL
        ttl: 过期时间（秒）

    Returns:
        RedisChatHistory 实例
    """
    return RedisChatHistory(
        session_id=session_id,
        redis_url=redis_url,
        ttl=ttl,
    )


def create_file_history(
    session_id: str,
    file_path: str = "./data/chat_history",
) -> FileChatHistory:
    """
    工厂函数：创建文件存储 Chat History

    Args:
        session_id: 会话 ID
        file_path: 文件存储目录

    Returns:
        FileChatHistory 实例
    """
    return FileChatHistory(
        session_id=session_id,
        file_path=file_path,
    )


__all__ = [
    "ChatHistoryAdapter",
    "RedisChatHistory",
    "FileChatHistory",
    "PostgresChatHistory",
    "MongoDBChatHistory",
    "create_chat_history",
    "create_redis_history",
    "create_file_history",
]
