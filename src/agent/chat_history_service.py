# -*- coding: utf-8 -*-
"""
会话历史服务

基于 MySQL (SQLAlchemy) 的对话历史持久化。
提供上下文管理器支持，自动管理 Session 生命周期。
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import TYPE_CHECKING, Generator, Optional

from sqlalchemy.orm import Session

from core.logger import logger


class ChatHistoryService:
    """
    对话历史持久化服务

    封装 Conversation/Message 模型的 CRUD 操作。
    支持上下文管理器和依赖注入。

    使用方式：
        ```python
        # 方式1：上下文管理器（推荐，自动管理 Session）
        with chat_history_service() as svc:
            svc.add_message(session_id, "user", "你好")
            svc.add_message(session_id, "assistant", "你好，有什么帮助？")

        # 方式2：工厂函数
        svc = create_chat_history_service()
        try:
            svc.add_message(session_id, "user", "你好")
        finally:
            svc.close()
        ```
    """

    def __init__(self, db: Session, *, auto_commit: bool = True):
        """
        初始化对话历史服务

        Args:
            db: SQLAlchemy Session 实例
            auto_commit: 是否在操作后自动提交（默认 True）
        """
        self._db: Session = db
        self._auto_commit: bool = auto_commit

    def __enter__(self) -> "ChatHistoryService":
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """上下文管理器退出，自动关闭 Session"""
        self.close()

    def close(self) -> None:
        """关闭 Session，释放数据库连接"""
        if self._db is not None:
            try:
                self._db.close()
            except Exception as e:
                logger.warning(f"关闭 Session 失败: {e}")
            self._db = None

    def _commit(self) -> None:
        """提交事务（仅当 auto_commit 为 True 时）"""
        if self._auto_commit:
            try:
                self._db.commit()
            except Exception:
                self._db.rollback()
                raise

    def _get_conversation_model(self):
        """获取 Conversation 模型类"""
        from infras.mysql.models import Conversation
        return Conversation

    def _get_message_model(self):
        """获取 Message 模型类"""
        from infras.mysql.models import Message
        return Message

    def get_or_create_conversation(
        self,
        session_id: str,
        model_provider: str = "tongyi",
        title: str | None = None,
    ):
        """
        根据 session_id 获取或创建会话。

        Args:
            session_id: 会话唯一 ID
            model_provider: 模型提供者
            title: 会话标题

        Returns:
            Conversation 实例
        """
        Conversation = self._get_conversation_model()
        conv = self._db.query(Conversation).filter(Conversation.id == session_id).first()
        if conv is None:
            conv = Conversation(
                id=session_id,
                title=title or "新对话",
                model_provider=model_provider,
            )
            self._db.add(conv)
            self._db.flush()
            self._commit()
            logger.info(f"创建新会话: {session_id}")
        return conv

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        model_provider: str = "tongyi",
    ):
        """
        添加消息到指定会话。

        Args:
            session_id: 会话 ID
            role: 角色 (user / assistant)
            content: 消息内容
            model_provider: 模型提供者

        Returns:
            Message 实例
        """
        Message = self._get_message_model()
        conv = self.get_or_create_conversation(session_id, model_provider)
        msg = Message(
            conversation_id=conv.id,
            role=role,
            content=content,
        )
        self._db.add(msg)
        conv.updated_at = datetime.now()
        self._db.flush()
        self._commit()
        logger.debug(f"添加消息 [{role}] 到会话 {session_id}: {content[:50]}...")
        return msg

    def add_messages_batch(
        self,
        session_id: str,
        messages: list[tuple[str, str]],
        model_provider: str = "tongyi",
    ):
        """
        批量添加消息（在一个事务中）。

        Args:
            session_id: 会话 ID
            messages: 消息列表 [(role, content), ...]
            model_provider: 模型提供者

        Returns:
            Message 实例列表
        """
        Message = self._get_message_model()
        conv = self.get_or_create_conversation(session_id, model_provider)
        msg_objects = []
        for role, content in messages:
            msg = Message(
                conversation_id=conv.id,
                role=role,
                content=content,
            )
            self._db.add(msg)
            msg_objects.append(msg)

        conv.updated_at = datetime.now()
        self._db.flush()
        self._commit()
        logger.debug(f"批量添加 {len(messages)} 条消息到会话 {session_id}")
        return msg_objects

    def get_messages(self, session_id: str) -> list:
        """
        获取指定会话的所有消息（按时间升序）。

        Args:
            session_id: 会话 ID

        Returns:
            Message 列表
        """
        Message = self._get_message_model()
        return (
            self._db.query(Message)
            .filter(Message.conversation_id == session_id)
            .order_by(Message.created_at.asc())
            .all()
        )

    def get_recent_messages(self, session_id: str, limit: int = 10) -> list:
        """
        获取指定会话最近的 N 条消息。

        Args:
            session_id: 会话 ID
            limit: 返回消息数量

        Returns:
            Message 列表（按时间升序）
        """
        Message = self._get_message_model()
        results = (
            self._db.query(Message)
            .filter(Message.conversation_id == session_id)
            .order_by(Message.created_at.desc())
            .limit(limit)
            .all()
        )
        return list(reversed(results))

    def get_conversation(self, session_id: str):
        """
        获取指定会话。

        Args:
            session_id: 会话 ID

        Returns:
            Conversation 实例，不存在则返回 None
        """
        Conversation = self._get_conversation_model()
        return self._db.query(Conversation).filter(Conversation.id == session_id).first()

    def list_conversations(
        self,
        limit: int = 20,
        offset: int = 0,
    ) -> list:
        """
        获取会话列表（按更新时间降序）。

        Args:
            limit: 返回数量
            offset: 偏移量

        Returns:
            Conversation 列表
        """
        Conversation = self._get_conversation_model()
        return (
            self._db.query(Conversation)
            .order_by(Conversation.updated_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )

    def update_title(self, session_id: str, title: str) -> bool:
        """
        更新会话标题。

        Args:
            session_id: 会话 ID
            title: 新标题

        Returns:
            是否更新成功
        """
        Conversation = self._get_conversation_model()
        conv = self._db.query(Conversation).filter(Conversation.id == session_id).first()
        if conv is None:
            return False
        conv.title = title
        conv.updated_at = datetime.now()
        self._db.flush()
        self._commit()
        return True

    def update_summary(self, session_id: str, summary: str) -> bool:
        """
        更新会话摘要。

        Args:
            session_id: 会话 ID
            summary: 摘要内容

        Returns:
            是否更新成功
        """
        Conversation = self._get_conversation_model()
        conv = self._db.query(Conversation).filter(Conversation.id == session_id).first()
        if conv is None:
            return False
        conv.summary = summary
        conv.updated_at = datetime.now()
        self._db.flush()
        self._commit()
        return True

    def delete_conversation(self, session_id: str) -> bool:
        """
        删除指定会话及其所有消息。

        Args:
            session_id: 会话 ID

        Returns:
            是否删除成功
        """
        Conversation = self._get_conversation_model()
        conv = self._db.query(Conversation).filter(Conversation.id == session_id).first()
        if conv is None:
            return False
        self._db.delete(conv)
        self._db.flush()
        self._commit()
        logger.info(f"删除会话: {session_id}")
        return True

    def clear_conversation(self, session_id: str) -> bool:
        """
        清空指定会话的所有消息（保留会话）。

        Args:
            session_id: 会话 ID

        Returns:
            是否清空成功
        """
        Message = self._get_message_model()
        count = (
            self._db.query(Message)
            .filter(Message.conversation_id == session_id)
            .delete()
        )
        self._db.flush()
        self._commit()
        logger.info(f"清空会话 {session_id}，删除 {count} 条消息")
        return count > 0 or self.get_conversation(session_id) is not None

    def search_messages(self, session_id: str, keyword: str, limit: int = 20) -> list:
        """
        在指定会话中搜索包含关键词的消息。

        Args:
            session_id: 会话 ID
            keyword: 搜索关键词
            limit: 返回数量上限

        Returns:
            匹配的消息列表
        """
        Message = self._get_message_model()
        escaped_keyword = keyword.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        return (
            self._db.query(Message)
            .filter(Message.conversation_id == session_id)
            .filter(Message.content.ilike(f"%{escaped_keyword}%"))
            .order_by(Message.created_at.desc())
            .limit(limit)
            .all()
        )

    def get_session_summary(self, session_id: str) -> dict:
        """
        获取会话摘要统计信息。

        Args:
            session_id: 会话 ID

        Returns:
            包含消息数、最后更新时间等统计信息
        """
        Conversation = self._get_conversation_model()
        Message = self._get_message_model()

        conv = self._db.query(Conversation).filter(Conversation.id == session_id).first()
        if conv is None:
            return {}

        msg_count = (
            self._db.query(Message)
            .filter(Message.conversation_id == session_id)
            .count()
        )
        user_count = (
            self._db.query(Message)
            .filter(Message.conversation_id == session_id, Message.role == "user")
            .count()
        )
        assistant_count = msg_count - user_count

        return {
            "session_id": session_id,
            "title": conv.title,
            "model_provider": conv.model_provider,
            "total_messages": msg_count,
            "user_messages": user_count,
            "assistant_messages": assistant_count,
            "created_at": conv.created_at.isoformat() if conv.created_at else None,
            "updated_at": conv.updated_at.isoformat() if conv.updated_at else None,
            "summary": conv.summary,
        }


@contextmanager
def chat_history_service(
    auto_commit: bool = True,
) -> Generator[ChatHistoryService, None, None]:
    """
    创建 ChatHistoryService 实例的上下文管理器（推荐用法）。

    自动管理 Session 的创建和关闭，确保资源正确释放。

    用法:
        ```python
        with chat_history_service() as svc:
            svc.add_message(session_id, "user", "你好")
            svc.add_message(session_id, "assistant", "你好！")
        # Session 自动关闭
        ```

    Args:
        auto_commit: 是否自动提交事务

    Yields:
        ChatHistoryService 实例
    """
    from infras.mysql import SessionLocal

    db = SessionLocal()
    try:
        yield ChatHistoryService(db, auto_commit=auto_commit)
    except Exception:
        db.rollback()
        raise
    finally:
        try:
            db.close()
        except Exception as e:
            logger.warning(f"关闭 Session 失败: {e}")


def create_chat_history_service(
    db: Optional[Session] = None,
    auto_commit: bool = True,
) -> ChatHistoryService:
    """
    创建 ChatHistoryService 实例的工厂函数。

    Args:
        db: 可选的外部 Session，不提供则自动创建
        auto_commit: 是否自动提交事务

    Returns:
        ChatHistoryService 实例
    """
    if db is not None:
        return ChatHistoryService(db, auto_commit=auto_commit)

    from infras.mysql import SessionLocal

    db = SessionLocal()
    return ChatHistoryService(db, auto_commit=auto_commit)


def generate_session_id() -> str:
    """生成新的会话 ID（UUID4）。"""
    return str(uuid.uuid4())


__all__ = [
    "ChatHistoryService",
    "chat_history_service",
    "create_chat_history_service",
    "generate_session_id",
]
