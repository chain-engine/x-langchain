# -*- coding: utf-8 -*-
"""
会话历史服务

基于 MySQL (SQLAlchemy) 的对话历史持久化。
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Generator, Optional

from sqlalchemy.orm import Session

from core.logger import logger

if TYPE_CHECKING:
    from infras.mysql.models import Conversation, Message


class ChatHistoryService:
    """
    对话历史持久化服务

    封装 Conversation/Message 模型的 CRUD 操作。
    """

    def __init__(self, db: Session):
        self._db: Session = db

    def get_or_create_conversation(
        self,
        session_id: str,
        model_provider: str = "tongyi",
        title: str | None = None,
    ) -> "Conversation":
        """
        根据 session_id 获取或创建会话。

        Args:
            session_id: 会话唯一 ID
            model_provider: 模型提供者
            title: 会话标题

        Returns:
            Conversation 实例
        """
        from infras.mysql.models import Conversation

        conv = self._db.query(Conversation).filter(Conversation.id == session_id).first()
        if conv is None:
            conv = Conversation(
                id=session_id,
                title=title or "新对话",
                model_provider=model_provider,
            )
            self._db.add(conv)
            self._db.flush()
            logger.info(f"创建新会话: {session_id}")
        return conv

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        model_provider: str = "tongyi",
    ) -> "Message":
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
        from infras.mysql.models import Message

        conv = self.get_or_create_conversation(session_id, model_provider)
        msg = Message(
            conversation_id=conv.id,
            role=role,
            content=content,
        )
        self._db.add(msg)
        conv.updated_at = datetime.now()
        self._db.flush()
        logger.debug(f"添加消息 [{role}] 到会话 {session_id}: {content[:50]}...")
        return msg

    def get_messages(self, session_id: str) -> list["Message"]:
        """
        获取指定会话的所有消息（按时间升序）。

        Args:
            session_id: 会话 ID

        Returns:
            Message 列表
        """
        from infras.mysql.models import Message

        return (
            self._db.query(Message)
            .filter(Message.conversation_id == session_id)
            .order_by(Message.created_at.asc())
            .all()
        )

    def get_conversation(self, session_id: str) -> "Conversation | None":
        """
        获取指定会话。

        Args:
            session_id: 会话 ID

        Returns:
            Conversation 实例，不存在则返回 None
        """
        from infras.mysql.models import Conversation

        return self._db.query(Conversation).filter(Conversation.id == session_id).first()

    def list_conversations(
        self,
        limit: int = 20,
        offset: int = 0,
    ) -> list["Conversation"]:
        """
        获取会话列表（按更新时间降序）。

        Args:
            limit: 返回数量
            offset: 偏移量

        Returns:
            Conversation 列表
        """
        from infras.mysql.models import Conversation

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
        from infras.mysql.models import Conversation

        conv = self._db.query(Conversation).filter(Conversation.id == session_id).first()
        if conv is None:
            return False
        conv.title = title
        conv.updated_at = datetime.now()
        self._db.flush()
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
        from infras.mysql.models import Conversation

        conv = self._db.query(Conversation).filter(Conversation.id == session_id).first()
        if conv is None:
            return False
        conv.summary = summary
        conv.updated_at = datetime.now()
        self._db.flush()
        return True

    def delete_conversation(self, session_id: str) -> bool:
        """
        删除指定会话及其所有消息。

        Args:
            session_id: 会话 ID

        Returns:
            是否删除成功
        """
        from infras.mysql.models import Conversation

        conv = self._db.query(Conversation).filter(Conversation.id == session_id).first()
        if conv is None:
            return False
        self._db.delete(conv)
        self._db.flush()
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
        from infras.mysql.models import Message

        count = (
            self._db.query(Message)
            .filter(Message.conversation_id == session_id)
            .delete()
        )
        self._db.flush()
        logger.info(f"清空会话 {session_id}，删除 {count} 条消息")
        return count > 0 or self.get_conversation(session_id) is not None

    def search_messages(self, session_id: str, keyword: str, limit: int = 20) -> list["Message"]:
        """
        在指定会话中搜索包含关键词的消息。

        Args:
            session_id: 会话 ID
            keyword: 搜索关键词
            limit: 返回数量上限

        Returns:
            匹配的消息列表
        """
        from infras.mysql.models import Message

        return (
            self._db.query(Message)
            .filter(Message.conversation_id == session_id)
            .filter(Message.content.ilike(f"%{keyword}%"))
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
        from infras.mysql.models import Conversation, Message

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


def create_chat_history_service(db: Optional[Session] = None) -> ChatHistoryService:
    """
    创建 ChatHistoryService 实例。

    Args:
        db: 可选的外部 Session，不提供则自动创建

    Returns:
        ChatHistoryService 实例
    """
    if db is not None:
        return ChatHistoryService(db)

    from infras.mysql.mysql import SessionLocal

    db = SessionLocal()
    try:
        return ChatHistoryService(db)
    except Exception:
        db.close()
        raise


def chat_history_context() -> Generator[ChatHistoryService, None, None]:
    """
    生成器形式的 ChatHistoryService（自动管理 Session）。

    用法:
        for svc in chat_history_context():
            svc.add_message(...)
    """
    from infras.mysql.mysql import SessionLocal

    db = SessionLocal()
    try:
        yield ChatHistoryService(db)
    finally:
        db.close()


def generate_session_id() -> str:
    """生成新的会话 ID（UUID4）。"""
    return str(uuid.uuid4())


__all__ = [
    "ChatHistoryService",
    "create_chat_history_service",
    "chat_history_context",
    "generate_session_id",
]
