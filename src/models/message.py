# -*- coding: utf-8 -*-
"""
消息模型

定义消息（Message）的数据库结构。
"""

from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base

if TYPE_CHECKING:
    from .conversation import Conversation


class Message(Base):
    """
    消息模型

    存储会话中的单条消息。

    Attributes:
        id: 消息唯一标识（自增）
        conversation_id: 所属会话 ID
        role: 消息角色（user/assistant/system）
        content: 消息内容
        created_at: 创建时间
        conversation: 关联的会话对象
    """
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("conversations.id", ondelete="CASCADE")
    )
    role: Mapped[str] = mapped_column(String(20))  # user / assistant / system
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now()
    )

    conversation: Mapped["Conversation"] = relationship(
        "Conversation",
        back_populates="messages",
    )
