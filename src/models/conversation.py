# -*- coding: utf-8 -*-
"""
会话模型

定义会话（Conversation）的数据库结构。
"""

from datetime import datetime
from typing import TYPE_CHECKING, List, Optional

from sqlalchemy import DateTime, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base

if TYPE_CHECKING:
    from .message import Message


class Conversation(Base):
    """
    会话模型

    存储用户的对话会话信息。

    Attributes:
        id: 会话唯一标识（UUID）
        title: 会话标题
        summary: 会话摘要/总结
        model_provider: 使用的模型提供者
        created_at: 创建时间
        updated_at: 更新时间
        messages: 关联的消息列表
    """
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    title: Mapped[str] = mapped_column(String(255), default="新对话")
    summary: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    model_provider: Mapped[str] = mapped_column(String(50), default="tongyi")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    messages: Mapped[List["Message"]] = relationship(
        "Message",
        back_populates="conversation",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
