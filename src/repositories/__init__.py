# -*- coding: utf-8 -*-
"""
数据访问层

封装业务 CRUD、多表联查、分页、条件查询。
依赖 infra 获取数据库会话，infra 永不反向依赖 repository/service/api。
"""

from .base import Repository
from .conversation import ConversationRepository
from .message import MessageRepository
from .chat import ChatRepository, chat_repository, generate_session_id

__all__ = [
    "Repository",
    "ConversationRepository",
    "MessageRepository",
    "ChatRepository",
    "chat_repository",
    "generate_session_id",
]
