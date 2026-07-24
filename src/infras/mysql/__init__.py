# -*- coding: utf-8 -*-
"""
MySQL 数据库模块

提供 SQLAlchemy 数据库连接和会话管理。
"""

from core.container import container

from .models import Base, Conversation, Message
from .mysql import (
    AsyncSessionLocal,
    async_engine,
    async_init_db,
    engine,
    get_async_db,
    get_db,
    init_db,
)
from .operations import (
    DBOperations,
    apply_default_limit,
    get_db_url,
    is_safe_select_sql,
)

__all__ = [
    # 容器快捷访问
    "container",
    # 会话管理
    "get_db",
    "get_async_db",
    # 引擎
    "engine",
    "async_engine",
    "AsyncSessionLocal",
    # 表模型
    "Base",
    "Conversation",
    "Message",
    # 初始化
    "init_db",
    "async_init_db",
    # 数据库操作工具（TextToSQL）
    "DBOperations",
    "apply_default_limit",
    "get_db_url",
    "is_safe_select_sql",
]
