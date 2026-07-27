# -*- coding: utf-8 -*-
"""
基础设施层（infras）

封装第三方中间件、客户端、连接生命周期、底层资源管理。
仅提供基础资源，不包含任何业务逻辑。

目录结构：
    infras/
    ├── mysql/      # MySQL 数据库基础设施
    └── redis/      # Redis 缓存基础设施（预留）
"""

from .mysql import (
    Base,
    SessionLocal,
    AsyncSessionLocal,
    engine,
    async_engine,
    get_db,
    get_async_db,
    init_db,
    async_init_db,
    dispose_engine,
    async_dispose_engine,
)

__all__ = [
    "Base",
    "SessionLocal",
    "AsyncSessionLocal",
    "engine",
    "async_engine",
    "get_db",
    "get_async_db",
    "init_db",
    "async_init_db",
    "dispose_engine",
    "async_dispose_engine",
]
