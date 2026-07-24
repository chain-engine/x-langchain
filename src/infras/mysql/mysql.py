# -*- coding: utf-8 -*-
"""
数据库连接管理

提供同步和异步的数据库连接。
"""

from typing import AsyncGenerator, Generator

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, sessionmaker

from core.config import settings

from .models import Base

# 同步引擎
engine = create_engine(
    settings.get_db_url(),
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=settings.DEBUG,
)

# 异步引擎 - 将同步 URL 转换为异步 URL
# mysql+pymysql:// -> mysql+aiomysql://
_async_db_url = settings.get_db_url().replace("mysql+pymysql", "mysql+aiomysql")
async_engine = create_async_engine(
    _async_db_url,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=settings.DEBUG,
)

# 同步会话工厂
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 异步会话工厂
AsyncSessionLocal = async_sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


def get_db() -> Generator[Session, None, None]:
    """
    获取同步数据库会话（用于依赖注入）
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """
    获取异步数据库会话（用于依赖注入）
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


def init_db() -> None:
    """
    初始化数据库表结构
    """
    Base.metadata.create_all(bind=engine)


async def async_init_db() -> None:
    """
    异步初始化数据库表结构
    """
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
