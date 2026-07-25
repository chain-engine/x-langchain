# -*- coding: utf-8 -*-
"""
依赖注入容器

基于 LangChain 标准组件的统一依赖管理。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generator, Optional, TypeVar

from sqlalchemy import Engine
from sqlalchemy.orm import Session

from core.logger import logger

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, AsyncEngine

    from llms.providers import BaseChatModel
    from memories.langchain_memory import BaseMemory

_T = TypeVar("_T")


class Container:
    """
    依赖注入容器

    使用单例模式管理全局依赖实例，支持懒加载。
    """

    _instance: Optional[Container] = None
    _init_guard: bool = False

    def __new__(cls) -> "Container":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if Container._init_guard:
            return
        Container._init_guard = True

        self._engine: Optional[Engine] = None
        self._llm_providers: dict[str, Any] = {}

    # region Database

    @property
    def engine(self) -> Engine:
        """获取同步数据库引擎（懒加载）"""
        if self._engine is None:
            from infras.mysql.mysql import engine as _engine
            self._engine = _engine
        return self._engine

    def get_db(self) -> Generator[Session, None, None]:
        """获取同步数据库会话（依赖注入用）"""
        from infras.mysql.mysql import get_db as _get_db
        yield from _get_db()

    def get_db_operations(self) -> Any:
        """获取数据库操作工具（TextToSQL 用）"""
        from infras.mysql.operations import DBOperations
        return DBOperations()

    # endregion

    # region LLM

    def get_llm_provider(
        self,
        provider_name: str = "deepseek",
        **kwargs,
    ) -> Any:
        """获取 LLM 提供者实例"""
        from llms.providers import get_llm_provider
        key = f"{provider_name}:{hash(frozenset(kwargs.items()))}"
        if key not in self._llm_providers:
            self._llm_providers[key] = get_llm_provider(provider_name, **kwargs)
        return self._llm_providers[key]

    def create_chat_model(
        self,
        provider_name: str = "deepseek",
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> Any:
        """创建聊天模型"""
        from llms.providers import create_chat_model
        return create_chat_model(
            provider_name=provider_name,
            model_name=model_name,
            temperature=temperature,
            **kwargs,
        )

    # endregion

    # region Agent

    def create_agent(
        self,
        model_provider: str = "deepseek",
        model_name: Optional[str] = None,
        tools: Optional[list] = None,
        system_message: Optional[str] = None,
        **config_kwargs,
    ) -> Any:
        """
        创建 LangChain Agent

        Args:
            model_provider: 模型提供者
            model_name: 模型名称
            tools: 工具列表
            system_message: 系统消息
            **config_kwargs: 其他配置

        Returns:
            LCAgent 实例
        """
        from agent.lc_agent import LCAgent

        llm = self.create_chat_model(
            provider_name=model_provider,
            model_name=model_name,
        )

        return LCAgent(
            llm=llm,
            tools=tools,
            system_message=system_message,
        )

    # endregion

    def reset(self) -> None:
        """重置容器，清除所有缓存的依赖实例"""
        self._llm_providers.clear()
        logger.info("容器已重置")


container = Container()

__all__ = ["Container", "container"]
