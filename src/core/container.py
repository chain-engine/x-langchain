# -*- coding: utf-8 -*-
"""
依赖注入容器

基于 LangChain 标准组件的统一依赖管理。
所有核心模块应通过此容器获取依赖。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, AsyncGenerator, Generator, Optional, TypeVar

from sqlalchemy import Engine
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from sqlalchemy.orm import Session

from core.config import settings
from core.logger import logger

if TYPE_CHECKING:
    from llms.providers import BaseChatModel
    from memories.langchain_memory import BaseMemory

_T = TypeVar("_T")


class Container:
    """
    依赖注入容器

    使用单例模式管理全局依赖实例，支持懒加载和依赖注入。

    Example:
        # 获取 LLM
        llm = container.create_chat_model("deepseek")

        # 获取 Agent
        agent = container.create_agent(tools=[...])

        # 获取数据库操作
        with container.get_db_operations() as db:
            schema = db.get_schema_info()
    """

    _instance: Optional["Container"] = None
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
        self._async_engine: Optional[AsyncEngine] = None
        self._llm_providers: dict[str, Any] = {}
        self._memory_instances: dict[str, Any] = {}
        self._initialized: bool = False

    # region 生命周期管理

    def initialize(self) -> None:
        """初始化容器，预热关键依赖（可选）"""
        if self._initialized:
            return
        logger.info("初始化依赖注入容器")
        self._initialized = True

    def reset(self) -> None:
        """重置容器，清除所有缓存的依赖实例"""
        self._llm_providers.clear()
        self._memory_instances.clear()
        self._initialized = False
        logger.info("容器已重置")

    def close(self) -> None:
        """关闭容器，释放所有资源"""
        if self._engine:
            self._engine.dispose()
            self._engine = None
        if self._async_engine:
            self._async_engine.dispose()
            self._async_engine = None
        self._llm_providers.clear()
        self._memory_instances.clear()
        logger.info("容器资源已释放")

    def __enter__(self) -> "Container":
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # endregion

    # region Database

    @property
    def engine(self) -> Engine:
        """获取同步数据库引擎（懒加载）"""
        if self._engine is None:
            from infras.mysql.mysql import engine as _engine
            self._engine = _engine
        return self._engine

    @property
    def async_engine(self) -> AsyncEngine:
        """获取异步数据库引擎（懒加载）"""
        if self._async_engine is None:
            from infras.mysql.mysql import async_engine as _async_engine
            self._async_engine = _async_engine
        return self._async_engine

    def get_db(self) -> Generator[Session, None, None]:
        """获取同步数据库会话（依赖注入用）"""
        from infras.mysql.mysql import get_db as _get_db
        yield from _get_db()

    def get_async_db(self) -> AsyncGenerator[AsyncSession, None]:
        """获取异步数据库会话（依赖注入用）"""
        from infras.mysql.mysql import get_async_db as _get_async_db
        return _get_async_db()

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

    # region Memory

    def get_memory(self, memory_type: str = "conversation", **kwargs) -> Any:
        """获取记忆实例"""
        from memories import create_conversation_memory, create_buffer_memory

        if memory_type not in self._memory_instances:
            if memory_type == "conversation":
                self._memory_instances[memory_type] = create_conversation_memory(**kwargs)
            elif memory_type == "buffer":
                self._memory_instances[memory_type] = create_buffer_memory(**kwargs)
            else:
                raise ValueError(f"Unknown memory type: {memory_type}")

        return self._memory_instances[memory_type]

    # endregion

    # region Agent

    def create_agent(
        self,
        model_provider: str = "deepseek",
        model_name: Optional[str] = None,
        tools: Optional[list] = None,
        system_message: Optional[str] = None,
        memory_type: Optional[str] = None,
        **config_kwargs,
    ) -> Any:
        """
        创建 LangChain Agent

        Args:
            model_provider: 模型提供者
            model_name: 模型名称
            tools: 工具列表
            system_message: 系统消息
            memory_type: 记忆类型 ("conversation" | "buffer")
            **config_kwargs: 其他配置

        Returns:
            LCAgent 实例
        """
        from agent.lc_agent import LCAgent

        llm = self.create_chat_model(
            provider_name=model_provider,
            model_name=model_name,
        )

        memory = None
        if memory_type:
            memory = self.get_memory(memory_type)

        return LCAgent(
            llm=llm,
            tools=tools,
            system_message=system_message,
            memory=memory,
        )

    # endregion

    # region Tools

    def create_text_to_sql_tools(self) -> list[Any]:
        """创建 TextToSQL 工具集"""
        from tools.text_to_sql import (
            GenerateSQLTool,
            GetSchemaTool,
            ExecuteSQLTool,
            ValidateSQLTool,
            ConvertToNaturalLanguageTool,
        )
        return [
            GetSchemaTool(),
            GenerateSQLTool(),
            ValidateSQLTool(),
            ExecuteSQLTool(),
            ConvertToNaturalLanguageTool(),
        ]

    # endregion


# 全局单例实例
container = Container()

__all__ = ["Container", "container"]
