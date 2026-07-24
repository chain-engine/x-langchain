# -*- coding: utf-8 -*-
"""
依赖注入容器

提供全局统一的依赖管理，支持：
- 数据库连接（同步/异步）
- LLM 提供者
- 工具注册表
- 记忆管理器
- 规划管理器
- 行动调度器
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generator, Optional, TypeVar

from sqlalchemy import Engine
from sqlalchemy.orm import Session

from core.logger import logger

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, AsyncEngine

    from src.infras.mysql.models import Base
    from src.llms.providers import BaseLLMProvider
    from src.memories.base import BaseMemory
    from src.planning.manager import PlanningManager
    from src.actions.dispatcher import ActionDispatcher

_T = TypeVar("_T")


class Container:
    """
    依赖注入容器

    使用单例模式管理全局依赖实例，支持懒加载和依赖自动注入。
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
        self._async_engine: Optional[Any] = None
        self._db_session_factory: Optional[Any] = None
        self._async_session_factory: Optional[Any] = None
        self._llm_providers: dict[str, Any] = {}
        self._default_memory: Optional[Any] = None
        self._planning_manager: Optional[Any] = None
        self._action_dispatcher: Optional[Any] = None
        self._tool_registry: Optional[Any] = None

    # region Database

    @property
    def engine(self) -> Engine:
        """获取同步数据库引擎（懒加载）"""
        if self._engine is None:
            from src.infras.mysql.mysql import engine as _engine
            self._engine = _engine
        return self._engine

    @property
    def async_engine(self) -> Any:
        """获取异步数据库引擎（懒加载）"""
        if self._async_engine is None:
            from src.infras.mysql.mysql import async_engine as _engine
            self._async_engine = _engine
        return self._async_engine

    def get_db(self) -> Generator[Session, None, None]:
        """获取同步数据库会话（依赖注入用）"""
        from src.infras.mysql.mysql import get_db as _get_db
        yield from _get_db()

    def get_async_db(self) -> Any:
        """获取异步数据库会话（依赖注入用）"""
        from src.infras.mysql.mysql import get_async_db as _get_async_db
        return _get_async_db()

    def init_db(self) -> None:
        """初始化数据库表结构"""
        from src.infras.mysql.mysql import init_db
        init_db()
        logger.info("数据库表结构初始化完成")

    async def async_init_db(self) -> None:
        """异步初始化数据库表结构"""
        from src.infras.mysql.mysql import async_init_db
        await async_init_db()
        logger.info("数据库表结构初始化完成")

    def get_db_operations(self) -> Any:
        """获取数据库操作工具（TextToSQL 用）"""
        from src.infras.mysql.operations import DBOperations
        return DBOperations()

    # endregion

    # region LLM

    def get_llm_provider(
        self,
        provider_name: str = "deepseek",
        **kwargs,
    ) -> Any:
        """
        获取 LLM 提供者实例

        Args:
            provider_name: 提供者名称
            **kwargs: 额外参数

        Returns:
            LLM 提供者实例
        """
        from src.llms.providers import get_llm_provider
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
        """
        创建聊天模型

        Args:
            provider_name: 提供者名称
            model_name: 模型名称
            temperature: 温度参数
            **kwargs: 额外参数

        Returns:
            聊天模型实例
        """
        from src.llms.providers import create_chat_model
        return create_chat_model(
            provider_name=provider_name,
            model_name=model_name,
            temperature=temperature,
            **kwargs,
        )

    # endregion

    # region Tools

    @property
    def tool_registry(self) -> Any:
        """获取工具注册表（懒加载）"""
        if self._tool_registry is None:
            from src.tools import ToolRegistry
            self._tool_registry = ToolRegistry
        return self._tool_registry

    def get_tools(self) -> list[Any]:
        """获取所有已注册工具"""
        registry = self.tool_registry
        if hasattr(registry, "get_all"):
            return registry.get_all()
        return []

    # endregion

    # region Memory

    def get_memory(
        self,
        system_message: Optional[str] = None,
        max_messages: int = 100,
    ) -> Any:
        """
        获取记忆实例

        Args:
            system_message: 系统消息
            max_messages: 最大消息数

        Returns:
            记忆实例
        """
        from src.memories import ConversationHistoryMemory
        return ConversationHistoryMemory(
            system_message=system_message,
            max_messages=max_messages,
        )

    def get_memory_manager(self) -> Any:
        """获取记忆管理器"""
        from src.memories import MemoryManager
        return MemoryManager()

    # endregion

    # region Planning

    def get_planning_manager(self, planner: Optional[Any] = None) -> Any:
        """
        获取规划管理器

        Args:
            planner: 规划器实例

        Returns:
            规划管理器实例
        """
        from src.planning import PlanningManager
        return PlanningManager(planner=planner)

    # endregion

    # region Action

    def get_action_dispatcher(
        self,
        tool_registry: Optional[Any] = None,
    ) -> Any:
        """
        获取行动调度器

        Args:
            tool_registry: 工具注册表

        Returns:
            行动调度器实例
        """
        from src.actions import ActionDispatcher
        return ActionDispatcher(tool_registry=tool_registry or self.tool_registry)

    # endregion

    # region Agent

    def create_agent(
        self,
        model_provider: str = "deepseek",
        model_name: Optional[str] = None,
        memory: Optional[Any] = None,
        planning_manager: Optional[Any] = None,
        action_dispatcher: Optional[Any] = None,
        enable_memory: bool = True,
        enable_planning: bool = False,
        enable_tools: bool = True,
        system_prompt: Optional[str] = None,
        **config_kwargs,
    ) -> Any:
        """
        创建 Agent 实例

        Args:
            model_provider: 模型提供者
            model_name: 模型名称
            memory: 记忆实例
            planning_manager: 规划管理器
            action_dispatcher: 行动调度器
            enable_memory: 是否启用记忆
            enable_planning: 是否启用规划
            enable_tools: 是否启用工具
            system_prompt: 系统提示词
            **config_kwargs: AgentConfig 其他参数

        Returns:
            Agent 实例
        """
        # Agent creation delegated to src.agent.agent.Agent
        # See src.agent.agent.Agent for the self-built Agent implementation
        pass

    # endregion

    def reset(self) -> None:
        """重置容器，清除所有缓存的依赖实例"""
        self._llm_providers.clear()
        self._default_memory = None
        self._planning_manager = None
        self._action_dispatcher = None
        self._tool_registry = None
        logger.info("容器已重置")


container = Container()

__all__ = ["Container", "container"]
