# -*- coding: utf-8 -*-
"""
依赖注入容器

基于 LangChain 标准组件的统一依赖管理。
支持上下文管理器模式管理生命周期。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generator, Optional, TypeVar

from sqlalchemy import Engine
from sqlalchemy.orm import Session

from core.logger import logger

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

_T = TypeVar("_T")


class Container:
    """
    依赖注入容器（单例模式）

    使用上下文管理器模式管理全局依赖实例的生命周期。

    使用方式有两种：

    方式一：分步写法（先获取容器，再进入上下文）
        from core.container import lifespan_container

        container = lifespan_container()      # 创建/获取单例
        with container:                       # 进入上下文，触发 startup
            agent = container.agent            # 使用依赖
        # 退出上下文时自动调用 shutdown

    方式二：链式写法（直接在 with 中调用）
        from core.container import lifespan_container

        with lifespan_container() as container:   # 创建容器并立即进入上下文
            agent = container.agent                 # 使用依赖
        # 退出上下文时自动调用 shutdown
    """

    _instance: Optional["Container"] = None

    def __new__(cls) -> "Container":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if hasattr(self, "_initialized"):
            return

        self._initialized: bool = False
        self._engine: Optional[Engine] = None
        self._llm_providers: dict[str, Any] = {}
        self._agent: Any = None
        self._memory: Any = None

    def _startup(self) -> None:
        """启动时初始化依赖"""
        if self._initialized:
            return

        logger.info("初始化依赖注入容器...")

        # 初始化数据库连接（懒加载，首次使用时才真正连接）
        # 这里只是标记预初始化，实际连接在首次使用时建立
        self._engine = None
        self._initialized = True

        logger.info("依赖注入容器初始化完成")

    def _shutdown(self) -> None:
        """关闭时清理资源"""
        if not self._initialized:
            return

        logger.info("清理依赖注入容器...")

        # 清理 LLM 提供者缓存
        self._llm_providers.clear()

        # 清理 Agent
        self._agent = None

        # 清理 Memory
        self._memory = None

        # 关闭数据库引擎
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None

        self._initialized = False
        # 清除单例引用，允许下次创建新的实例
        Container._instance = None
        logger.info("依赖注入容器清理完成")

    def __enter__(self) -> "Container":
        """同步上下文管理器入口"""
        self._startup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """同步上下文管理器出口"""
        self._shutdown()

    # region Database

    @property
    def db_engine(self) -> Engine:
        """获取数据库引擎（懒加载）"""
        if self._engine is None:
            from infras.mysql.mysql import engine as _engine

            self._engine = _engine
        return self._engine

    def get_db(self) -> Generator[Session, None, None]:
        """获取同步数据库会话（依赖注入用）"""
        from infras.mysql.mysql import get_db as _get_db

        yield from _get_db()

    def get_db_operations(self) -> Any:
        """获取数据库操作工具（TextToSQL 用，单例）"""
        from infras.mysql.operations import DBOperations

        # 使用类级别的单例缓存
        if not hasattr(DBOperations, "_singleton_instance"):
            DBOperations._singleton_instance = DBOperations()
        return DBOperations._singleton_instance

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

    @property
    def agent(self) -> Any:
        """获取 Agent 实例（懒加载）"""
        if self._agent is None:
            from . import settings

            from agent import LCAgent

            self._agent = LCAgent(config=settings.agent)
            logger.info("Agent 实例已创建")
        return self._agent

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
        self._agent = None
        self._memory = None
        logger.info("容器已重置")


def lifespan_container() -> Container:
    """
    创建依赖注入容器（返回单例）

    用法示例：

        from core.container import lifespan_container

        # 方式一：分步写法
        container = lifespan_container()
        with container:
            agent = container.agent

        # 方式二：链式写法（更简洁，推荐）
        with lifespan_container() as container:
            agent = container.agent
    """
    return Container()


__all__ = ["Container", "lifespan_container"]
