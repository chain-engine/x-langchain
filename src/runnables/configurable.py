# -*- coding: utf-8 -*-
"""支持运行时切换聊天模型的 LCEL Runnable。"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableConfig


class ConfigurableLLM(Runnable[Any, Any]):
    """将多个聊天模型封装为可按运行配置切换的 Runnable。

    模型工厂采用惰性创建方式，并在首次使用后缓存实例。调用方可以通过
    ``configurable`` 配置中的 ``llm`` 或 ``llm_name`` 字段选择模型。
    """

    def __init__(
        self,
        default_llm: BaseChatModel,
        **llm_factories: Callable[[], BaseChatModel],
    ) -> None:
        """初始化可配置模型。

        Args:
            default_llm: 未指定模型名称时使用的默认聊天模型。
            **llm_factories: 模型名称到无参数模型工厂的映射。

        Raises:
            ValueError: 默认模型为空时抛出。
            TypeError: 模型工厂不是可调用对象时抛出。
        """
        if default_llm is None:
            raise ValueError("default_llm is required")

        invalid_factories = [
            name for name, factory in llm_factories.items() if not callable(factory)
        ]
        if invalid_factories:
            names = ", ".join(invalid_factories)
            raise TypeError(f"LLM factories must be callable: {names}")

        self._default_llm = default_llm
        self._llm_factories = dict(llm_factories)
        self._llm_cache: dict[str, BaseChatModel] = {"default": default_llm}
        self._selected_name = "default"

    @property
    def default_llm(self) -> BaseChatModel:
        """获取默认聊天模型。"""
        return self._default_llm

    @property
    def llm_factories(self) -> dict[str, Callable[[], BaseChatModel]]:
        """获取模型工厂映射的副本。"""
        return self._llm_factories.copy()

    @property
    def selected_name(self) -> str:
        """获取当前选择的模型名称。"""
        return self._selected_name

    @property
    def current_llm(self) -> BaseChatModel:
        """获取当前选择的聊天模型。"""
        return self.get_llm(self._selected_name)

    def _validate_name(self, name: str) -> None:
        """校验模型名称是否存在。"""
        if not isinstance(name, str) or not name:
            raise ValueError("LLM name must be a non-empty string")
        if name != "default" and name not in self._llm_factories:
            available = ["default", *self._llm_factories.keys()]
            names = ", ".join(available)
            raise ValueError(f"Unknown LLM: {name}. Available LLMs: {names}")

    def get_llm(self, name: str = "default") -> BaseChatModel:
        """按名称获取并切换当前聊天模型。

        Args:
            name: 模型名称，``default`` 表示构造函数传入的默认模型。

        Returns:
            选中的聊天模型实例。

        Raises:
            ValueError: 模型名称不存在时抛出。
            TypeError: 工厂返回异步结果时抛出，因为同步接口无法等待它。
        """
        self._validate_name(name)
        if name not in self._llm_cache:
            llm = self._llm_factories[name]()
            if inspect.isawaitable(llm):
                raise TypeError("同步 get_llm 不支持返回 awaitable 的模型工厂")
            self._llm_cache[name] = llm

        self._selected_name = name
        return self._llm_cache[name]

    def with_llm(self, name: str) -> "ConfigurableLLM":
        """返回一个选择了指定模型的新 Runnable 实例。

        原实例的当前模型和缓存不会被修改；已创建的模型缓存会被新实例复用。

        Args:
            name: 要选择的模型名称。

        Returns:
            选择指定模型后的新 ConfigurableLLM 实例。
        """
        self._validate_name(name)
        configured = ConfigurableLLM(
            self._default_llm,
            **self._llm_factories,
        )
        configured._llm_cache = self._llm_cache.copy()
        configured.get_llm(name)
        return configured

    @staticmethod
    def _configured_name(
        config: RunnableConfig | None,
        fallback: str,
    ) -> str:
        """从 RunnableConfig 中提取模型选择名称。"""
        if not config:
            return fallback

        configurable = config.get("configurable", {})
        if not isinstance(configurable, dict):
            return fallback

        for key in ("llm", "llm_name", "model"):
            value = configurable.get(key)
            if value is not None:
                return value
        return fallback

    def _resolve_llm(self, config: RunnableConfig | None) -> BaseChatModel:
        """根据运行配置解析本次调用使用的模型。"""
        name = self._configured_name(config, self._selected_name)
        return self.get_llm(name)

    @staticmethod
    def _invoke(
        llm: Any,
        input: Any,
        config: RunnableConfig | None,
        kwargs: dict[str, Any],
    ) -> Any:
        """调用底层模型并避免重复传递配置参数。"""
        if config is None:
            return llm.invoke(input, **kwargs)
        return llm.invoke(input, config=config, **kwargs)

    def invoke(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        """使用运行配置选择模型并同步调用。"""
        llm = self._resolve_llm(config)
        return self._invoke(llm, input, config, kwargs)

    async def ainvoke(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        """使用运行配置选择模型并异步调用。"""
        llm = self._resolve_llm(config)
        method = getattr(llm, "ainvoke", None)
        if method is None:
            return await asyncio.to_thread(self._invoke, llm, input, config, kwargs)

        result = method(input, **kwargs) if config is None else method(
            input,
            config=config,
            **kwargs,
        )
        if inspect.isawaitable(result):
            return await result
        return result

    def stream(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Iterator[Any]:
        """使用运行配置选择模型并同步流式调用。"""
        llm = self._resolve_llm(config)
        if config is None:
            yield from llm.stream(input, **kwargs)
        else:
            yield from llm.stream(input, config=config, **kwargs)

    async def astream(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """使用运行配置选择模型并异步流式调用。"""
        llm = self._resolve_llm(config)
        method = getattr(llm, "astream", None)
        if method is None:
            for chunk in await asyncio.to_thread(
                lambda: list(self.stream(input, config=config, **kwargs))
            ):
                yield chunk
            return

        stream = method(input, **kwargs) if config is None else method(
            input,
            config=config,
            **kwargs,
        )
        if inspect.isawaitable(stream):
            stream = await stream

        if hasattr(stream, "__aiter__"):
            async for chunk in stream:
                yield chunk
        else:
            for chunk in stream:
                yield chunk


def configurable_llm(
    default_llm: BaseChatModel,
    **llm_factories: Callable[[], BaseChatModel],
) -> ConfigurableLLM:
    """创建可动态切换模型的 ConfigurableLLM。"""
    return ConfigurableLLM(default_llm, **llm_factories)


__all__ = ["ConfigurableLLM", "configurable_llm"]
