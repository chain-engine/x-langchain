# -*- coding: utf-8 -*-
"""基于 LangGraph 的异步 Agent 封装。"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Sequence
from typing import Any, AsyncGenerator, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langgraph.prebuilt import create_react_agent

from agent.chat_history_service import ChatHistoryService, create_chat_history_service
from agent.lc_agent import AgentResponse, LCAgent
from core.logger import logger
from core.middleware import DEFAULT_MIDDLEWARE_CHAIN, MiddlewareChain
from llms import create_chat_model

_MISSING = object()
_DEFAULT_SYSTEM_MESSAGE = (
    "你是一个智能助手，可以帮助用户完成各种任务。当需要外部信息时，使用工具来获取。不要编造信息。"
)
_AGENT_OPTION_NAMES = {
    "response_format",
    "pre_model_hook",
    "post_model_hook",
    "context_schema",
    "checkpointer",
    "store",
    "interrupt_before",
    "interrupt_after",
    "debug",
    "version",
    "name",
}


class AsyncLCAgent(LCAgent):
    """提供异步调用、流式输出和历史持久化的 LangGraph Agent。"""

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        tools: Optional[Sequence[Any]] = None,
        system_message: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """初始化异步 Agent。

        Args:
            llm: 聊天模型；未提供时根据 ``config`` 创建默认模型。
            tools: Agent 可使用的工具列表，未提供时加载项目默认工具。
            system_message: Agent 的系统提示词。
            **kwargs: 与 LCAgent 相同的会话、中间件和状态配置；也可以通过
                ``agent_kwargs`` 传递额外的 LangGraph Agent 参数。
        """
        config = kwargs.pop("config", None)
        state_schema = kwargs.pop("state_schema", None)
        middleware_chain = kwargs.pop("middleware_chain", None)
        session_id = kwargs.pop("session_id", None)
        chat_history_service = kwargs.pop("chat_history_service", None)
        async_chat_history_service = kwargs.pop("async_chat_history_service", None)
        auto_persist = kwargs.pop("auto_persist", True)
        agent_kwargs = dict(kwargs.pop("agent_kwargs", {}) or {})
        for option_name in _AGENT_OPTION_NAMES:
            if option_name in kwargs:
                agent_kwargs[option_name] = kwargs.pop(option_name)

        if llm is None:
            from core.config import AgentConfig

            agent_config = config or AgentConfig()
            llm = create_chat_model(
                provider_name=agent_config.model_provider,
                model_name=agent_config.model_name,
                temperature=agent_config.temperature,
                max_tokens=agent_config.max_tokens,
            )

        self._llm = llm
        self._tools: Sequence[Any] = list(tools) if tools is not None else self._get_default_tools()
        self._system_message = system_message or _DEFAULT_SYSTEM_MESSAGE
        self._state_schema = state_schema
        self._middleware: MiddlewareChain = middleware_chain or DEFAULT_MIDDLEWARE_CHAIN
        self._session_id: Optional[str] = session_id
        self._chat_history_service: Optional[ChatHistoryService] = chat_history_service
        self._async_chat_history_service = async_chat_history_service
        self._auto_persist: bool = auto_persist
        self._agent_options = agent_kwargs

        create_kwargs = {
            "model": self._llm,
            "tools": self._tools,
            "prompt": self._system_message,
            "state_schema": self._state_schema,
            **self._agent_options,
        }
        self._agent: Runnable = create_react_agent(**create_kwargs)

        logger.info(
            f"初始化 AsyncLCAgent, tools={len(self._tools)}, "
            f"session_id={self._session_id}, auto_persist={self._auto_persist}"
        )

    @staticmethod
    async def _maybe_await(value: Any) -> Any:
        """兼容同步和异步扩展点的返回值。"""
        if inspect.isawaitable(value):
            return await value
        return value

    async def _get_history_service_async(self) -> Any:
        """异步获取历史服务，优先使用调用方注入的服务。"""
        if self._async_chat_history_service is not None:
            return self._async_chat_history_service
        if self._chat_history_service is not None:
            return self._chat_history_service

        try:
            factory = create_chat_history_service
            if inspect.iscoroutinefunction(factory):
                service = await factory()
            else:
                service = await asyncio.to_thread(factory)
            self._chat_history_service = service
            return service
        except Exception as exc:
            logger.warning(f"无法创建异步 ChatHistoryService: {exc}")
            return None

    @staticmethod
    async def _call_service(
        service: Any,
        method_names: tuple[str, ...],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """调用历史服务的异步或同步方法。"""
        for method_name in method_names:
            method = getattr(service, method_name, None)
            if method is None:
                continue

            if inspect.iscoroutinefunction(method):
                result = method(*args, **kwargs)
            else:
                result = await asyncio.to_thread(method, *args, **kwargs)
            if inspect.isawaitable(result):
                result = await result
            return result
        return _MISSING

    async def aget_history(self, session_id: str) -> list:
        """异步加载会话历史并转换为 LangChain 消息元组。

        Args:
            session_id: 要读取的会话 ID。

        Returns:
            按时间顺序排列的 ``(role, content)`` 消息列表。
        """
        service = await self._get_history_service_async()
        if service is None:
            return []

        try:
            messages = await self._call_service(
                service,
                ("aget_messages", "get_messages_async", "get_messages"),
                session_id,
            )
            if messages is _MISSING:
                return []

            history = []
            for message in messages:
                if isinstance(message, dict):
                    role = message.get("role")
                    content = message.get("content", "")
                else:
                    role = getattr(message, "role", None)
                    content = getattr(message, "content", "")
                history.append(("user" if role == "user" else "ai", content))

            logger.debug(f"异步加载 {len(history)} 条历史消息到会话 {session_id}")
            return history
        except Exception as exc:
            logger.warning(f"加载会话历史失败: {exc}")
            return []

    async def asave_message(
        self,
        session_id: str,
        role: str,
        content: str,
    ) -> None:
        """异步保存一条会话消息。

        Args:
            session_id: 会话 ID。
            role: 消息角色，例如 ``user`` 或 ``assistant``。
            content: 消息内容。
        """
        service = await self._get_history_service_async()
        if service is None:
            return

        model_provider = self._llm.__class__.__name__ if self._llm is not None else "unknown"
        try:
            result = await self._call_service(
                service,
                ("aadd_message", "add_message_async", "add_message"),
                session_id=session_id,
                role=role,
                content=content,
                model_provider=model_provider,
            )
            if result is _MISSING:
                return
        except TypeError:
            try:
                await self._call_service(
                    service,
                    ("aadd_message", "add_message_async", "add_message"),
                    session_id,
                    role,
                    content,
                )
            except Exception as exc:
                logger.warning(f"保存消息失败: {exc}")
        except Exception as exc:
            logger.warning(f"保存消息失败: {exc}")

    @staticmethod
    def _session_from_kwargs(kwargs: dict[str, Any], default: Optional[str]) -> Optional[str]:
        """从调用参数中解析会话 ID。"""
        return kwargs.get("session_id") or default

    @staticmethod
    def _graph_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
        """移除 Agent 封装参数，保留 LangGraph 调用参数。"""
        graph_kwargs = dict(kwargs)
        graph_kwargs.pop("session_id", None)
        graph_kwargs.pop("config", None)
        return graph_kwargs

    async def _prepare_messages(
        self,
        user_input: str,
        session_id: Optional[str],
    ) -> list:
        """异步加载历史并拼接本轮用户消息。"""
        history = []
        if self._auto_persist and session_id:
            history = await self.aget_history(session_id)

        messages = [("user", user_input)]
        if history:
            messages = history + messages
        return messages

    async def ainvoke(self, user_input: str, **kwargs: Any) -> AgentResponse:
        """异步处理用户输入并返回 AgentResponse。"""
        session_id = self._session_from_kwargs(kwargs, self._session_id)
        context = {
            "user_input": user_input,
            "iteration": 0,
            "_metrics": {},
            "_tool_call_log": [],
        }

        try:
            context = await self._maybe_await(self._middleware.before_invoke(context))
            messages = await self._prepare_messages(user_input, session_id)
            config = kwargs.get("config", {})
            result = await self._agent.ainvoke(
                {"messages": messages},
                config=config,
                **self._graph_kwargs(kwargs),
            )
            context["_last_result"] = result
            context["_iterations"] = self._count_ai_messages(result)
            context["_content"] = self._extract_content(result)
            context["_tool_results"] = self._extract_tool_results(result)
            await self._maybe_await(self._middleware.after_invoke(context, result))

            if self._auto_persist and session_id:
                await self.asave_message(session_id, "user", user_input)
                await self.asave_message(session_id, "assistant", context["_content"])

            return AgentResponse(
                content=context["_content"],
                success=True,
                tool_results=context.get("_tool_results", []),
                iterations=context["_iterations"],
                metadata={
                    "agent_type": "langgraph.react.async",
                    "metrics": context.get("_metrics", {}),
                },
            )
        except Exception as exc:
            error = await self._maybe_await(self._middleware.on_error(context, exc))
            logger.error(f"异步 Agent 执行错误: {error}")
            return AgentResponse(
                content=f"处理请求时出错: {str(error)}",
                success=False,
                metadata={"error": str(error)},
            )

    @classmethod
    def _content_from_value(cls, value: Any) -> str:
        """从消息、状态或事件中提取文本内容。"""
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if hasattr(value, "content"):
            return cls._content_from_value(getattr(value, "content"))
        if isinstance(value, list):
            parts = []
            for item in value:
                if isinstance(item, dict) and "text" in item:
                    parts.append(str(item["text"]))
                else:
                    parts.append(cls._content_from_value(item))
            return "".join(parts)
        if isinstance(value, dict):
            for key in ("messages", "output", "chunk", "content"):
                if key in value:
                    candidate = value[key]
                    if key == "messages" and candidate:
                        return cls._content_from_value(candidate[-1])
                    text = cls._content_from_value(candidate)
                    if text:
                        return text
            for candidate in reversed(list(value.values())):
                text = cls._content_from_value(candidate)
                if text:
                    return text
        return ""

    async def _stream_agent(
        self,
        method_name: str,
        input_data: dict[str, Any],
        config: Any,
        graph_kwargs: dict[str, Any],
    ) -> AsyncGenerator[Any, None]:
        """统一迭代 LangGraph 的异步流接口。"""
        method = getattr(self._agent, method_name)
        stream = method(input_data, config=config, **graph_kwargs)
        if inspect.isawaitable(stream):
            stream = await stream
        if hasattr(stream, "__aiter__"):
            async for event in stream:
                yield event
        else:
            for event in stream:
                yield event

    async def astream(self, user_input: str, **kwargs: Any) -> AsyncGenerator[Any, None]:
        """异步流式处理用户输入并逐个产出 LangGraph 状态事件。"""
        session_id = self._session_from_kwargs(kwargs, self._session_id)
        context = {
            "user_input": user_input,
            "iteration": 0,
            "_metrics": {},
            "_tool_call_log": [],
        }
        last_event: Any = None

        try:
            context = await self._maybe_await(self._middleware.before_invoke(context))
            messages = await self._prepare_messages(user_input, session_id)
            async for event in self._stream_agent(
                "astream",
                {"messages": messages},
                kwargs.get("config", {}),
                self._graph_kwargs(kwargs),
            ):
                last_event = event
                yield event

            await self._maybe_await(self._middleware.after_invoke(context, last_event))
            if self._auto_persist and session_id:
                response = self._content_from_value(last_event)
                await self.asave_message(session_id, "user", user_input)
                await self.asave_message(session_id, "assistant", response)
        except Exception as exc:
            error = await self._maybe_await(self._middleware.on_error(context, exc))
            logger.error(f"异步 Agent 流式处理错误: {error}")
            yield {"type": "error", "error": str(error)}

    async def astream_events(self, user_input: str, **kwargs: Any) -> AsyncGenerator[Any, None]:
        """异步转发 LangGraph 的完整 astream_events 事件流。"""
        session_id = self._session_from_kwargs(kwargs, self._session_id)
        context = {
            "user_input": user_input,
            "iteration": 0,
            "_metrics": {},
            "_tool_call_log": [],
        }
        last_event: Any = None

        try:
            context = await self._maybe_await(self._middleware.before_invoke(context))
            messages = await self._prepare_messages(user_input, session_id)
            async for event in self._stream_agent(
                "astream_events",
                {"messages": messages},
                kwargs.get("config", {}),
                self._graph_kwargs(kwargs),
            ):
                last_event = event
                yield event

            await self._maybe_await(self._middleware.after_invoke(context, last_event))
            if self._auto_persist and session_id:
                response = self._content_from_value(last_event)
                await self.asave_message(session_id, "user", user_input)
                await self.asave_message(session_id, "assistant", response)
        except Exception as exc:
            error = await self._maybe_await(self._middleware.on_error(context, exc))
            logger.error(f"异步 Agent 事件流处理错误: {error}")
            yield {"type": "error", "error": str(error)}

    def __repr__(self) -> str:
        """返回 Agent 的简要描述。"""
        return f"<AsyncLCAgent: tools={len(self._tools)}>"


def create_async_agent(
    llm: Optional[BaseChatModel] = None,
    tools: Optional[Sequence[Any]] = None,
    system_message: Optional[str] = None,
    **kwargs: Any,
) -> AsyncLCAgent:
    """创建异步 LangGraph Agent。"""
    return AsyncLCAgent(
        llm=llm,
        tools=tools,
        system_message=system_message,
        **kwargs,
    )


__all__ = ["AsyncLCAgent", "create_async_agent"]
