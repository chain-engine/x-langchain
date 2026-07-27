# -*- coding: utf-8 -*-
"""
Callback Handlers - LangChain 可观测性处理器

实现 langchain_core.callbacks.BaseCallbackHandler，
支持 LangSmith 追踪、Token 统计、耗时分析等。
"""

from __future__ import annotations

import time
from typing import Any, Optional
from dataclasses import dataclass, field

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult

from core.logger import logger


# =============================================================================
# 配置
# =============================================================================


@dataclass
class CallbackConfig:
    """Callback 配置"""

    # LangSmith
    langsmith_api_key: Optional[str] = None
    langsmith_project: str = "x-langchain"
    langsmith_enabled: bool = False

    # Token 统计
    track_token_usage: bool = True

    # 耗时统计
    track_timing: bool = True

    # Streaming 事件
    stream_to_stdout: bool = False


# =============================================================================
# Token 计数处理器
# =============================================================================


class TokenCountCallbackHandler(BaseCallbackHandler):
    """
    Token 使用统计处理器

    在 LLM 调用结束后统计 token 消耗。
    """

    def __init__(self):
        super().__init__()
        self._total_tokens: int = 0
        self._prompt_tokens: int = 0
        self._completion_tokens: int = 0
        self._request_count: int = 0

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """LLM 调用结束时统计 token"""
        for generation_chunk in response.generations:
            for gen in generation_chunk:
                if hasattr(gen, "usage_metadata") and gen.usage_metadata:
                    usage = gen.usage_metadata
                    self._prompt_tokens += usage.get("input_tokens", 0)
                    self._completion_tokens += usage.get("output_tokens", 0)
                    self._total_tokens += usage.get("total_tokens", 0)
                elif hasattr(gen, "generation_info"):
                    info = gen.generation_info or {}
                    self._prompt_tokens += info.get("prompt_tokens", 0)
                    self._completion_tokens += info.get("completion_tokens", 0)
                    self._total_tokens += info.get("total_tokens", 0)
        self._request_count += 1

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    @property
    def prompt_tokens(self) -> int:
        return self._prompt_tokens

    @property
    def completion_tokens(self) -> int:
        return self._completion_tokens

    @property
    def request_count(self) -> int:
        return self._request_count

    def get_summary(self) -> dict[str, Any]:
        """获取统计摘要"""
        return {
            "total_tokens": self._total_tokens,
            "prompt_tokens": self._prompt_tokens,
            "completion_tokens": self._completion_tokens,
            "request_count": self._request_count,
        }

    def reset(self) -> None:
        """重置计数器"""
        self._total_tokens = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._request_count = 0


# =============================================================================
# 耗时统计处理器
# =============================================================================


@dataclass
class TimingStats:
    """耗时统计数据"""

    llm_time: float = 0.0
    total_time: float = 0.0
    llm_call_count: int = 0
    tool_call_count: int = 0


class TimingCallbackHandler(BaseCallbackHandler):
    """
    耗时统计处理器

    记录 LLM 调用和工具调用的耗时。
    """

    def __init__(self):
        super().__init__()
        self._llm_start: float = 0.0
        self._total_start: float = 0.0
        self._llm_time: float = 0.0
        self._llm_call_count: int = 0
        self._tool_call_count: int = 0
        self._stats = TimingStats()

    def on_chain_start(self, serialized: dict[str, Any], inputs: dict[str, Any], **kwargs: Any) -> None:
        if "name" not in serialized or serialized.get("name") != "LLM":
            self._total_start = time.perf_counter()

    def on_chain_end(self, outputs: dict[str, Any], **kwargs: Any) -> None:
        if self._total_start > 0:
            self._stats.total_time = time.perf_counter() - self._total_start
            self._total_start = 0.0

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any) -> None:
        self._llm_start = time.perf_counter()
        self._llm_call_count += 1

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        if self._llm_start > 0:
            self._llm_time += time.perf_counter() - self._llm_start
            self._llm_start = 0.0
        self._stats.llm_time = self._llm_time
        self._stats.llm_call_count = self._llm_call_count

    def on_tool_start(self, serialized: dict[str, Any], inputs: str, **kwargs: Any) -> None:
        self._tool_call_count += 1
        self._stats.tool_call_count = self._tool_call_count

    def get_summary(self) -> dict[str, Any]:
        return {
            "llm_time_ms": round(self._llm_time * 1000, 2),
            "total_time_ms": round(self._stats.total_time * 1000, 2),
            "llm_call_count": self._llm_call_count,
            "tool_call_count": self._tool_call_count,
            "avg_llm_time_ms": (
                round(self._llm_time / self._llm_call_count * 1000, 2)
                if self._llm_call_count > 0
                else 0
            ),
        }

    def reset(self) -> None:
        self._llm_start = 0.0
        self._total_start = 0.0
        self._llm_time = 0.0
        self._llm_call_count = 0
        self._tool_call_count = 0
        self._stats = TimingStats()


# =============================================================================
# LangSmith 追踪处理器
# =============================================================================


class TracingCallbackHandler(BaseCallbackHandler):
    """
    LangSmith 追踪处理器

    将 LLM 调用记录发送到 LangSmith 进行追踪和分析。
    需要设置环境变量 LANGCHAIN_API_KEY。
    """

    def __init__(
        self,
        project_name: str = "x-langchain",
        api_key: Optional[str] = None,
        verbose: bool = False,
    ):
        super().__init__(verbose=verbose)
        self._project_name = project_name
        self._api_key = api_key or self._get_api_key()
        self._enabled = bool(self._api_key)

        if self._enabled:
            self._setup_tracing()

    def _get_api_key(self) -> Optional[str]:
        import os

        return os.environ.get("LANGCHAIN_API_KEY")

    def _setup_tracing(self) -> None:
        """配置 LangSmith 环境变量"""
        import os

        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGCHAIN_PROJECT", self._project_name)
        if self._api_key:
            os.environ.setdefault("LANGCHAIN_API_KEY", self._api_key)
        logger.info(f"LangSmith 追踪已启用: project={self._project_name}")

    @property
    def is_enabled(self) -> bool:
        return self._enabled


# =============================================================================
# Streaming 事件处理器
# =============================================================================


class StreamingCallbackHandler(BaseCallbackHandler):
    """
    流式事件处理器

    在流式 LLM 调用中捕获每个 token 并通过回调输出。
    """

    def __init__(
        self,
        on_token: Optional[callable] = None,
        on_complete: Optional[callable] = None,
        on_error: Optional[callable] = None,
    ):
        super().__init__()
        self._on_token = on_token
        self._on_complete = on_complete
        self._on_error = on_error
        self._buffer: list[str] = []
        self._error: Optional[Exception] = None

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """每个新 token 时触发"""
        self._buffer.append(token)
        if self._on_token:
            self._on_token(token)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """流结束时触发"""
        full_text = "".join(self._buffer)
        self._buffer = []
        if self._on_complete:
            self._on_complete(full_text)

    def on_llm_error(self, error: Exception | BaseException, **kwargs: Any) -> None:
        """出错时触发"""
        self._error = error
        if self._on_error:
            self._on_error(error)

    def get_buffer(self) -> str:
        """获取当前 buffer"""
        return "".join(self._buffer)

    def clear_buffer(self) -> None:
        self._buffer = []

    @property
    def error(self) -> Optional[Exception]:
        return self._error


# =============================================================================
# 多处理器组合
# =============================================================================


class MultiCallbackHandler:
    """
    多 Callback 处理器组合

    将多个 Callback Handler 组合为一个统一的处理器，
    自动分发给所有注册的 Handler。

    使用方式：
        ```python
        multi = MultiCallbackHandler()
        multi.add_handler(TokenCountCallbackHandler())
        multi.add_handler(TimingCallbackHandler())

        llm.invoke([...], callbacks=multi.get_handlers())
        ```
    """

    def __init__(self, handlers: Optional[list[BaseCallbackHandler]] = None):
        self._handlers: list[BaseCallbackHandler] = handlers or []

    def add_handler(self, handler: BaseCallbackHandler) -> None:
        """添加一个 Handler"""
        self._handlers.append(handler)

    def remove_handler(self, handler: BaseCallbackHandler) -> None:
        """移除一个 Handler"""
        if handler in self._handlers:
            self._handlers.remove(handler)

    def get_handlers(self) -> list[BaseCallbackHandler]:
        """获取所有 Handler（用于传入 LLM）"""
        return list(self._handlers)

    def get_handler(self, handler_type: type) -> Optional[BaseCallbackHandler]:
        """按类型获取 Handler"""
        for h in self._handlers:
            if isinstance(h, handler_type):
                return h
        return None

    def __len__(self) -> int:
        return len(self._handlers)


# =============================================================================
# 工厂函数
# =============================================================================


def get_default_callbacks(
    config: Optional[CallbackConfig] = None,
) -> list[BaseCallbackHandler]:
    """
    根据配置获取默认的 Callback Handlers

    Args:
        config: Callback 配置，为 None 时使用默认配置

    Returns:
        Callback Handler 列表
    """
    config = config or CallbackConfig()
    handlers: list[BaseCallbackHandler] = []

    if config.track_token_usage:
        handlers.append(TokenCountCallbackHandler())

    if config.track_timing:
        handlers.append(TimingCallbackHandler())

    if config.langsmith_enabled and config.langsmith_api_key:
        handlers.append(
            TracingCallbackHandler(
                project_name=config.langsmith_project,
                api_key=config.langsmith_api_key,
            )
        )

    return handlers


def create_callback_handler(
    handler_type: str,
    **kwargs: Any,
) -> BaseCallbackHandler:
    """
    工厂函数：创建指定类型的 Callback Handler

    Args:
        handler_type: 类型名（token / timing / tracing / streaming）
        **kwargs: 传递给 Handler 的参数

    Returns:
        BaseCallbackHandler 实例
    """
    handlers = {
        "token": TokenCountCallbackHandler,
        "timing": TimingCallbackHandler,
        "tracing": TracingCallbackHandler,
    }

    if handler_type not in handlers:
        raise ValueError(f"不支持的 handler 类型: {handler_type}，支持的: {list(handlers.keys())}")

    return handlers[handler_type](**kwargs)


__all__ = [
    "CallbackConfig",
    "TokenCountCallbackHandler",
    "TimingCallbackHandler",
    "TracingCallbackHandler",
    "StreamingCallbackHandler",
    "MultiCallbackHandler",
    "get_default_callbacks",
    "create_callback_handler",
]
