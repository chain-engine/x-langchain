# -*- coding: utf-8 -*-
"""
中间件模块

为 Agent 运行过程提供各阶段的拦截和处理能力：
- 请求预处理（输入验证、清洗）
- 响应后处理（格式化、过滤）
- 执行监控（耗时、迭代、工具调用统计）
- 错误处理（统一异常转换）
- 钩子扩展点（before/after each step）
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable


class BaseMiddleware(ABC):
    """中间件基类"""

    name: str = "BaseMiddleware"

    @abstractmethod
    def before_invoke(self, context: dict[str, Any]) -> dict[str, Any]:
        """预处理"""

    @abstractmethod
    def after_invoke(self, context: dict[str, Any], response: Any) -> Any:
        """后处理"""

    def on_error(self, context: dict[str, Any], error: Exception) -> Exception:  # noqa: ARG002
        """错误处理，默认原样抛出"""
        return error


class InputValidationMiddleware(BaseMiddleware):
    """
    输入验证中间件

    校验和清洗用户输入，防止注入攻击。
    """

    name = "InputValidationMiddleware"

    def __init__(
        self,
        max_length: int = 10000,
        strip_whitespace: bool = True,
    ) -> None:
        self.max_length = max_length
        self.strip_whitespace = strip_whitespace

    def before_invoke(self, context: dict[str, Any]) -> dict[str, Any]:
        user_input = context.get("user_input", "")
        if not isinstance(user_input, str):
            raise ValueError("user_input 必须是字符串类型")

        if self.strip_whitespace:
            user_input = user_input.strip()

        if len(user_input) > self.max_length:
            raise ValueError(f"输入内容过长，最大支持 {self.max_length} 字符")

        context["user_input"] = user_input
        return context

    def after_invoke(self, context: dict[str, Any], response: Any) -> Any:
        return response


class TimingMiddleware(BaseMiddleware):
    """
    执行计时中间件

    记录每个迭代步骤的耗时，供后续分析和优化使用。
    """

    name = "TimingMiddleware"

    def before_invoke(self, context: dict[str, Any]) -> dict[str, Any]:
        context["_timing"] = {
            "start_time": time.time(),
            "step_name": context.get("step_name", "unknown"),
        }
        return context

    def after_invoke(self, context: dict[str, Any], response: Any) -> Any:
        timing = context.get("_timing", {})
        elapsed_ms = (time.time() - timing.get("start_time", time.time())) * 1000
        timing["elapsed_ms"] = elapsed_ms

        metrics = context.get("_metrics", {})
        step_name = timing.get("step_name", "unknown")
        metrics[f"step_{step_name}_ms"] = elapsed_ms
        context["_metrics"] = metrics

        return response


class IterationGuardMiddleware(BaseMiddleware):
    """
    迭代保护中间件

    防止 Agent 在单次请求中迭代次数过多导致资源耗尽。
    配置从 settings.middleware 读取
    """

    name = "IterationGuardMiddleware"

    def __init__(
        self,
        max_iterations: int | None = None,
        warn_threshold: float | None = None,
    ) -> None:
        from .config import settings
        self.max_iterations = max_iterations or settings.middleware.max_iterations
        self.warn_threshold = warn_threshold or settings.middleware.warn_threshold

    def before_invoke(self, context: dict[str, Any]) -> dict[str, Any]:
        iteration = context.get("iteration", 0) + 1
        context["iteration"] = iteration

        if iteration > self.max_iterations:
            raise RuntimeError(
                f"迭代次数 {iteration} 超过上限 {self.max_iterations}，Agent 已停止"
            )

        if iteration >= int(self.max_iterations * self.warn_threshold):
            context["_iteration_warning"] = True

        return context

    def after_invoke(self, context: dict[str, Any], response: Any) -> Any:
        return response


class ToolCallLoggerMiddleware(BaseMiddleware):
    """
    工具调用日志中间件

    记录每次工具调用的名称、参数和结果摘要。
    """

    name = "ToolCallLoggerMiddleware"

    def before_invoke(self, context: dict[str, Any]) -> dict[str, Any]:
        tool_name = context.get("tool_name")
        tool_args = context.get("tool_args", {})

        call_log = context.get("_tool_call_log", [])
        call_log.append({
            "name": tool_name,
            "args": tool_args,
            "status": "invoking",
        })
        context["_tool_call_log"] = call_log
        return context

    def after_invoke(self, context: dict[str, Any], response: Any) -> Any:
        call_log = context.get("_tool_call_log", [])
        if call_log:
            call_log[-1]["status"] = "completed"
            call_log[-1]["response_summary"] = self._summarize_response(response)
        return response

    def on_error(self, context: dict[str, Any], error: Exception) -> Exception:
        call_log = context.get("_tool_call_log", [])
        if call_log:
            call_log[-1]["status"] = "error"
            call_log[-1]["error"] = str(error)
        return error

    @staticmethod
    def _summarize_response(response: Any) -> str:
        if response is None:
            return "None"
        content = getattr(response, "content", None) or str(response)
        if len(content) > 200:
            return content[:200] + "..."
        return content


@dataclass
class MiddlewareChain:
    """
    中间件链

    将多个中间件串联起来，按顺序执行。
    """

    middlewares: list[BaseMiddleware] = field(default_factory=list)

    def add(self, middleware: BaseMiddleware) -> "MiddlewareChain":
        """追加中间件"""
        self.middlewares.append(middleware)
        return self

    def before_invoke(self, context: dict[str, Any]) -> dict[str, Any]:
        """依次执行所有中间件的 before_invoke"""
        for mw in self.middlewares:
            context = mw.before_invoke(context)
        return context

    def after_invoke(self, context: dict[str, Any], response: Any) -> Any:
        """反向依次执行所有中间件的 after_invoke"""
        for mw in reversed(self.middlewares):
            response = mw.after_invoke(context, response)
        return response

    def on_error(self, context: dict[str, Any], error: Exception) -> Exception:
        """反向依次执行所有中间件的 on_error"""
        for mw in reversed(self.middlewares):
            error = mw.on_error(context, error)
        return error

    def wrap(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """
        将一个函数包装为支持中间件的版本

        Args:
            func: 被包装的函数，签名为 (context) -> response

        Returns:
            包装后的函数
        """

        def wrapped(context: dict[str, Any]) -> Any:
            ctx = self.before_invoke(context)
            try:
                response = func(ctx)
                return self.after_invoke(ctx, response)
            except Exception as exc:
                raise self.on_error(ctx, exc) from exc

        return wrapped


DEFAULT_MIDDLEWARE_CHAIN = MiddlewareChain(
    middlewares=[
        InputValidationMiddleware(),
        TimingMiddleware(),
        IterationGuardMiddleware(),
        ToolCallLoggerMiddleware(),
    ]
)

__all__ = [
    "BaseMiddleware",
    "InputValidationMiddleware",
    "TimingMiddleware",
    "IterationGuardMiddleware",
    "ToolCallLoggerMiddleware",
    "MiddlewareChain",
    "DEFAULT_MIDDLEWARE_CHAIN",
]
