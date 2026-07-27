# -*- coding: utf-8 -*-
"""
带重试的健壮输出解析器

包装任意 Output Parser，在解析失败时自动重试。
"""

from __future__ import annotations

from typing import Any, Callable, Type

from langchain_core.runnables import Runnable, RunnableLambda

from core.logger import logger


class RetryOutputParser:
    """
    带重试能力的输出解析器

    包装一个基础的输出解析器，当解析失败时：
    1. 记录错误
    2. 抛出异常（由上层决定是否让 LLM 重试）

    在实际 Agent 场景中，可以在 middleware 层捕获异常后
    向 LLM 发送修正提示，实现自动重试。

    使用方式：
        ```python
        from output_parsers import JsonOutputParser, RetryOutputParser

        base_parser = JsonOutputParser()
        parser = RetryOutputParser(
            base_parser=base_parser,
            max_attempts=3,
            retry_on=(ValueError,),
        )

        # 解析失败时会记录 warning，可配合 Agent 实现自动重试
        result = parser.invoke(llm_output)
        ```
    """

    def __init__(
        self,
        base_parser: Any,
        *,
        max_attempts: int = 3,
        retry_on: Type[Exception] | tuple[Type[Exception], ...] = (ValueError,),
        on_retry: Callable[[Exception, int], None] | None = None,
    ):
        """
        初始化重试解析器

        Args:
            base_parser: 基础解析器（任意有 parse 方法的对象）
            max_attempts: 最大尝试次数
            retry_on: 需要触发重试的异常类型
            on_retry: 每次重试前的回调函数，签名为 (exception, attempt) -> None
        """
        self._base_parser = base_parser
        self._max_attempts = max_attempts
        self._retry_on = retry_on
        self._on_retry = on_retry
        self._attempt_count = 0

    @property
    def attempt_count(self) -> int:
        """当前尝试次数"""
        return self._attempt_count

    def parse(self, text: str) -> Any:
        """
        带重试的解析

        Args:
            text: LLM 原始输出

        Returns:
            解析后的对象

        Raises:
            最后一次解析的异常（如果所有尝试都失败）
        """
        last_error: Exception | None = None

        for attempt in range(1, self._max_attempts + 1):
            self._attempt_count = attempt
            try:
                result = self._base_parser.parse(text)
                if attempt > 1:
                    logger.info(f"解析成功（第 {attempt} 次尝试）")
                return result
            except self._retry_on as e:
                last_error = e
                logger.warning(f"解析失败（第 {attempt}/{self._max_attempts} 次）: {e}")
                if self._on_retry:
                    self._on_retry(e, attempt)
                continue
            except Exception as e:
                # 非预期异常不重试，直接抛出
                raise

        # 所有尝试都失败
        raise last_error from None

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        """Runnable 接口入口"""
        text = self._extract_text(input)
        return self.parse(text)

    def _extract_text(self, input: Any) -> str:
        if isinstance(input, str):
            return input
        if isinstance(input, dict):
            for key in ("text", "content", "output", "raw_output"):
                if key in input:
                    return str(input[key])
            return str(input)
        if hasattr(input, "content"):
            return str(input.content)
        return str(input)

    def __or__(self, other: Any) -> Any:
        def _parse_wrapper(text: Any) -> Any:
            return self.parse(text)

        return RunnableLambda(_parse_wrapper) | other

    def __rrshift__(self, other: Any) -> Any:
        def _parse_wrapper(text: Any) -> Any:
            return self.parse(text)

        return other | RunnableLambda(_parse_wrapper)

    def __repr__(self) -> str:
        return (
            f"RetryOutputParser(base={type(self._base_parser).__name__}, "
            f"max_attempts={self._max_attempts})"
        )


def create_retry_parser(
    base_parser: Any,
    *,
    max_attempts: int = 3,
    retry_on: Type[Exception] | tuple[Type[Exception], ...] = (ValueError,),
) -> RetryOutputParser:
    """
    工厂函数：创建带重试的解析器

    Args:
        base_parser: 基础解析器
        max_attempts: 最大重试次数
        retry_on: 需要重试的异常类型

    Returns:
        RetryOutputParser 实例
    """
    return RetryOutputParser(
        base_parser=base_parser,
        max_attempts=max_attempts,
        retry_on=retry_on,
    )


__all__ = ["RetryOutputParser", "create_retry_parser"]
