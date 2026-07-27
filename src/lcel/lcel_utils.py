# -*- coding: utf-8 -*-
"""
LCEL 工具函数

提供 LCEL Runnable 组件的快捷封装。
"""

from __future__ import annotations

from typing import Any, Callable

from langchain_core.runnables import RunnableLambda as LCRunnableLambda
from langchain_core.runnables import RunnablePassthrough as LCRunnablePassthrough


class RunnableLambda:
    """
    Python 函数 -> Runnable 转换器

    包装任意 Python 函数为 Runnable，支持 LCEL 流水线组合。

    使用方式：
        ```python
        from lcel import RunnableLambda

        def to_upper(text: str) -> str:
            return text.upper()

        runnable = RunnableLambda(to_upper)
        result = runnable.invoke("hello")  # "HELLO"

        # LCEL 组合
        chain = prompt | llm | RunnableLambda(lambda x: x.content.upper())
        ```
    """

    def __init__(self, func: Callable[..., Any], *, name: str | None = None):
        self._func = func
        self._name = name or func.__name__
        self._runnable = LCRunnableLambda(func, name=name)

    def invoke(self, input: Any, **kwargs: Any) -> Any:
        return self._runnable.invoke(input, **kwargs)

    def __or__(self, other: Any) -> Any:
        return self._runnable | other

    def __ror__(self, other: Any) -> Any:
        return other | self._runnable

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._func(*args, **kwargs)

    def __repr__(self) -> str:
        return f"RunnableLambda({self._name})"


class RunnablePassthrough:
    """
    输入透传

    在 LCEL 流水线中将输入原样传递到下一步。

    使用方式：
        ```python
        from langchain_core.runnables import RunnablePassthrough

        chain = {
            "question": RunnablePassthrough(),
            "context": retriever,
        } | prompt | llm
        ```
    """

    def __init__(self, *, name: str | None = None):
        self._runnable = LCRunnablePassthrough(name=name)

    def invoke(self, input: Any, **kwargs: Any) -> Any:
        return self._runnable.invoke(input, **kwargs)

    def assign(self, **kwargs: Any) -> Any:
        """扩展输入（添加字段）"""
        return self._runnable.assign(**kwargs)

    def __or__(self, other: Any) -> Any:
        return self._runnable | other

    def __ror__(self, other: Any) -> Any:
        return other | self._runnable


__all__ = ["RunnableLambda", "RunnablePassthrough"]
