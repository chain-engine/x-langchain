# -*- coding: utf-8 -*-
"""LCEL 中常用的 Runnable 组合工具。"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import (
    RunnableLambda,
    RunnablePassthrough,
    RunnableSequence,
)


def create_pipe_chain(*runnables: Any) -> RunnableSequence:
    """将多个 Runnable 通过 | 管道符串联成链。

    Args:
        *runnables: 按执行顺序排列的 Runnable 对象。

    Returns:
        由输入 Runnable 组合成的 RunnableSequence。

    Raises:
        ValueError: 未提供任何 Runnable 时抛出。
    """
    if not runnables:
        raise ValueError("At least one runnable is required")

    result = runnables[0]
    for runnable in runnables[1:]:
        result = result | runnable
    return result


def pipe_chain(runnables: list[Any]) -> RunnableSequence:
    """将列表中的 Runnable 通过管道符串联成链。

    Args:
        runnables: 按执行顺序排列的 Runnable 列表。

    Returns:
        由输入 Runnable 组合成的 RunnableSequence。

    Raises:
        ValueError: 列表为空时抛出。
    """
    return create_pipe_chain(*runnables)


__all__ = [
    "RunnableLambda",
    "RunnablePassthrough",
    "RunnableSequence",
    "create_pipe_chain",
    "pipe_chain",
]
