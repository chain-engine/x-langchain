# -*- coding: utf-8 -*-
"""
流式传输模式常量

本模块定义了 LangChain 流式传输的模式常量。

参考文档：https://langchain-doc.cn/v1/python/langchain/streaming.html
"""

from .base import BaseEnum


class StreamMode(BaseEnum):
    UPDATES = ("updates", "增量更新模式")
    CUSTOM = ("custom", "自定义模式")
    MESSAGES = ("messages", "消息模式")


DEFAULT_STREAM_MODES: list[StreamMode] = [StreamMode.UPDATES, StreamMode.CUSTOM, StreamMode.MESSAGES]
