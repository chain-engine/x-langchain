# -*- coding: utf-8 -*-
"""
常量模块

本模块定义了项目中使用的各种常量。
"""

from enum import Enum

from .streaming_modes import StreamMode, DEFAULT_STREAM_MODES
from .develop import (
    HttpContentType,
    HashAlgorithm,
    SymmetricCipher,
    AsymmetricCipher,
    CipherMode,
    SymmetricPadding,
    AsymmetricPadding,
    EncodingType,
)


class AgentMode(str, Enum):
    """Agent 运行模式"""
    CHAIN = "chain"      # 简单链式调用
    REACT = "react"      # ReAct 推理模式
    PLAN = "plan"        # 规划模式


__all__ = [
    "StreamMode",
    "DEFAULT_STREAM_MODES",
    "HttpContentType",
    "HashAlgorithm",
    "SymmetricCipher",
    "AsymmetricCipher",
    "CipherMode",
    "SymmetricPadding",
    "AsymmetricPadding",
    "EncodingType",
    "AgentMode",
]
