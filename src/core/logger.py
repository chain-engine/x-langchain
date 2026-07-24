# -*- coding: utf-8 -*-
"""
日志模块

提供统一的日志配置和记录功能。
配置现在统一从 core.config.settings 读取。
"""

import os
from typing import Callable, Final
from loguru import logger as _logger

from .config import settings


# 确保日志目录存在
log_dir: Final[str] = settings.logging.log_dir
os.makedirs(log_dir, exist_ok=True)

# 移除默认的控制台处理器，避免重复输出
_logger.remove()

# 配置日志输出到文件
_log_file_path = os.path.join(log_dir, "x-langchain_{time}.log")
_logger.add(
    _log_file_path,
    rotation=settings.logging.rotation,
    retention=settings.logging.retention,
    compression="zip",
    level=settings.logging.level,
    enqueue=True,
)

# 配置日志输出到控制台
console_sink: Callable[[str], None] = lambda msg: print(msg, end="")
_logger.add(
    sink=console_sink,
    level=settings.logging.console_level,
    enqueue=True,
)


def update_log_level(level: str) -> None:
    """动态更新日志级别"""
    _logger.remove()
    _logger.add(
        sink=console_sink,
        level=level,
        enqueue=True,
    )
    _logger.add(
        _log_file_path,
        rotation=settings.logging.rotation,
        retention=settings.logging.retention,
        compression="zip",
        level=level,
        enqueue=True,
    )


# 导出 logger 实例
logger = _logger

__all__: Final[list[str]] = ["logger", "update_log_level"]
