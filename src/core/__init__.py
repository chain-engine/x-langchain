# -*- coding: utf-8 -*-
"""
核心模块包

包含配置管理、日志等核心功能。
"""

from .config import settings
from .logger import logger

__all__ = ["settings", "logger"]
