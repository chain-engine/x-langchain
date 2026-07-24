# -*- coding: utf-8 -*-
"""
工具模块包

使用插件化系统管理所有工具：
- 所有工具在模块加载时自动注册到 ToolRegistry
- 使用 @register_tool 装饰器注册新工具
- 使用 discover_tools() 自动发现工具目录中的工具
"""

import warnings
import os
from typing import Any, List

# =============================================================================
# 导入注册表和装饰器
# =============================================================================
from .registry import ToolRegistry, register_tool, discover_tools, print_registry_info

# =============================================================================
# MCP 工具（从 MCP 服务器动态获取）
# =============================================================================
from .qiuchi_mcp import (
    get_all_tools as get_qiuchi_tools,
    get_all_tools_async as get_qiuchi_tools_async,
)

# MCP may be temporarily unavailable in some deployments.
# When disabled, we skip registering MCP tools to avoid startup failures/timeouts.

# =============================================================================
# 工具获取接口
# =============================================================================
def get_function_calling_tools() -> List[Any]:
    """获取所有 Function Calling 工具"""
    return ToolRegistry.get_all(category="function_calling")


def get_function_calling_tools_by_subcategory(subcategory: str) -> List[Any]:
    """获取指定子类的 Function Calling 工具

    Args:
        subcategory: 子类名称，如 "text_to_sql"

    Returns:
        工具列表
    """
    return ToolRegistry.get_all(category="function_calling", subcategory=subcategory)


def register_mcp_tools() -> int:
    """注册所有 MCP 工具（秋池 + 天气）到 ToolRegistry

    Returns:
        成功注册的工具数量
    """
    from core.config import settings
    if not settings.MCP_ENABLED:
        return 0

    count = 0

    try:
        qiuchi_tools = get_qiuchi_tools()
        if qiuchi_tools:
            tool_names = [getattr(t, "name", str(t)) for t in qiuchi_tools]
            print(f"[秋池工具] 获取到 {len(qiuchi_tools)} 个工具: {tool_names}")
        for tool in qiuchi_tools:
            tool_name = getattr(tool, "name", str(tool))
            if not ToolRegistry.contains(tool_name):
                ToolRegistry.register(tool, name=tool_name, category="mcp", subcategory="qiuchi_mcp")
                count += 1
    except Exception as e:
        warnings.warn(f"获取秋池 MCP 工具失败: {e}")

    return count


async def register_mcp_tools_async() -> int:
    """异步注册所有 MCP 工具（秋池 + 天气）到 ToolRegistry

    Returns:
        成功注册的工具数量
    """
    from core.config import settings
    if not settings.MCP_ENABLED:
        return 0

    count = 0

    try:
        qiuchi_tools = await get_qiuchi_tools_async()
        if qiuchi_tools:
            tool_names = [getattr(t, "name", str(t)) for t in qiuchi_tools]
            print(f"[秋池工具] 获取到 {len(qiuchi_tools)} 个工具: {tool_names}")
        for tool in qiuchi_tools:
            tool_name = getattr(tool, "name", str(tool))
            if not ToolRegistry.contains(tool_name):
                ToolRegistry.register(tool, name=tool_name, category="mcp", subcategory="qiuchi_mcp")
                count += 1
    except Exception as e:
        warnings.warn(f"获取秋池 MCP 工具失败: {e}")

    return count


# =============================================================================
# 导出常用工具（简化导入）
# =============================================================================
from .weather_tool import weather_search_tool
from .web_tool import WebSearchTool, WebSearchTool as web_search_tool
from .exchange_rate_tool import exchange_rate_tool
from .calendar_tool import search_calendar, CalendarTool


def get_mcp_tools() -> List[Any]:
    """获取所有已注册的 MCP 工具（秋池 + 天气）"""
    return ToolRegistry.get_all(category="mcp")


def get_mcp_tools_by_subcategory(subcategory: str) -> List[Any]:
    """获取指定子类的 MCP 工具

    Args:
        subcategory: 子类名称，如 "weather_mcp", "qiuchi_mcp"

    Returns:
        工具列表
    """
    return ToolRegistry.get_all(category="mcp", subcategory=subcategory)


async def get_mcp_tools_async() -> List[Any]:
    """异步获取所有已注册的 MCP 工具（秋池 + 天气）"""
    return ToolRegistry.get_all(category="mcp")


def get_all_tools() -> List[Any]:
    """获取所有工具"""
    # 确保 Function Calling 工具已通过插件系统注册
    discover_function_calling_tools()
    # 确保 MCP 工具已注册
    register_mcp_tools()
    
    return [
        *get_function_calling_tools(),
        *get_mcp_tools(),
    ]


async def get_all_tools_async() -> List[Any]:
    """异步获取所有工具"""
    # 确保 Function Calling 工具已通过插件系统注册
    discover_function_calling_tools()
    #  注册所有的 MCP 工具
    await register_mcp_tools_async()
    
    fc_tools = get_function_calling_tools()
    mcp_tools = await get_mcp_tools_async()
    return fc_tools + mcp_tools


# =============================================================================
# 插件系统：自动发现工具
# =============================================================================
def discover_function_calling_tools() -> int:
    """
    自动发现并加载 Function Calling 工具

    这是插件系统的核心函数，它会：
    1. 扫描 tools 目录
    2. 识别所有 BaseTool 子类
    3. 自动推断 category 为 "function_calling"
    4. 注册到 ToolRegistry

    Returns:
        成功注册的工具数量
    """
    return discover_tools()


# =============================================================================
# 可导出
# =============================================================================
__all__ = [
    # 核心功能
    "ToolRegistry",
    "register_tool",
    "discover_tools",
    "discover_function_calling_tools",
    "print_registry_info",
    # 工具注册接口
    "register_mcp_tools",
    "register_mcp_tools_async",
    # 工具获取接口
    "get_function_calling_tools",
    "get_function_calling_tools_by_subcategory",
    "get_mcp_tools",
    "get_mcp_tools_by_subcategory",
    "get_mcp_tools_async",
    "get_all_tools",
    "get_all_tools_async",
    # 常用工具
    "weather_search_tool",
    "web_search_tool",
    "WebSearchTool",
    "exchange_rate_tool",
    "CalendarTool",
    "search_calendar",
]
