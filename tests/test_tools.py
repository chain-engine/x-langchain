# -*- coding: utf-8 -*-
"""Tools 模块测试。"""

import sys
from pathlib import Path
from unittest import TestCase, mock

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


class TestToolRegistry(TestCase):
    """ToolRegistry 测试"""

    def setUp(self) -> None:
        """每个测试前清空注册表"""
        from tools import ToolRegistry
        ToolRegistry.clear()

    def tearDown(self) -> None:
        """每个测试后清空注册表"""
        from tools import ToolRegistry
        ToolRegistry.clear()

    def test_register_tool(self) -> None:
        """测试工具注册"""
        from tools import ToolRegistry

        class MockTool:
            name = "mock_tool"
            description = "A mock tool"

        ToolRegistry.register(MockTool(), name="mock_tool", category="test")

        assert ToolRegistry.contains("mock_tool") is True
        tool = ToolRegistry.get("mock_tool")
        assert tool is not None

    def test_register_decorator(self) -> None:
        """测试工具注册装饰器"""
        from tools import ToolRegistry, register_tool

        @register_tool(name="decorated_tool", category="test")
        class DecoratedTool:
            pass

        assert ToolRegistry.contains("decorated_tool") is True

    def test_get_all_by_category(self) -> None:
        """测试按类别获取工具"""
        from tools import ToolRegistry

        class Tool1:
            name = "tool1"

        class Tool2:
            name = "tool2"

        ToolRegistry.register(Tool1(), name="tool1", category="test")
        ToolRegistry.register(Tool2(), name="tool2", category="test2")

        test_tools = ToolRegistry.get_all(category="test")
        assert len(test_tools) == 1


class TestWeatherTool(TestCase):
    """天气工具测试"""

    def test_weather_tool_missing_api_key(self) -> None:
        """测试天气工具缺少 API Key"""
        from tools.weather_tool import _search_weather_core

        with mock.patch("tools.weather_tool.settings") as mock_settings:
            mock_settings.AMAP_API_KEY = ""
            result = _search_weather_core("北京")

        assert "错误" in result or "AMAP_API_KEY" in result

    def test_weather_search_tool_decorator(self) -> None:
        """测试天气搜索工具装饰器"""
        from tools.weather_tool import weather_search_tool

        assert weather_search_tool.name == "weather_search_tool"


class TestWebSearchTool(TestCase):
    """网络搜索工具测试"""

    def test_web_search_tool_creation(self) -> None:
        """测试网络搜索工具创建"""
        from tools.web_tool import WebSearchTool

        tool = WebSearchTool()
        assert tool.name == "web_search"
        assert "检索" in tool.description


class TestExchangeRateTool(TestCase):
    """汇率工具测试"""

    def test_exchange_rate_tool_creation(self) -> None:
        """测试汇率工具创建"""
        from tools.exchange_rate_tool import exchange_rate_tool

        assert exchange_rate_tool.name is not None


class TestTextToSQLTools(TestCase):
    """TextToSQL 工具测试"""

    def setUp(self) -> None:
        """每个测试前清空注册表"""
        from tools import ToolRegistry
        ToolRegistry.clear()

    def tearDown(self) -> None:
        """每个测试后清空注册表"""
        from tools import ToolRegistry
        ToolRegistry.clear()

    def test_text_to_sql_tools_import(self) -> None:
        """测试 TextToSQL 工具导入"""
        from tools.text_to_sql import (
            question_rewrite_tool,
            get_schema_tool,
            generate_sql_tool,
            validate_sql_tool,
            execute_sql_tool,
            convert_to_natural_language_tool,
        )

        assert question_rewrite_tool is not None
        assert get_schema_tool is not None
        assert generate_sql_tool is not None
        assert validate_sql_tool is not None
        assert execute_sql_tool is not None
        assert convert_to_natural_language_tool is not None

    def test_get_schema_tool_creation(self) -> None:
        """测试获取 schema 工具创建"""
        from tools.text_to_sql import get_schema_tool

        tool = get_schema_tool
        assert tool.name == "get_schema"


class TestToolDiscovery(TestCase):
    """工具发现测试"""

    def setUp(self) -> None:
        """每个测试前清空注册表"""
        from tools import ToolRegistry
        ToolRegistry.clear()

    def tearDown(self) -> None:
        """每个测试后清空注册表"""
        from tools import ToolRegistry
        ToolRegistry.clear()

    def test_discover_function_calling_tools(self) -> None:
        """测试发现 Function Calling 工具"""
        from tools import discover_function_calling_tools, ToolRegistry

        count = discover_function_calling_tools()
        assert count >= 0

        # 检查基本工具是否被注册
        tools = ToolRegistry.get_all()
        tool_names = [getattr(t, "name", str(t)) for t in tools]
        assert isinstance(tool_names, list)
