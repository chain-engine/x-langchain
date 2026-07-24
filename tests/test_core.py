# -*- coding: utf-8 -*-
"""Core 模块测试。"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


class TestSettings:
    """Settings 测试"""

    def test_settings_singleton(self) -> None:
        """测试 settings 单例"""
        from core.config import settings

        assert settings is not None
        assert hasattr(settings, "agent")
        assert hasattr(settings, "database")
        assert hasattr(settings, "logging")

    def test_settings_defaults(self) -> None:
        """测试默认配置"""
        from core.config import settings

        assert settings.agent.model_provider == "deepseek"
        assert settings.agent.max_iterations == 10
        assert settings.database.text_to_sql_max_rows == 100

    def test_settings_get_db_url(self) -> None:
        """测试数据库 URL 获取"""
        from core.config import settings

        url = settings.get_db_url()
        assert isinstance(url, str)
        assert "mysql" in url or url == ""

    def test_settings_validate_model_config(self) -> None:
        """测试模型配置验证"""
        from core.config import settings

        # Mock 总是有效
        assert settings.validate_model_config("mock") is True

        # 未知模型无效
        assert settings.validate_model_config("unknown") is False


class TestLogger:
    """Logger 测试"""

    def test_logger_exists(self) -> None:
        """测试 logger 存在"""
        from core.logger import logger

        assert logger is not None
        assert hasattr(logger, "info")
        assert hasattr(logger, "debug")
        assert hasattr(logger, "error")
        assert hasattr(logger, "warning")

    def test_logger_update_level(self) -> None:
        """测试更新日志级别"""
        from core.logger import update_log_level

        # 不应该抛出异常
        update_log_level("DEBUG")
        update_log_level("INFO")


class TestContainer:
    """Container 测试"""

    def test_container_singleton(self) -> None:
        """测试容器单例"""
        from core.container import container

        assert container is not None

    def test_container_reset(self) -> None:
        """测试容器重置"""
        from core.container import container

        container.reset()


class TestXLangChainError:
    """XLangChainError 测试"""

    def test_error_creation(self) -> None:
        """测试异常创建"""
        from core.exceptions import XLangChainError

        error = XLangChainError("Test error")
        assert str(error) == "[UNKNOWN] Test error"
        assert error.code == "UNKNOWN"

    def test_error_with_details(self) -> None:
        """测试带详情的异常"""
        from core.exceptions import XLangChainError

        error = XLangChainError(
            message="Test error",
            code="TEST_ERROR",
            details={"key": "value"},
        )

        assert error.code == "TEST_ERROR"
        assert error.details == {"key": "value"}

    def test_error_subclasses(self) -> None:
        """测试异常子类"""
        from core.exceptions import (
            LLMError,
            DatabaseError,
            ToolError,
            AgentError,
            ConfigError,
        )

        assert LLMError().code == "LLM_ERROR"
        assert DatabaseError().code == "DATABASE_ERROR"
        assert ToolError().code == "TOOL_ERROR"
        assert AgentError().code == "AGENT_ERROR"
        assert ConfigError().code == "CONFIG_ERROR"


class TestMiddleware:
    """Middleware 测试"""

    def test_input_validation_middleware(self) -> None:
        """测试输入验证中间件"""
        from core.middleware import InputValidationMiddleware

        mw = InputValidationMiddleware(max_length=100)
        context = {"user_input": "test"}

        result = mw.before_invoke(context)
        assert result["user_input"] == "test"

    def test_timing_middleware(self) -> None:
        """测试计时中间件"""
        from core.middleware import TimingMiddleware

        mw = TimingMiddleware()
        context = {"step_name": "test_step"}

        ctx_with_timing = mw.before_invoke(context)
        assert "_timing" in ctx_with_timing
        assert "start_time" in ctx_with_timing["_timing"]

    def test_iteration_guard_middleware(self) -> None:
        """测试迭代保护中间件"""
        from core.middleware import IterationGuardMiddleware

        mw = IterationGuardMiddleware(max_iterations=5)

        context = {"iteration": 3}
        result = mw.before_invoke(context)
        assert result["iteration"] == 4

    def test_middleware_chain(self) -> None:
        """测试中间件链"""
        from core.middleware import (
            MiddlewareChain,
            InputValidationMiddleware,
            TimingMiddleware,
        )

        chain = MiddlewareChain()
        chain.add(InputValidationMiddleware())
        chain.add(TimingMiddleware())

        context = {"user_input": "test"}
        result = chain.before_invoke(context)

        assert "_timing" in result
