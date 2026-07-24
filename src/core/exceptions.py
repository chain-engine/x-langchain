# -*- coding: utf-8 -*-
"""
异常模块

定义项目全局的异常层次结构，包括：
- 基础异常类（XLangChainError）
- 配置异常
- 数据库异常
- LLM/Agent 异常
- 工具执行异常
- 规划执行异常
"""

from __future__ import annotations

from typing import Any, Optional


class XLangChainError(Exception):
    """
    项目基础异常类

    所有自定义异常的基类，统一异常处理和日志记录。
    """

    code: str = "UNKNOWN"
    message: str = "未知错误"

    def __init__(
        self,
        message: Optional[str] = None,
        code: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        self.message = message or self.__class__.message
        self.code = code or self.__class__.code
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.details:
            return f"[{self.code}] {self.message} | {self.details}"
        return f"[{self.code}] {self.message}"

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }


# region 配置异常


class ConfigError(XLangChainError):
    """配置相关错误"""

    code = "CONFIG_ERROR"
    message = "配置错误"


class ConfigValidationError(ConfigError):
    """配置校验失败"""

    code = "CONFIG_VALIDATION_ERROR"
    message = "配置校验失败"


class MissingConfigError(ConfigError):
    """缺少必需配置"""

    code = "MISSING_CONFIG"
    message = "缺少必需配置项"

    def __init__(
        self,
        missing_key: str,
        message: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        self.missing_key = missing_key
        details = details or {}
        details["missing_key"] = missing_key
        super().__init__(message=message, details=details)


# endregion

# region 数据库异常


class DatabaseError(XLangChainError):
    """数据库相关错误"""

    code = "DATABASE_ERROR"
    message = "数据库错误"


class DBConnectionError(DatabaseError):
    """数据库连接错误"""

    code = "DB_CONNECTION_ERROR"
    message = "数据库连接失败"


class DBSessionError(DatabaseError):
    """数据库会话错误"""

    code = "DB_SESSION_ERROR"
    message = "数据库会话错误"


class QueryExecutionError(DatabaseError):
    """查询执行错误"""

    code = "QUERY_EXECUTION_ERROR"
    message = "SQL 查询执行失败"

    def __init__(
        self,
        sql: Optional[str] = None,
        original_error: Optional[Exception] = None,
        message: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        self.sql = sql
        self.original_error = original_error
        details = details or {}
        if sql:
            details["sql"] = sql[:500]
        if original_error:
            details["original_error"] = str(original_error)
        super().__init__(message=message, details=details)


class SQLValidationError(DatabaseError):
    """SQL 校验失败（不安全或语法错误）"""

    code = "SQL_VALIDATION_ERROR"
    message = "SQL 校验失败"


# endregion

# region LLM / Agent 异常


class LLMError(XLangChainError):
    """LLM 相关错误"""

    code = "LLM_ERROR"
    message = "大语言模型调用错误"


class LLMConnectionError(LLMError):
    """LLM 连接错误"""

    code = "LLM_CONNECTION_ERROR"
    message = "无法连接到 LLM 服务"


class LLMTimeoutError(LLMError):
    """LLM 调用超时"""

    code = "LLM_TIMEOUT_ERROR"
    message = "LLM 调用超时"


class LLMResponseError(LLMError):
    """LLM 响应错误"""

    code = "LLM_RESPONSE_ERROR"
    message = "LLM 响应解析错误"


class InvalidLLMProviderError(LLMError):
    """不支持的 LLM 提供者"""

    code = "INVALID_LLM_PROVIDER"
    message = "不支持的 LLM 提供者"


class AgentError(XLangChainError):
    """Agent 执行错误"""

    code = "AGENT_ERROR"
    message = "Agent 执行错误"


class AgentMaxIterationsError(AgentError):
    """Agent 达到最大迭代次数"""

    code = "AGENT_MAX_ITERATIONS"
    message = "Agent 执行超过最大迭代次数"


class AgentConfigurationError(AgentError):
    """Agent 配置错误"""

    code = "AGENT_CONFIG_ERROR"
    message = "Agent 配置错误"


# endregion

# region 工具异常


class ToolError(XLangChainError):
    """工具相关错误"""

    code = "TOOL_ERROR"
    message = "工具执行错误"


class ToolNotFoundError(ToolError):
    """工具不存在"""

    code = "TOOL_NOT_FOUND"
    message = "未找到指定的工具"

    def __init__(
        self,
        tool_name: str,
        message: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        self.tool_name = tool_name
        details = details or {}
        details["tool_name"] = tool_name
        super().__init__(message=message, details=details)


class ToolExecutionError(ToolError):
    """工具执行失败"""

    code = "TOOL_EXECUTION_ERROR"
    message = "工具执行失败"

    def __init__(
        self,
        tool_name: Optional[str] = None,
        original_error: Optional[Exception] = None,
        message: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        self.tool_name = tool_name
        self.original_error = original_error
        details = details or {}
        if tool_name:
            details["tool_name"] = tool_name
        if original_error:
            details["original_error"] = str(original_error)
        super().__init__(message=message, details=details)


class ToolArgumentError(ToolError):
    """工具参数错误"""

    code = "TOOL_ARGUMENT_ERROR"
    message = "工具参数错误"


# endregion

# region 规划异常


class PlanningError(XLangChainError):
    """规划相关错误"""

    code = "PLANNING_ERROR"
    message = "任务规划错误"


class TaskExecutionError(PlanningError):
    """任务执行错误"""

    code = "TASK_EXECUTION_ERROR"
    message = "任务执行失败"


class PlanningTimeoutError(PlanningError):
    """规划超时"""

    code = "PLANNING_TIMEOUT"
    message = "任务规划超时"


# endregion

# region 记忆异常


class MemoryOperationError(XLangChainError):
    """记忆相关错误"""

    code = "MEMORY_ERROR"
    message = "记忆操作错误"


class MemoryPersistenceError(MemoryOperationError):
    """记忆持久化错误"""

    code = "MEMORY_PERSISTENCE_ERROR"
    message = "记忆持久化失败"


# endregion

# region 网络 / IO 异常


class NetworkError(XLangChainError):
    """网络相关错误"""

    code = "NETWORK_ERROR"
    message = "网络请求错误"


class HTTPError(NetworkError):
    """HTTP 请求错误"""

    code = "HTTP_ERROR"
    message = "HTTP 请求失败"

    def __init__(
        self,
        status_code: Optional[int] = None,
        response_body: Optional[str] = None,
        message: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        self.status_code = status_code
        self.response_body = response_body
        details = details or {}
        if status_code:
            details["status_code"] = status_code
        if response_body:
            details["response_body"] = response_body[:500]
        super().__init__(message=message, details=details)


# endregion
