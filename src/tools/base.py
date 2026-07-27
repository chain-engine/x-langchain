# -*- coding: utf-8 -*-
"""
LangChain 工具基类

在 langchain_core.tools.BaseTool 基础上提供：
- JSON Schema 参数描述
- LangChain 标准格式导出
- 统一的工具元数据结构
"""

from __future__ import annotations

from typing import Any, Callable, Optional, get_type_hints

from langchain_core.tools import BaseTool
from pydantic import BaseModel

from core.logger import logger


class ToolCallError(Exception):
    """工具调用异常"""

    def __init__(self, tool_name: str, message: str):
        self.tool_name = tool_name
        super().__init__(f"[{tool_name}] {message}")


class BaseXTool(BaseTool):
    """
    x-langchain 统一工具基类

    在 langchain_core.tools.BaseTool 基础上扩展：
    - 自动生成 JSON Schema 参数描述
    - 提供 to_langchain_format() 导出 LangChain 兼容格式
    - 支持 retry_on_error 错误重试
    - 支持 usage tracking 使用统计

    子类必须：
    1. 定义 name 类属性（str）
    2. 定义 description 类属性（str）
    3. 实现 _run 方法
    """

    # 覆盖此属性以自定义参数 JSON Schema
    args_schema: type[BaseModel] | None = None

    # 错误重试次数（0 = 不重试）
    retry_count: int = 0

    # 示例输入（用于 LLM 理解）
    examples: list[dict] | None = None

    # 内部使用统计
    _call_count: int = 0
    _error_count: int = 0

    def _run(self, **kwargs) -> str:
        """子类必须实现此方法"""
        raise NotImplementedError("子类必须实现 _run 方法")

    def invoke(self, tool_input: str | dict, **kwargs) -> str:
        """
        调用工具（支持重试）

        Args:
            tool_input: 字符串或字典形式的输入
            **kwargs: 额外参数

        Returns:
            工具执行结果的字符串
        """
        import json

        if isinstance(tool_input, str):
            try:
                tool_input = json.loads(tool_input)
            except json.JSONDecodeError:
                pass

        params = {**(tool_input or {}), **kwargs}

        for attempt in range(self.retry_count + 1):
            try:
                self._call_count += 1
                result = self._run(**params)

                if isinstance(result, dict):
                    return json.dumps(result, ensure_ascii=False)
                return str(result) if result is not None else ""

            except Exception as e:
                self._error_count += 1
                if attempt < self.retry_count:
                    logger.warning(
                        f"[{self.name}] 调用失败（第 {attempt + 1} 次），重试: {e}"
                    )
                    continue
                raise ToolCallError(self.name, str(e)) from e

        return ""

    def to_langchain_format(self) -> dict[str, Any]:
        """
        导出为 LangChain 标准工具格式

        Returns:
            包含 name, description, schema 的字典
        """
        schema: dict[str, Any] = {"name": self.name, "description": self.description}

        if self.args_schema is not None:
            schema["parameters"] = self.args_schema.model_json_schema()
        elif hasattr(self, "args_schema") and self.args_schema is not None:
            try:
                schema["parameters"] = self.args_schema.schema()
            except Exception:
                pass

        if self.examples:
            schema["examples"] = self.examples

        return schema

    def get_usage_stats(self) -> dict[str, Any]:
        """获取工具使用统计"""
        return {
            "name": self.name,
            "call_count": self._call_count,
            "error_count": self._error_count,
            "success_rate": (
                (self._call_count - self._error_count) / self._call_count
                if self._call_count > 0
                else 0.0
            ),
        }

    def reset_stats(self) -> None:
        """重置使用统计"""
        self._call_count = 0
        self._error_count = 0


def create_tool_schema(
    name: str,
    description: str,
    parameters: type[BaseModel] | dict[str, Any] | None = None,
    examples: list[dict] | None = None,
) -> dict[str, Any]:
    """
    快捷工具：生成标准 JSON Schema 格式的工具定义

    Args:
        name: 工具名称
        description: 工具描述
        parameters: Pydantic 模型或 JSON Schema dict
        examples: 示例输入

    Returns:
        LangChain 格式的工具定义
    """
    schema: dict[str, Any] = {"name": name, "description": description}

    if parameters is not None:
        if isinstance(parameters, type) and issubclass(parameters, BaseModel):
            schema["parameters"] = parameters.model_json_schema()
        elif isinstance(parameters, dict):
            schema["parameters"] = parameters

    if examples:
        schema["examples"] = examples

    return schema


__all__ = [
    "BaseXTool",
    "ToolCallError",
    "create_tool_schema",
]
