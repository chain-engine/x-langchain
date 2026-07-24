# -*- coding: utf-8 -*-
"""
行动基类定义

定义行动系统的抽象接口和数据结构。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class ActionType(str, Enum):
    """行动类型枚举"""
    DIRECT_RESPONSE = "direct_response"
    TOOL_CALL = "tool_call"
    COMPOUND = "compound"
    PLANNING = "planning"


@dataclass
class ActionResult:
    """
    行动结果

    封装行动执行后的返回结果。
    """
    action_type: ActionType
    success: bool
    content: Optional[str] = None
    tool_calls: list[dict] = field(default_factory=list)
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "action_type": self.action_type.value,
            "success": self.success,
            "content": self.content,
            "tool_calls": self.tool_calls,
            "error": self.error,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }

    @property
    def requires_tool_execution(self) -> bool:
        """是否需要执行工具"""
        return self.action_type == ActionType.TOOL_CALL and bool(self.tool_calls)


class BaseAction(ABC):
    """
    行动基类

    定义行动的抽象接口。
    所有具体行动实现都需要继承此类。
    """

    action_type: ActionType = ActionType.DIRECT_RESPONSE

    @abstractmethod
    def execute(self, context: dict[str, Any]) -> ActionResult:
        """
        执行行动

        Args:
            context: 执行上下文，包含 LLM 输出等信息

        Returns:
            行动结果
        """
        pass

    @abstractmethod
    def should_execute(self, context: dict[str, Any]) -> bool:
        """
        判断是否应该执行此行动

        Args:
            context: 执行上下文

        Returns:
            是否执行
        """
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.action_type.value}>"


class BaseToolAction(BaseAction):
    """
    工具行动基类

    专门用于工具调用的行动基类。
    """

    action_type: ActionType = ActionType.TOOL_CALL

    def __init__(self, tool_name: str, tool_args: Optional[dict] = None):
        """
        初始化工具行动

        Args:
            tool_name: 工具名称
            tool_args: 工具参数
        """
        self._tool_name = tool_name
        self._tool_args = tool_args or {}

    @property
    def tool_name(self) -> str:
        """获取工具名称"""
        return self._tool_name

    @property
    def tool_args(self) -> dict:
        """获取工具参数"""
        return self._tool_args

    def should_execute(self, context: dict[str, Any]) -> bool:
        """检查上下文是否包含工具调用"""
        return "tool_calls" in context and len(context["tool_calls"]) > 0

    def execute(self, context: dict[str, Any]) -> ActionResult:
        """执行工具调用"""
        from tools import ToolRegistry

        tool_calls = context.get("tool_calls", [])

        for call in tool_calls:
            if call.get("name") == self._tool_name:
                try:
                    tool = ToolRegistry.get(self._tool_name)
                    if tool:
                        result = tool.invoke(call.get("args", {}))
                        return ActionResult(
                            action_type=self.action_type,
                            success=True,
                            content=str(result),
                            tool_calls=[{
                                "name": self._tool_name,
                                "args": call.get("args", {}),
                                "result": result,
                            }],
                        )
                except Exception as e:
                    return ActionResult(
                        action_type=self.action_type,
                        success=False,
                        error=str(e),
                    )

        return ActionResult(
            action_type=self.action_type,
            success=False,
            error=f"Tool {self._tool_name} not found or not called",
        )
