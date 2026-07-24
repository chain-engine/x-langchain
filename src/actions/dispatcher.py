# -*- coding: utf-8 -*-
"""
行动调度器

协调 LLM 输出和行动执行。
"""

from typing import Any, Callable, Optional

from core.logger import logger

from .base import ActionResult, ActionType
from .executors import ActionExecutor, CompoundAction, ToolCallAction


class ActionDispatcher:
    """
    行动调度器

    负责决策和调度行动：
    1. 分析 LLM 输出
    2. 判断是否需要工具调用
    3. 分发到对应的行动执行器
    """

    def __init__(
        self,
        tool_registry: Optional[Any] = None,
        action_executor: Optional[ActionExecutor] = None,
    ):
        """
        初始化行动调度器

        Args:
            tool_registry: 工具注册表
            action_executor: 行动执行器，None 则创建默认执行器
        """
        self._tool_registry = tool_registry
        self._action_executor = action_executor or ActionExecutor(
            tool_registry=tool_registry,
            enable_compound=True,
        )

        logger.debug("初始化行动调度器")

    @property
    def tool_registry(self) -> Any:
        """获取工具注册表"""
        if self._tool_registry is None:
            from tools import ToolRegistry
            return ToolRegistry
        return self._tool_registry

    @tool_registry.setter
    def tool_registry(self, value: Any) -> None:
        """设置工具注册表"""
        self._tool_registry = value
        self._action_executor.set_tool_registry(value)

    def analyze_llm_output(self, llm_output: Any) -> dict[str, Any]:
        """
        分析 LLM 输出

        Args:
            llm_output: LLM 输出

        Returns:
            解析后的上下文
        """
        context: dict[str, Any] = {
            "content": "",
            "tool_calls": [],
            "raw_output": llm_output,
        }

        if hasattr(llm_output, "tool_calls"):
            for call in llm_output.tool_calls:
                if hasattr(call, "function"):
                    context["tool_calls"].append({
                        "id": getattr(call, "id", None),
                        "name": call.function.name,
                        "args": self._parse_tool_args(call.function.arguments),
                    })
                elif isinstance(call, dict):
                    context["tool_calls"].append({
                        "id": call.get("id"),
                        "name": call.get("name", ""),
                        "args": call.get("args", {}),
                    })

        if hasattr(llm_output, "content"):
            context["content"] = llm_output.content

        logger.debug(
            f"分析 LLM 输出: {len(context['tool_calls'])} 个工具调用"
        )
        return context

    def _parse_tool_args(self, arguments: Any) -> dict:
        """解析工具参数"""
        if isinstance(arguments, dict):
            return arguments

        if isinstance(arguments, str):
            import json
            try:
                return json.loads(arguments)
            except json.JSONDecodeError:
                return {"raw": arguments}

        return {"raw": str(arguments)}

    def dispatch(self, llm_output: Any) -> ActionResult:
        """
        调度行动

        Args:
            llm_output: LLM 输出

        Returns:
            行动结果
        """
        context = self.analyze_llm_output(llm_output)
        result = self._action_executor.execute(context)

        logger.debug(f"行动调度完成: {result.action_type.value}")
        return result

    def dispatch_direct(self, content: str) -> ActionResult:
        """
        直接调度回复（不需要工具调用）

        Args:
            content: 回复内容

        Returns:
            行动结果
        """
        return ActionResult(
            action_type=ActionType.DIRECT_RESPONSE,
            success=True,
            content=content,
        )

    def dispatch_tool_call(
        self,
        tool_name: str,
        tool_args: dict,
    ) -> ActionResult:
        """
        直接调度工具调用

        Args:
            tool_name: 工具名称
            tool_args: 工具参数

        Returns:
            行动结果
        """
        context = {
            "tool_calls": [{
                "name": tool_name,
                "args": tool_args,
            }],
        }

        return self._action_executor.execute(context)

    def should_use_tools(self, context: dict[str, Any]) -> bool:
        """
        判断是否需要使用工具

        Args:
            context: 上下文信息

        Returns:
            是否需要工具
        """
        return len(context.get("tool_calls", [])) > 0

    def register_custom_action(
        self,
        condition: Callable[[dict[str, Any]], bool],
        action_type: str,
        handler: Callable[[dict[str, Any]], ActionResult],
    ) -> None:
        """
        注册自定义行动

        Args:
            condition: 条件函数
            action_type: 行动类型标识
            handler: 处理函数
        """
        custom_action = _ConditionalAction(
            condition=condition,
            action_type_str=action_type,
            handler=handler,
        )
        self._action_executor.register_action(custom_action)
        logger.debug(f"注册自定义行动: {action_type}")


class _ConditionalAction:
    """条件行动，用于自定义行动注册"""

    def __init__(
        self,
        condition: Callable[[dict[str, Any]], bool],
        action_type_str: str,
        handler: Callable[[dict[str, Any]], ActionResult],
    ):
        self._condition = condition
        self._action_type_str = action_type_str
        self._handler = handler

    def should_execute(self, context: dict[str, Any]) -> bool:
        return self._condition(context)

    def execute(self, context: dict[str, Any]) -> ActionResult:
        return self._handler(context)
