# -*- coding: utf-8 -*-
"""
行动执行器

实现具体的行动执行器。
"""

from typing import Any, Callable, Optional

from core.logger import logger

from .base import ActionResult, ActionType, BaseAction, BaseToolAction


class DirectResponseAction(BaseAction):
    """
    直接回复行动

    当 LLM 不需要调用工具时，直接返回回复内容。
    """

    action_type: ActionType = ActionType.DIRECT_RESPONSE

    def __init__(self, response_content: str):
        """
        初始化直接回复行动

        Args:
            response_content: 回复内容
        """
        self._content = response_content

    def should_execute(self, context: dict[str, Any]) -> bool:
        """检查是否不需要工具调用"""
        tool_calls = context.get("tool_calls", [])
        return len(tool_calls) == 0

    def execute(self, context: dict[str, Any]) -> ActionResult:
        """执行直接回复"""
        content = self._content or context.get("content", "")

        return ActionResult(
            action_type=self.action_type,
            success=True,
            content=content,
        )


class ToolCallAction(BaseAction):
    """
    工具调用行动

    当 LLM 需要调用工具时，执行工具调用。
    """

    action_type: ActionType = ActionType.TOOL_CALL

    def __init__(
        self,
        tool_registry: Optional[Any] = None,
    ):
        """
        初始化工具调用行动

        Args:
            tool_registry: 工具注册表
        """
        self._tool_registry = tool_registry

    @property
    def tool_registry(self) -> Any:
        """获取工具注册表"""
        if self._tool_registry is None:
            from tools import ToolRegistry
            return ToolRegistry
        return self._tool_registry

    def should_execute(self, context: dict[str, Any]) -> bool:
        """检查是否需要调用工具"""
        tool_calls = context.get("tool_calls", [])
        return len(tool_calls) > 0

    def execute(self, context: dict[str, Any]) -> ActionResult:
        """执行工具调用"""
        tool_calls = context.get("tool_calls", [])
        results = []
        errors = []

        for call in tool_calls:
            tool_name = call.get("name", "")
            tool_args = call.get("args", {})

            try:
                registry = self.tool_registry
                if hasattr(registry, "get"):
                    tool = registry.get(tool_name)
                else:
                    tool = None

                if tool:
                    if hasattr(tool, "invoke"):
                        result = tool.invoke(tool_args)
                    elif hasattr(tool, "run"):
                        result = tool.run(**tool_args)
                    else:
                        result = str(tool)

                    results.append({
                        "name": tool_name,
                        "args": tool_args,
                        "result": result,
                    })
                    logger.debug(f"工具 {tool_name} 执行成功")
                else:
                    error_msg = f"Tool {tool_name} not found"
                    errors.append(error_msg)
                    logger.warning(error_msg)

            except Exception as e:
                error_msg = f"Tool {tool_name} execution failed: {e}"
                errors.append(error_msg)
                logger.error(error_msg)

        if errors and not results:
            return ActionResult(
                action_type=self.action_type,
                success=False,
                error="; ".join(errors),
            )

        return ActionResult(
            action_type=self.action_type,
            success=True,
            tool_calls=results,
            content=self._format_tool_results(results),
        )

    def _format_tool_results(self, results: list[dict]) -> str:
        """格式化工具结果"""
        if not results:
            return ""

        formatted = []
        for r in results:
            tool_name = r.get("name", "unknown")
            result = r.get("result", "")
            formatted.append(f"[{tool_name}] {result}")

        return "\n".join(formatted)


class CompoundAction(BaseAction):
    """
    复合行动

    组合多个行动，按顺序执行。
    """

    action_type: ActionType = ActionType.COMPOUND

    def __init__(self, actions: Optional[list[BaseAction]] = None):
        """
        初始化复合行动

        Args:
            actions: 子行动列表
        """
        self._actions: list[BaseAction] = actions or []

    def add_action(self, action: BaseAction) -> "CompoundAction":
        """
        添加子行动

        Args:
            action: 行动实例

        Returns:
            self，支持链式调用
        """
        self._actions.append(action)
        return self

    def should_execute(self, context: dict[str, Any]) -> bool:
        """检查是否有行动需要执行"""
        return any(action.should_execute(context) for action in self._actions)

    def execute(self, context: dict[str, Any]) -> ActionResult:
        """顺序执行所有适用的行动"""
        all_results = []
        final_content = ""

        for action in self._actions:
            if action.should_execute(context):
                result = action.execute(context)
                all_results.append(result)

                if result.success and result.content:
                    final_content += result.content + "\n"

                if not result.success and not result.requires_tool_execution:
                    return ActionResult(
                        action_type=self.action_type,
                        success=False,
                        error=result.error,
                        metadata={"partial_results": [r.to_dict() for r in all_results]},
                    )

        return ActionResult(
            action_type=self.action_type,
            success=True,
            content=final_content.strip(),
            metadata={"actions_executed": len(all_results)},
        )

    def __len__(self) -> int:
        """返回子行动数量"""
        return len(self._actions)


class ActionExecutor:
    """
    行动执行器

    负责根据 LLM 输出决定执行哪种行动。
    """

    def __init__(
        self,
        tool_registry: Optional[Any] = None,
        enable_compound: bool = True,
    ):
        """
        初始化行动执行器

        Args:
            tool_registry: 工具注册表
            enable_compound: 是否启用复合行动
        """
        self._tool_registry = tool_registry
        self._enable_compound = enable_compound

        if enable_compound:
            self._default_action = CompoundAction([
                ToolCallAction(tool_registry=tool_registry),
            ])
        else:
            self._default_action = ToolCallAction(tool_registry=tool_registry)

    def execute(self, context: dict[str, Any]) -> ActionResult:
        """
        根据上下文执行行动

        Args:
            context: 上下文信息，包含 LLM 输出

        Returns:
            行动结果
        """
        tool_calls = context.get("tool_calls", [])

        if not tool_calls:
            content = context.get("content", "")
            return ActionResult(
                action_type=ActionType.DIRECT_RESPONSE,
                success=True,
                content=content,
            )

        return self._default_action.execute(context)

    def register_action(self, action: BaseAction) -> None:
        """
        注册行动

        Args:
            action: 行动实例
        """
        if isinstance(self._default_action, CompoundAction):
            self._default_action.add_action(action)
        else:
            compound = CompoundAction([self._default_action, action])
            self._default_action = compound

    def set_tool_registry(self, registry: Any) -> None:
        """设置工具注册表"""
        self._tool_registry = registry
