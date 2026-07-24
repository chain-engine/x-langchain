# -*- coding: utf-8 -*-
"""
Agent 主类

整合 LLM、Memory、Planning、Action、Tools 五大子系统。
"""

import json
from typing import Any, Generator, List, Optional

from core.logger import logger

from constants import AgentMode
from .core import AgentConfig, AgentResponse
from ..actions import ActionDispatcher
from ..llms import create_chat_model
from ..memories import BaseMemory, ConversationHistoryMemory
from ..planning import PlanningManager
from ..tools import ToolRegistry


class Agent:
    """
    Agent 主类

    整合五大子系统：
    - LLM（模型）
    - Memory（记忆）
    - Planning（规划）
    - Action（行动）
    - Tools（工具）
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        llm: Optional[Any] = None,
        memory: Optional[BaseMemory] = None,
        planning_manager: Optional[PlanningManager] = None,
        action_dispatcher: Optional[ActionDispatcher] = None,
        tools: Optional[List[Any]] = None,
    ):
        """
        初始化 Agent

        Args:
            config: Agent 配置，None 则使用默认配置
            llm: LLM 实例，None 则根据 config 创建
            memory: 记忆实例，None 则创建默认记忆
            planning_manager: 规划管理器，None 则创建默认管理器
            action_dispatcher: 行动调度器，None 则创建默认调度器
            tools: 工具列表，None 则使用 ToolRegistry 中的所有工具
        """
        self._config = config or AgentConfig()
        self._iteration_count = 0

        # LLM 子系统
        self._llm = llm
        if self._llm is None:
            self._llm = create_chat_model(
                provider_name=self._config.model_provider,
                model_name=self._config.model_name,
                temperature=self._config.temperature,
                max_tokens=self._config.max_tokens,
            )

        # Memory 子系统
        self._memory = memory
        if self._memory is None and self._config.enable_memory:
            self._memory = ConversationHistoryMemory(
                system_message=self._config.system_prompt or "你是一个智能助手。",
                max_messages=100,
            )

        # Planning 子系统
        self._planning_manager = planning_manager
        if self._planning_manager is None and self._config.enable_planning:
            self._planning_manager = PlanningManager()

        # Tools 子系统
        self._tools = tools
        if self._tools is None and self._config.enable_tools:
            from ..tools import get_all_tools
            self._tools = get_all_tools()
            logger.info(f"加载 {len(self._tools)} 个工具")

        # Action 子系统
        self._action_dispatcher = action_dispatcher
        if self._action_dispatcher is None:
            self._action_dispatcher = ActionDispatcher()

        logger.info(f"初始化 Agent: {self._config.model_provider}, mode={self._config.mode.value}")

    @property
    def config(self) -> AgentConfig:
        """获取配置"""
        return self._config

    @property
    def memory(self) -> Optional[BaseMemory]:
        """获取记忆子系统"""
        return self._memory

    @property
    def planning_manager(self) -> Optional[PlanningManager]:
        """获取规划子系统"""
        return self._planning_manager

    @property
    def action_dispatcher(self) -> ActionDispatcher:
        """获取行动调度器"""
        return self._action_dispatcher

    @property
    def tools(self) -> Optional[List[Any]]:
        """获取工具列表"""
        return self._tools

    def set_memory(self, memory: BaseMemory) -> None:
        """设置记忆"""
        self._memory = memory

    def set_planning_manager(self, manager: PlanningManager) -> None:
        """设置规划管理器"""
        self._planning_manager = manager

    def set_tools(self, tools: List[Any]) -> None:
        """设置工具列表"""
        self._tools = tools

    def invoke(self, user_input: str, **kwargs) -> AgentResponse:
        """
        处理用户输入

        Args:
            user_input: 用户输入
            **kwargs: 额外参数

        Returns:
            Agent 响应
        """
        self._iteration_count = 0
        tool_results = []

        # 1. Memory: 记录用户消息
        if self._config.enable_memory and self._memory:
            self._memory.add_user_message(user_input)

        # 2. 获取消息列表
        messages = self._get_messages_for_llm()

        if self._config.debug:
            logger.debug(f"发送给 LLM 的消息: {len(messages)} 条")

        # 3. ReAct 循环
        while self._iteration_count < self._config.max_iterations:
            self._iteration_count += 1

            try:
                # 调用 LLM 子系统
                response = self._llm.invoke(messages)

                if self._config.debug:
                    logger.debug(f"LLM 响应: {response}")

                # 检查是否有工具调用
                if hasattr(response, "tool_calls") and response.tool_calls:
                    for tool_call in response.tool_calls:
                        # Action 子系统: 执行工具
                        result = self._execute_tool_call(tool_call)
                        tool_results.append({
                            "tool": tool_call.function.name,
                            "result": result,
                        })

                        # 构建工具消息
                        tool_message = {
                            "role": "tool",
                            "content": str(result),
                            "tool_call_id": getattr(tool_call, "id", None),
                        }
                        messages.append(tool_message)

                        # Memory: 记录工具结果
                        if self._config.enable_memory and self._memory:
                            self._memory.add_tool_message(
                                content=str(result),
                                tool_name=tool_call.function.name,
                            )

                    continue

                # 无工具调用，返回最终结果
                content = response.content if hasattr(response, "content") else str(response)

                # Memory: 记录助手回复
                if self._config.enable_memory and self._memory:
                    self._memory.add_assistant_message(content)

                return AgentResponse(
                    content=content,
                    success=True,
                    tool_results=tool_results,
                    iterations=self._iteration_count,
                    metadata={"model": self._config.model_provider},
                )

            except Exception as e:
                logger.error(f"Agent 执行错误: {e}")
                return AgentResponse(
                    content=f"抱歉，处理您的请求时出现错误: {str(e)}",
                    success=False,
                    tool_results=tool_results,
                    iterations=self._iteration_count,
                    metadata={"error": str(e)},
                )

        # 超过最大迭代次数
        return AgentResponse(
            content="抱歉，任务执行超过最大迭代次数。",
            success=False,
            tool_results=tool_results,
            iterations=self._iteration_count,
            metadata={"error": "max_iterations_exceeded"},
        )

    def stream(self, user_input: str, **kwargs) -> Generator:
        """
        流式处理用户输入

        Args:
            user_input: 用户输入
            **kwargs: 额外参数

        Yields:
            流式响应块
        """
        # Memory: 记录用户消息
        if self._config.enable_memory and self._memory:
            self._memory.add_user_message(user_input)

        messages = self._get_messages_for_llm()
        content_buffer = ""

        # LLM 流式调用
        for chunk in self._llm.stream(messages):
            if hasattr(chunk, "content"):
                content_buffer += chunk.content
                yield chunk

        # Memory: 记录助手回复
        if self._config.enable_memory and self._memory:
            self._memory.add_assistant_message(content_buffer)

    def _get_messages_for_llm(self) -> list[dict]:
        """获取发送给 LLM 的消息列表"""
        if self._config.enable_memory and self._memory:
            return self._memory.get_messages_for_llm()
        return []

    def _execute_tool_call(self, tool_call: Any) -> str:
        """
        执行工具调用（委托给 Action 子系统）

        Args:
            tool_call: 工具调用对象

        Returns:
            工具执行结果
        """
        tool_name = tool_call.function.name
        tool_args = self._parse_tool_args(tool_call.function.arguments)

        try:
            result = self._action_dispatcher.dispatch_tool_call(
                tool_name=tool_name,
                tool_args=tool_args,
            )

            if result.success:
                return result.content or str(result.tool_calls)
            else:
                return f"工具执行错误: {result.error}"

        except Exception as e:
            logger.error(f"工具 {tool_name} 执行失败: {e}")
            return f"工具执行失败: {str(e)}"

    def _parse_tool_args(self, arguments: Any) -> dict:
        """解析工具参数"""
        if isinstance(arguments, dict):
            return arguments

        if isinstance(arguments, str):
            try:
                return json.loads(arguments)
            except json.JSONDecodeError:
                return {"raw": arguments}

        return {"raw": str(arguments)}

    def reset(self) -> None:
        """重置 Agent 状态"""
        self._iteration_count = 0
        if self._memory:
            self._memory.clear()
        logger.debug("重置 Agent 状态")

    def __repr__(self) -> str:
        return f"<Agent: {self._config.model_provider}>"
