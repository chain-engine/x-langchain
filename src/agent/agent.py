# -*- coding: utf-8 -*-
"""
Agent 主类

整合 LLM、Memory、Planning、Action、Tools 五大子系统。
"""

import json
from dataclasses import dataclass, field
from typing import Any, Generator, List, Optional

from core.logger import logger

from constants import AgentMode
from core.config import AgentConfig
from actions import ActionDispatcher
from llms import create_chat_model
from memories import BaseMemory, ConversationHistoryMemory
from planning import PlanningManager
from tools import ToolRegistry


@dataclass
class AgentResponse:
    """Agent 响应"""
    content: str
    success: bool
    tool_results: List[dict] = field(default_factory=list)
    iterations: int = 0
    metadata: dict = field(default_factory=dict)
    plan: Optional[list[dict]] = None  # 任务计划（如果有）
    execution_summary: Optional[dict] = None  # 执行摘要
    intent: Optional[str] = None  # 识别的用户意图


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

        self._init_subsystems(
            llm=llm,
            memory=memory,
            planning_manager=planning_manager,
            action_dispatcher=action_dispatcher,
            tools=tools,
        )

        logger.info(f"初始化 Agent: {self._config.model_provider}, mode={self._get_mode()}")

    def _get_mode(self) -> AgentMode:
        """获取 AgentMode，确保类型正确"""
        if isinstance(self._config.mode, AgentMode):
            return self._config.mode
        if isinstance(self._config.mode, str):
            return AgentMode.from_value(self._config.mode)
        return AgentMode.REACT

    def _init_subsystems(
        self,
        llm: Optional[Any] = None,
        memory: Optional[BaseMemory] = None,
        planning_manager: Optional[PlanningManager] = None,
        action_dispatcher: Optional[ActionDispatcher] = None,
        tools: Optional[List[Any]] = None,
    ) -> None:
        """
        初始化五大子系统

        Args:
            llm: LLM 实例
            memory: 记忆实例
            planning_manager: 规划管理器
            action_dispatcher: 行动调度器
            tools: 工具列表
        """
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
            if self._config.enable_memory_persistence and self._config.session_id:
                # 启用持久化记忆
                from memories import PersistentMemory, SQLiteMemoryStore
                store = None
                if self._config.memory_store_path:
                    store = SQLiteMemoryStore(self._config.memory_store_path)
                self._memory = PersistentMemory(
                    session_id=self._config.session_id,
                    store=store,
                    max_messages=100,
                    system_message=self._config.system_prompt or "你是一个智能助手。",
                )
                logger.info(f"启用持久化记忆，会话ID: {self._config.session_id}")
            else:
                self._memory = ConversationHistoryMemory(
                    system_message=self._config.system_prompt or "你是一个智能助手。",
                    max_messages=100,
                )

        # Planning 子系统（始终创建，由 invoke 根据模式决定是否使用）
        self._planning_manager = planning_manager
        if self._planning_manager is None:
            # 传递 LLM 以支持智能规划
            self._planning_manager = PlanningManager(llm=self._llm)

        # Tools 子系统
        self._tools = tools
        if self._tools is None and self._config.enable_tools:
            from tools import get_all_tools
            self._tools = get_all_tools()
            logger.info(f"加载 {len(self._tools)} 个工具")

        # Action 子系统
        self._action_dispatcher = action_dispatcher
        if self._action_dispatcher is None:
            self._action_dispatcher = ActionDispatcher()

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
        处理用户输入（混合模式：ReAct + Planning + 意图识别）

        流程：
        1. 意图识别与参数标准化
        2. 简单任务：直接走 ReAct 循环
        3. 复杂任务：先用 Planning 分解，按计划执行

        Args:
            user_input: 用户输入
            **kwargs: 额外参数

        Returns:
            Agent 响应
        """
        self._iteration_count = 0
        tool_results = []

        # 0. 意图识别（可选，debug 模式输出）
        intent_result = None
        if self._config.debug:
            from .intent import IntentRecognizer
            recognizer = IntentRecognizer()
            intent_result = recognizer.recognize(user_input)
            logger.debug(
                f"意图识别: intent={intent_result.intent.value}, "
                f"confidence={intent_result.confidence:.2f}, "
                f"entities={intent_result.entities}"
            )

        # 1. Memory: 记录用户消息
        if self._config.enable_memory and self._memory:
            self._memory.add_user_message(user_input)

        # 2. 获取消息列表
        messages = self._get_messages_for_llm()

        if self._config.debug:
            logger.debug(f"发送给 LLM 的消息: {len(messages)} 条")

        # 3. 混合模式决策
        if self._get_mode() == AgentMode.PLAN and self._config.enable_planning:
            # Plan 模式：使用 Planning 模块分解任务
            return self._invoke_with_planning(
                user_input, messages, tool_results, intent_result
            )

        # ReAct 模式：直接循环推理
        return self._invoke_react_loop(
            user_input, messages, tool_results, intent_result
        )

    def _invoke_react_loop(
        self,
        user_input: str,
        messages: list,
        tool_results: list,
        intent_result: Any = None,
    ) -> AgentResponse:
        """
        ReAct 推理循环

        循环：思考 → 行动 → 观测，直到得到最终答案
        """
        while self._iteration_count < self._config.max_iterations:
            self._iteration_count += 1

            try:
                # 调用 LLM
                response = self._llm.invoke(messages)

                if self._config.debug:
                    logger.debug(f"LLM 响应: {response}")

                # 检查工具调用
                if hasattr(response, "tool_calls") and response.tool_calls:
                    for tool_call in response.tool_calls:
                        result = self._execute_tool_call(tool_call)
                        tool_results.append({
                            "tool": tool_call.function.name,
                            "result": result,
                            "iteration": self._iteration_count,
                        })

                        tool_message = {
                            "role": "tool",
                            "content": str(result),
                            "tool_call_id": getattr(tool_call, "id", None),
                        }
                        messages.append(tool_message)

                        if self._config.enable_memory and self._memory:
                            self._memory.add_tool_message(
                                content=str(result),
                                tool_name=tool_call.function.name,
                            )

                    continue

                # 无工具调用，返回最终结果
                content = response.content if hasattr(response, "content") else str(response)

                if self._config.enable_memory and self._memory:
                    self._memory.add_assistant_message(content)

                return AgentResponse(
                    content=content,
                    success=True,
                    tool_results=tool_results,
                    iterations=self._iteration_count,
                    metadata={"mode": "react", "model": self._config.model_provider},
                    intent=intent_result.intent.value if intent_result else None,
                )

            except Exception as e:
                logger.error(f"Agent 执行错误: {e}")
                return AgentResponse(
                    content=f"抱歉，处理您的请求时出现错误: {str(e)}",
                    success=False,
                    tool_results=tool_results,
                    iterations=self._iteration_count,
                    metadata={"mode": "react", "error": str(e)},
                    intent=intent_result.intent.value if intent_result else None,
                )

        return AgentResponse(
            content="抱歉，任务执行超过最大迭代次数。",
            success=False,
            tool_results=tool_results,
            iterations=self._iteration_count,
            metadata={"mode": "react", "error": "max_iterations_exceeded"},
            intent=intent_result.intent.value if intent_result else None,
        )

    def _invoke_with_planning(
        self,
        user_input: str,
        messages: list,
        tool_results: list,
        intent_result: Any = None,
    ) -> AgentResponse:
        """
        Planning 模式：任务分解 + 计划执行

        1. 使用 LLM 分析任务复杂度
        2. 生成任务计划（子任务分解）
        3. 按计划执行每个子任务
        4. 汇总结果返回
        """
        try:
            # 3.1 任务分析：判断是否需要工具调用
            analysis = self._planning_manager.analyze(user_input, context={
                "messages": messages,
            })

            if not analysis.get("needs_tools"):
                # 无需工具，直接回答
                return self._invoke_react_loop(user_input, messages, tool_results)

            # 3.2 生成任务计划
            tasks = self._planning_manager.plan(user_input, context={
                "messages": messages,
            })

            if not tasks:
                # 规划器未生成有效计划，回退到 ReAct
                logger.info("Planning 未生成有效计划，回退到 ReAct 模式")
                return self._invoke_react_loop(user_input, messages, tool_results)

            logger.info(f"任务计划: {len(tasks)} 个子任务")
            plan_summary = [task.to_dict() for task in tasks]

            # 3.3 按计划执行子任务
            execution_results = self._execute_plan(tasks, messages, tool_results)

            # 3.4 汇总结果，生成最终回复
            final_response = self._synthesize_response(
                user_input,
                messages,
                execution_results,
                tool_results,
            )

            return AgentResponse(
                content=final_response,
                success=True,
                tool_results=tool_results,
                iterations=self._iteration_count,
                metadata={
                    "mode": "plan",
                    "model": self._config.model_provider,
                    "task_count": len(tasks),
                },
                plan=plan_summary,
                execution_summary={
                    "total_tasks": len(tasks),
                    "completed_tasks": len([r for r in execution_results if r.success]),
                    "failed_tasks": len([r for r in execution_results if not r.success]),
                },
                intent=intent_result.intent.value if intent_result else None,
            )

        except Exception as e:
            logger.error(f"Planning 模式执行错误: {e}")
            # 回退到 ReAct 模式
            return self._invoke_react_loop(user_input, messages, tool_results, intent_result)

    def _execute_plan(
        self,
        tasks: list,
        messages: list,
        tool_results: list,
    ) -> list:
        """
        执行任务计划

        Args:
            tasks: 任务列表
            messages: 消息历史
            tool_results: 工具结果列表

        Returns:
            任务结果列表
        """
        from planning.base import TaskResult

        results = []
        task_outputs = {}

        for task in tasks:
            self._iteration_count += 1

            if self._iteration_count >= self._config.max_iterations:
                task.mark_failed("max_iterations_exceeded")
                results.append(TaskResult(
                    task_id=task.id,
                    success=False,
                    error="max_iterations_exceeded",
                ))
                continue

            try:
                # 检查依赖
                deps_met = all(dep_id in task_outputs for dep_id in task.dependencies)
                if not deps_met:
                    task.mark_skipped()
                    results.append(TaskResult(
                        task_id=task.id,
                        success=False,
                        error="Dependencies not met",
                    ))
                    continue

                # 如果任务指定了工具，直接调用
                tool_name = task.metadata.get("tool")
                if tool_name:
                    task.mark_running()
                    result = self._execute_tool_by_name(tool_name, task.description)
                    task.mark_completed(result)
                    task_outputs[task.id] = result

                    tool_results.append({
                        "tool": tool_name,
                        "result": result,
                        "task_id": task.id,
                        "iteration": self._iteration_count,
                    })

                    messages.append({
                        "role": "tool",
                        "content": str(result),
                    })

                    if self._config.enable_memory and self._memory:
                        self._memory.add_tool_message(
                            content=str(result),
                            tool_name=tool_name,
                        )

                    results.append(TaskResult(
                        task_id=task.id,
                        success=True,
                        result=result,
                    ))
                else:
                    # 无指定工具，使用 LLM 继续推理
                    response = self._llm.invoke(messages)
                    if hasattr(response, "tool_calls") and response.tool_calls:
                        for tool_call in response.tool_calls:
                            result = self._execute_tool_call(tool_call)
                            task_outputs[task.id] = result
                            tool_results.append({
                                "tool": tool_call.function.name,
                                "result": result,
                                "task_id": task.id,
                                "iteration": self._iteration_count,
                            })
                            messages.append({
                                "role": "tool",
                                "content": str(result),
                            })
                            results.append(TaskResult(
                                task_id=task.id,
                                success=True,
                                result=result,
                            ))
                    else:
                        content = response.content if hasattr(response, "content") else str(response)
                        task_outputs[task.id] = content
                        task.mark_completed(content)
                        results.append(TaskResult(
                            task_id=task.id,
                            success=True,
                            result=content,
                        ))

            except Exception as e:
                logger.error(f"任务 {task.id} 执行失败: {e}")
                task.mark_failed(str(e))
                results.append(TaskResult(
                    task_id=task.id,
                    success=False,
                    error=str(e),
                ))

        return results

    def _execute_tool_by_name(self, tool_name: str, task_description: str) -> str:
        """根据工具名执行工具"""
        from actions.base import ActionResult

        try:
            result = self._action_dispatcher.dispatch_tool_call(
                tool_name=tool_name,
                tool_args={"description": task_description},
            )

            if result.success:
                return result.content or str(result.tool_calls)
            else:
                return f"工具执行错误: {result.error}"

        except Exception as e:
            logger.error(f"工具 {tool_name} 执行失败: {e}")
            return f"工具执行失败: {str(e)}"

    def _synthesize_response(
        self,
        user_input: str,
        messages: list,
        execution_results: list,
        tool_results: list,
    ) -> str:
        """汇总执行结果，生成最终回复"""
        # 构建汇总上下文
        summary_context = f"用户原始问题: {user_input}\n\n"
        summary_context += "任务执行结果汇总:\n"

        for result in execution_results:
            if result.success:
                summary_context += f"- [{result.task_id}] 成功: {result.result}\n"
            else:
                summary_context += f"- [{result.task_id}] 失败: {result.error}\n"

        # 让 LLM 总结最终回复
        synthesis_prompt = f"""{summary_context}

请根据以上任务执行结果，用自然语言为用户生成最终回复。
回复应该：
1. 直接回答用户的问题
2. 清晰地呈现执行结果
3. 如有数据，以易于理解的方式呈现"""

        synthesis_messages = messages + [{"role": "user", "content": synthesis_prompt}]

        try:
            response = self._llm.invoke(synthesis_messages)
            return response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            logger.error(f"汇总回复生成失败: {e}")
            # 回退到简单的结果拼接
            return summary_context

    def stream(self, user_input: str, **kwargs) -> Generator:
        """
        流式处理用户输入（支持工具调用）

        Args:
            user_input: 用户输入
            **kwargs: 额外参数

        Yields:
            流式响应块，可以是：
            - LLM 内容块 (AIMessageChunk)
            - 工具调用事件 (dict with "type": "tool_call")
            - 工具结果事件 (dict with "type": "tool_result")
        """
        from dataclasses import dataclass, field
        from typing import Iterator, Union, Any

        self._iteration_count = 0
        tool_results = []

        # Memory: 记录用户消息
        if self._config.enable_memory and self._memory:
            self._memory.add_user_message(user_input)

        messages = self._get_messages_for_llm()

        # 流式 ReAct 循环
        while self._iteration_count < self._config.max_iterations:
            self._iteration_count += 1

            try:
                content_buffer = ""
                has_tool_calls = False
                tool_calls_data = []

                # 流式调用 LLM
                for chunk in self._llm.stream(messages):
                    # 处理工具调用
                    if hasattr(chunk, "tool_calls") and chunk.tool_calls:
                        has_tool_calls = True
                        for tc in chunk.tool_calls:
                            tool_calls_data.append({
                                "id": getattr(tc, "id", None),
                                "name": tc.function.name,
                                "args": tc.function.arguments,
                            })
                        # yield 工具调用事件
                        yield {
                            "type": "tool_call",
                            "tool_calls": tool_calls_data,
                            "iteration": self._iteration_count,
                        }

                    # 处理内容
                    if hasattr(chunk, "content") and chunk.content:
                        content_buffer += chunk.content
                        yield chunk

                # 检查是否有工具调用
                if has_tool_calls:
                    # 执行工具
                    for tc_data in tool_calls_data:
                        tool_name = tc_data["name"]
                        tool_args = self._parse_tool_args(tc_data["args"])

                        # 执行工具
                        result = self._execute_tool_by_name(tool_name, tool_args)
                        tool_results.append({
                            "tool": tool_name,
                            "result": result,
                            "iteration": self._iteration_count,
                        })

                        # yield 工具结果事件
                        yield {
                            "type": "tool_result",
                            "tool": tool_name,
                            "result": result,
                            "iteration": self._iteration_count,
                        }

                        # 添加到消息
                        messages.append({
                            "role": "tool",
                            "content": str(result),
                            "tool_call_id": tc_data.get("id"),
                        })

                        # Memory: 记录工具结果
                        if self._config.enable_memory and self._memory:
                            self._memory.add_tool_message(
                                content=str(result),
                                tool_name=tool_name,
                            )

                    continue

                # 无工具调用，输出完成
                if content_buffer:
                    if self._config.enable_memory and self._memory:
                        self._memory.add_assistant_message(content_buffer)

                    yield {
                        "type": "done",
                        "content": content_buffer,
                        "iterations": self._iteration_count,
                        "tool_results": tool_results,
                    }
                    return

            except Exception as e:
                logger.error(f"流式处理错误: {e}")
                yield {
                    "type": "error",
                    "error": str(e),
                    "iteration": self._iteration_count,
                }
                return

        # 超过最大迭代次数
        yield {
            "type": "error",
            "error": "max_iterations_exceeded",
            "iterations": self._iteration_count,
            "tool_results": tool_results,
        }

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
                # 提取结构化结果
                self._extract_and_store_result(tool_name, result.content)
                return result.content or str(result.tool_calls)
            else:
                return f"工具执行错误: {result.error}"

        except Exception as e:
            logger.error(f"工具 {tool_name} 执行失败: {e}")
            return f"工具执行失败: {str(e)}"

    def _extract_and_store_result(self, tool_name: str, result: Any) -> None:
        """提取并存储结构化结果"""
        if not self._config.enable_memory or not self._memory:
            return

        try:
            from memories.structured_result import extract_structured_result

            structured = extract_structured_result(tool_name, result)

            # 将结构化结果添加到记忆元数据
            if hasattr(self._memory, "add_observation"):
                self._memory.add_observation(tool_name, structured)
            else:
                # 降级：保存为工具消息
                self._memory.add_tool_message(
                    content=str(result),
                    tool_name=tool_name,
                    metadata=structured.to_dict(),
                )

            if self._config.debug:
                logger.debug(f"提取结构化结果: {tool_name}, fields={list(structured.fields.keys())}")

        except Exception as e:
            logger.debug(f"结构化结果提取失败: {e}")
            # 降级：保存原始结果
            if hasattr(self._memory, "add_tool_message"):
                self._memory.add_tool_message(content=str(result), tool_name=tool_name)

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
