# -*- coding: utf-8 -*-
"""
规划管理器

整合规划器、任务执行器和记忆系统。
"""

from typing import Any, Optional

from core.logger import logger

from .base import BasePlanner, Task, TaskResult, TaskStatus
from .executor import TaskExecutor
from .strategies import LLMPlanner, SimplePlanner


class PlanningManager:
    """
    规划管理器

    协调规划器、任务执行器和上下文管理。
    支持复杂任务的分析、规划和执行。
    """

    def __init__(
        self,
        planner: Optional[BasePlanner] = None,
        executor: Optional[TaskExecutor] = None,
        llm: Optional[Any] = None,
        auto_llm_planning: bool = True,
    ):
        """
        初始化规划管理器

        Args:
            planner: 规划器，None 则根据 llm 参数自动选择
            executor: 任务执行器，None 则使用顺序执行器
            llm: LLM 实例，用于智能规划
            auto_llm_planning: 是否自动使用 LLM 规划器（当 llm 可用时）
        """
        self._llm = llm
        self._auto_llm_planning = auto_llm_planning

        # 选择规划器
        if planner is not None:
            self._planner = planner
        elif llm and auto_llm_planning:
            self._planner = LLMPlanner(llm=llm)
        else:
            self._planner = SimplePlanner()

        self._executor = executor or TaskExecutor(strategy="sequential")
        self._current_plan: list[Task] = []
        self._execution_results: list[TaskResult] = []
        logger.debug(f"初始化规划管理器，使用 {self._planner}")

    def set_llm(self, llm: Any) -> None:
        """设置 LLM 并切换到 LLM 规划器"""
        self._llm = llm
        if self._auto_llm_planning:
            self._planner = LLMPlanner(llm=llm)
            logger.info("已切换到 LLM 规划器")

    @property
    def planner(self) -> BasePlanner:
        """获取规划器"""
        return self._planner

    @planner.setter
    def planner(self, value: BasePlanner) -> None:
        """设置规划器"""
        self._planner = value
        logger.debug(f"更新规划器: {value}")

    @property
    def executor(self) -> TaskExecutor:
        """获取执行器"""
        return self._executor

    @executor.setter
    def executor(self, value: TaskExecutor) -> None:
        """设置执行器"""
        self._executor = value
        logger.debug(f"更新执行器: {value}")

    def analyze(self, user_input: str, context: Optional[dict] = None) -> dict[str, Any]:
        """
        分析用户输入

        Args:
            user_input: 用户输入
            context: 上下文信息

        Returns:
            分析结果
        """
        should_use_tools = self._planner.should_use_tools(user_input, context)
        required_tools = self._planner.get_required_tools(user_input, context)

        result = {
            "needs_tools": should_use_tools,
            "required_tools": required_tools,
            "analysis_complete": True,
        }

        logger.debug(f"分析结果: {result}")
        return result

    def plan(self, user_input: str, context: Optional[dict] = None) -> list[Task]:
        """
        生成任务计划

        Args:
            user_input: 用户输入
            context: 上下文信息

        Returns:
            任务列表
        """
        self._current_plan = self._planner.plan(user_input, context)
        logger.info(f"生成计划，共 {len(self._current_plan)} 个任务")
        return self._current_plan

    def execute(
        self,
        task_func: Optional[Any] = None,
        tasks: Optional[list[Task]] = None,
    ) -> list[TaskResult]:
        """
        执行任务

        Args:
            task_func: 任务执行函数，接收 Task，返回结果
            tasks: 要执行的任务列表，None 则使用当前计划

        Returns:
            任务结果列表
        """
        tasks_to_execute = tasks or self._current_plan

        if not tasks_to_execute:
            logger.warning("没有可执行的任务")
            return []

        if task_func is None:
            logger.warning("没有提供任务执行函数，仅更新任务状态")
            for task in tasks_to_execute:
                task.mark_skipped()
            return []

        self._execution_results = self._executor.execute(
            tasks_to_execute,
            task_func,
        )

        logger.info(f"执行完成，{len(self._execution_results)} 个结果")
        return self._execution_results

    def execute_and_plan(
        self,
        user_input: str,
        task_func: Optional[Any] = None,
        context: Optional[dict] = None,
    ) -> dict[str, Any]:
        """
        分析、规划和执行的完整流程

        Args:
            user_input: 用户输入
            task_func: 任务执行函数
            context: 上下文信息

        Returns:
            包含分析、计划和执行结果的字典
        """
        analysis = self.analyze(user_input, context)

        if not analysis["needs_tools"]:
            return {
                "analysis": analysis,
                "plan": [],
                "results": [],
                "requires_direct_response": True,
            }

        tasks = self.plan(user_input, context)

        results = []
        if task_func and tasks:
            results = self.execute(task_func, tasks)

        return {
            "analysis": analysis,
            "plan": [task.to_dict() for task in tasks],
            "results": [r.to_dict() for r in results],
            "requires_direct_response": False,
        }

    def get_current_plan(self) -> list[Task]:
        """获取当前计划"""
        return self._current_plan

    def get_execution_results(self) -> list[TaskResult]:
        """获取执行结果"""
        return self._execution_results

    def clear(self) -> None:
        """清空计划和结果"""
        self._current_plan = []
        self._execution_results = []
        logger.debug("清空规划和执行结果")

    def get_task_by_id(self, task_id: str) -> Optional[Task]:
        """根据 ID 获取任务"""
        for task in self._current_plan:
            if task.id == task_id:
                return task
        return None

    def get_result_by_task_id(self, task_id: str) -> Optional[TaskResult]:
        """根据任务 ID 获取结果"""
        for result in self._execution_results:
            if result.task_id == task_id:
                return result
        return None

    @classmethod
    def create_with_llm_planner(cls, llm: Any) -> "PlanningManager":
        """
        创建使用 LLM 规划器的管理器

        Args:
            llm: LLM 实例

        Returns:
            配置好的规划管理器
        """
        planner = LLMPlanner(llm=llm)
        return cls(planner=planner)
