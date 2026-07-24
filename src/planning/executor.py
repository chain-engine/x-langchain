# -*- coding: utf-8 -*-
"""
任务执行器

实现任务执行逻辑，包括顺序执行和并行执行。
"""

import asyncio
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Optional

from core.logger import logger

from .base import BasePlanner, Task, TaskResult, TaskStatus


class BaseTaskExecutor(ABC):
    """
    任务执行器基类

    定义任务执行的抽象接口。
    """

    def __init__(self, max_retries: int = 0):
        """
        初始化任务执行器

        Args:
            max_retries: 最大重试次数
        """
        self._max_retries = max_retries

    @abstractmethod
    def execute(self, tasks: list[Task], executor_func: Callable[[Task], Any]) -> list[TaskResult]:
        """
        执行任务列表

        Args:
            tasks: 任务列表
            executor_func: 任务执行函数

        Returns:
            任务结果列表
        """
        pass

    def _run_task_with_retry(self, task: Task, executor_func: Callable[[Task], Any]) -> TaskResult:
        """带重试的任务执行"""
        last_error = None

        for attempt in range(self._max_retries + 1):
            try:
                start_time = time.time()
                task.mark_running()

                if asyncio.iscoroutinefunction(executor_func):
                    result = asyncio.run(executor_func(task))
                else:
                    result = executor_func(task)

                execution_time = time.time() - start_time
                task.mark_completed(result)

                return TaskResult(
                    task_id=task.id,
                    success=True,
                    result=result,
                    execution_time=execution_time,
                )

            except Exception as e:
                last_error = str(e)
                logger.warning(f"任务 {task.id} 执行失败 (尝试 {attempt + 1}/{self._max_retries + 1}): {e}")

                if attempt < self._max_retries:
                    task.status = TaskStatus.PENDING
                    task.started_at = None
                else:
                    task.mark_failed(last_error)

                    return TaskResult(
                        task_id=task.id,
                        success=False,
                        error=last_error,
                        execution_time=time.time() - start_time,
                    )

        return TaskResult(
            task_id=task.id,
            success=False,
            error=last_error or "Unknown error",
        )


class SequentialExecutor(BaseTaskExecutor):
    """
    顺序执行器

    按顺序执行任务，确保依赖关系正确。
    """

    def execute(self, tasks: list[Task], executor_func: Callable[[Task], Any]) -> list[TaskResult]:
        """
        顺序执行任务

        Args:
            tasks: 任务列表
            executor_func: 任务执行函数

        Returns:
            任务结果列表
        """
        results = []
        task_results: dict[str, Any] = {}

        for task in tasks:
            if self._check_dependencies(task, task_results):
                result = self._run_task_with_retry(task, executor_func)
                results.append(result)

                if result.success and result.result is not None:
                    task_results[task.id] = result.result

                if not result.success:
                    self._skip_dependent_tasks(tasks, task.id, results)
            else:
                task.mark_skipped()
                results.append(TaskResult(
                    task_id=task.id,
                    success=False,
                    error="Dependencies not met",
                ))

        return results

    def _check_dependencies(self, task: Task, completed_results: dict[str, Any]) -> bool:
        """检查依赖是否满足"""
        return all(dep_id in completed_results for dep_id in task.dependencies)

    def _skip_dependent_tasks(
        self,
        tasks: list[Task],
        failed_task_id: str,
        results: list[TaskResult],
    ) -> None:
        """跳过依赖于失败任务的其他任务"""
        failed_task = next((t for t in tasks if t.id == failed_task_id), None)
        if not failed_task:
            return

        for task in tasks:
            if task.status == TaskStatus.PENDING and failed_task_id in task.dependencies:
                task.mark_skipped()
                results.append(TaskResult(
                    task_id=task.id,
                    success=False,
                    error=f"Skipped due to failed dependency: {failed_task_id}",
                ))


class ParallelExecutor(BaseTaskExecutor):
    """
    并行执行器

    并行执行没有依赖关系的任务。
    """

    def __init__(self, max_workers: int = 4, max_retries: int = 0):
        """
        初始化并行执行器

        Args:
            max_workers: 最大并行工作数
            max_retries: 最大重试次数
        """
        super().__init__(max_retries)
        self._max_workers = max_workers

    def execute(self, tasks: list[Task], executor_func: Callable[[Task], Any]) -> list[TaskResult]:
        """
        并行执行任务

        Args:
            tasks: 任务列表
            executor_func: 任务执行函数

        Returns:
            任务结果列表
        """
        results: list[TaskResult] = []
        task_results: dict[str, Any] = {}
        pending_tasks = {task.id: task for task in tasks}

        while pending_tasks:
            ready_tasks = [
                task for task_id, task in pending_tasks.items()
                if self._check_dependencies(task, task_results)
            ]

            if not ready_tasks:
                remaining = list(pending_tasks.values())
                for task in remaining:
                    task.mark_skipped()
                    results.append(TaskResult(
                        task_id=task.id,
                        success=False,
                        error="Unmet dependencies (circular or missing)",
                    ))
                break

            with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
                futures = {
                    executor.submit(self._run_task_with_retry, task, executor_func): task
                    for task in ready_tasks
                }

                for future in as_completed(futures):
                    task = futures[future]
                    try:
                        result = future.result()
                        results.append(result)

                        if result.success:
                            task_results[task.id] = result.result

                        del pending_tasks[task.id]
                    except Exception as e:
                        logger.error(f"任务 {task.id} 执行异常: {e}")
                        task.mark_failed(str(e))
                        results.append(TaskResult(
                            task_id=task.id,
                            success=False,
                            error=str(e),
                        ))
                        del pending_tasks[task.id]

        return sorted(results, key=lambda r: r.task_id)

    def _check_dependencies(self, task: Task, completed_results: dict[str, Any]) -> bool:
        """检查依赖是否满足"""
        return all(dep_id in completed_results for dep_id in task.dependencies)


class TaskExecutor:
    """
    任务执行器门面

    提供统一的任务执行接口，支持多种执行策略。
    """

    def __init__(
        self,
        strategy: str = "sequential",
        max_workers: int = 4,
        max_retries: int = 0,
    ):
        """
        初始化任务执行器

        Args:
            strategy: 执行策略，"sequential" 或 "parallel"
            max_workers: 并行执行的最大工作数
            max_retries: 最大重试次数
        """
        self._strategy = strategy
        self._max_workers = max_workers
        self._max_retries = max_retries

        if strategy == "parallel":
            self._executor = ParallelExecutor(
                max_workers=max_workers,
                max_retries=max_retries,
            )
        else:
            self._executor = SequentialExecutor(max_retries=max_retries)

        logger.debug(f"初始化任务执行器: strategy={strategy}, max_workers={max_workers}")

    def execute(
        self,
        tasks: list[Task],
        executor_func: Callable[[Task], Any],
    ) -> list[TaskResult]:
        """
        执行任务列表

        Args:
            tasks: 任务列表
            executor_func: 任务执行函数

        Returns:
            任务结果列表
        """
        return self._executor.execute(tasks, executor_func)

    @property
    def strategy(self) -> str:
        """获取执行策略"""
        return self._strategy
