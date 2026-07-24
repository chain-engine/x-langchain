# -*- coding: utf-8 -*-
"""
规划基类定义

定义任务规划系统的抽象接口和数据结构。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class TaskStatus(str, Enum):
    """任务状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Task:
    """
    任务定义

    表示一个可执行的原子任务单元。
    """
    id: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    dependencies: list[str] = field(default_factory=list)
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "id": self.id,
            "description": self.description,
            "status": self.status.value,
            "dependencies": self.dependencies,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "metadata": self.metadata,
        }

    def mark_running(self) -> None:
        """标记为运行中"""
        self.status = TaskStatus.RUNNING
        self.started_at = datetime.now()

    def mark_completed(self, result: Any = None) -> None:
        """标记为完成"""
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.now()
        self.result = result

    def mark_failed(self, error: str) -> None:
        """标记为失败"""
        self.status = TaskStatus.FAILED
        self.completed_at = datetime.now()
        self.error = error

    def mark_skipped(self) -> None:
        """标记为跳过"""
        self.status = TaskStatus.SKIPPED
        self.completed_at = datetime.now()


@dataclass
class TaskResult:
    """
    任务执行结果

    封装任务执行后的返回结果。
    """
    task_id: str
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "task_id": self.task_id,
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "execution_time": self.execution_time,
            "metadata": self.metadata,
        }


class BasePlanner(ABC):
    """
    规划器基类

    定义任务规划的抽象接口。
    所有具体规划器实现都需要继承此类。
    """

    @abstractmethod
    def plan(self, user_input: str, context: Optional[dict] = None) -> list[Task]:
        """
        根据用户输入生成任务计划

        Args:
            user_input: 用户输入
            context: 上下文信息

        Returns:
            任务列表
        """
        pass

    @abstractmethod
    def should_use_tools(self, user_input: str, context: Optional[dict] = None) -> bool:
        """
        判断是否需要使用工具

        Args:
            user_input: 用户输入
            context: 上下文信息

        Returns:
            是否需要工具
        """
        pass

    @abstractmethod
    def get_required_tools(self, user_input: str, context: Optional[dict] = None) -> list[str]:
        """
        获取所需工具列表

        Args:
            user_input: 用户输入
            context: 上下文信息

        Returns:
            工具名称列表
        """
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"
