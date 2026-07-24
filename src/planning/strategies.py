# -*- coding: utf-8 -*-
"""
规划策略

实现不同的规划策略：
- LLM 驱动的智能规划
- 简单的规则匹配规划
"""

from typing import Any, Optional

from core.logger import logger

from .base import BasePlanner, Task


class SimplePlanner(BasePlanner):
    """
    简单规划器

    基于关键词和规则的简单规划策略。
    适用于简单场景，不依赖 LLM。
    """

    TOOL_KEYWORDS = {
        "天气": ["weather", "天气", "温度", "气温", "下雨"],
        "搜索": ["search", "搜索", "查找", "查询", "找"],
        "数据库": ["database", "数据库", "DB", "表", "SQL", "查询"],
        "日历": ["calendar", "日历", "日程", "会议", "schedule"],
        "汇率": ["exchange", "汇率", "货币", "currency"],
    }

    def __init__(self, custom_keywords: Optional[dict[str, list[str]]] = None):
        """
        初始化简单规划器

        Args:
            custom_keywords: 自定义关键词映射
        """
        if custom_keywords:
            self._tool_keywords = {**self.TOOL_KEYWORDS, **custom_keywords}
        else:
            self._tool_keywords = self.TOOL_KEYWORDS.copy()

    def plan(self, user_input: str, context: Optional[dict] = None) -> list[Task]:
        """
        根据用户输入生成任务计划

        Args:
            user_input: 用户输入
            context: 上下文信息

        Returns:
            任务列表
        """
        tasks = []
        input_lower = user_input.lower()

        for tool_name, keywords in self._tool_keywords.items():
            for keyword in keywords:
                if keyword.lower() in input_lower:
                    task = Task(
                        id=f"task_{tool_name.lower()}",
                        description=f"执行 {tool_name} 相关操作",
                        metadata={"tool": tool_name, "trigger_keyword": keyword},
                    )
                    tasks.append(task)
                    break

        if not tasks:
            logger.debug(f"简单规划器未匹配到任何工具: {user_input}")

        return tasks

    def should_use_tools(self, user_input: str, context: Optional[dict] = None) -> bool:
        """判断是否需要使用工具"""
        return len(self.plan(user_input, context)) > 0

    def get_required_tools(self, user_input: str, context: Optional[dict] = None) -> list[str]:
        """获取所需工具列表"""
        tasks = self.plan(user_input, context)
        return [task.metadata.get("tool", "") for task in tasks if task.metadata.get("tool")]


class LLMPlanner(BasePlanner):
    """
    LLM 驱动的智能规划器

    使用 LLM 来分析和规划复杂任务。
    """

    def __init__(
        self,
        llm: Any,
        system_prompt: Optional[str] = None,
    ):
        """
        初始化 LLM 规划器

        Args:
            llm: LLM 实例
            system_prompt: 系统提示词
        """
        self._llm = llm
        self._system_prompt = system_prompt or self._get_default_system_prompt()

    def _get_default_system_prompt(self) -> str:
        """获取默认系统提示"""
        return """你是一个任务规划专家。

根据用户输入，分析需要执行的任务，并按顺序列出。

规则：
1. 如果需要调用工具，使用 "TOOL_CALL:工具名" 格式
2. 如果是简单问答，直接回答
3. 每个任务用一行描述
4. 考虑任务之间的依赖关系

示例输入：查询北京天气并告诉我穿什么
示例输出：
1. TOOL_CALL:weather - 获取北京天气
2. 根据天气给出穿衣建议"""

    def plan(self, user_input: str, context: Optional[dict] = None) -> list[Task]:
        """
        使用 LLM 生成任务计划

        Args:
            user_input: 用户输入
            context: 上下文信息

        Returns:
            任务列表
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content=self._system_prompt),
            HumanMessage(content=f"用户输入: {user_input}"),
        ]

        try:
            response = self._llm.invoke(messages)
            content = response.content if hasattr(response, "content") else str(response)

            tasks = self._parse_llm_response(content, user_input)
            logger.info(f"LLM 规划器生成了 {len(tasks)} 个任务")
            return tasks

        except Exception as e:
            logger.error(f"LLM 规划失败: {e}")
            return []

    def _parse_llm_response(self, content: str, original_input: str) -> list[Task]:
        """解析 LLM 返回的计划"""
        tasks = []
        lines = content.strip().split("\n")

        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            if "TOOL_CALL:" in line.upper():
                parts = line.split(":", 1)
                if len(parts) == 2:
                    tool_name = parts[1].strip().split()[0]
                    tasks.append(Task(
                        id=f"task_{i}_{tool_name.lower()}",
                        description=f"执行 {tool_name} 工具",
                        metadata={"tool": tool_name},
                    ))
            elif line[0].isdigit() and "." in line[:3]:
                desc = line.split(".", 1)[-1].strip()
                if desc:
                    tasks.append(Task(
                        id=f"task_{i}",
                        description=desc,
                    ))

        return tasks

    def should_use_tools(self, user_input: str, context: Optional[dict] = None) -> bool:
        """判断是否需要使用工具"""
        tasks = self.plan(user_input, context)
        return len(tasks) > 0

    def get_required_tools(self, user_input: str, context: Optional[dict] = None) -> list[str]:
        """获取所需工具列表"""
        tasks = self.plan(user_input, context)
        return [
            task.metadata.get("tool", "")
            for task in tasks
            if task.metadata.get("tool")
        ]
