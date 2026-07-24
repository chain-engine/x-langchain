# -*- coding: utf-8 -*-
"""Agent 集成测试。"""

import sys
from pathlib import Path
from unittest import TestCase, mock

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


class TestAgentConfig(TestCase):
    """AgentConfig 测试"""

    def test_agent_config_defaults(self) -> None:
        """测试 AgentConfig 默认值"""
        from core.config import AgentConfig
        config = AgentConfig()
        assert config.model_provider == "deepseek"
        assert config.enable_memory is True
        assert config.enable_tools is True
        assert config.enable_planning is False

    def test_agent_config_custom(self) -> None:
        """测试 AgentConfig 自定义值"""
        from core.config import AgentConfig
        from constants import AgentMode

        config = AgentConfig(
            model_provider="doubao",
            max_iterations=5,
            debug=True,
            mode=AgentMode.REACT,
        )
        assert config.model_provider == "doubao"
        assert config.max_iterations == 5
        assert config.debug is True
        assert config.mode == AgentMode.REACT


class TestAgentResponse(TestCase):
    """AgentResponse 测试"""

    def test_agent_response_creation(self) -> None:
        """测试 AgentResponse 创建"""
        from agent import AgentResponse

        response = AgentResponse(
            content="Hello, World!",
            success=True,
            tool_results=[{"tool": "weather", "result": "sunny"}],
            iterations=2,
        )

        assert response.content == "Hello, World!"
        assert response.success is True
        assert len(response.tool_results) == 1
        assert response.iterations == 2

    def test_agent_response_defaults(self) -> None:
        """测试 AgentResponse 默认值"""
        from agent import AgentResponse

        response = AgentResponse(content="Test", success=True)
        assert response.success is True
        assert response.tool_results == []
        assert response.iterations == 0
        assert response.metadata == {}


class TestAgentMemory(TestCase):
    """Agent Memory 测试"""

    def test_conversation_memory_add_messages(self) -> None:
        """测试对话记忆添加消息"""
        from memories import ConversationHistoryMemory, MemoryMessage, MessageRole

        memory = ConversationHistoryMemory(max_messages=10)
        memory.add_user_message("Hello")
        memory.add_assistant_message("Hi there!")

        messages = memory.get_messages()
        assert len(messages) == 2
        assert messages[0].role == MessageRole.USER
        assert messages[0].content == "Hello"
        assert messages[1].role == MessageRole.ASSISTANT
        assert messages[1].content == "Hi there!"

    def test_conversation_memory_max_messages(self) -> None:
        """测试对话记忆消息数量限制"""
        from memories import ConversationHistoryMemory

        memory = ConversationHistoryMemory(max_messages=3)
        for i in range(5):
            memory.add_user_message(f"Message {i}")

        messages = memory.get_messages()
        assert len(messages) == 3

    def test_conversation_memory_clear(self) -> None:
        """测试对话记忆清空"""
        from memories import ConversationHistoryMemory

        memory = ConversationHistoryMemory()
        memory.add_user_message("Hello")
        memory.add_assistant_message("Hi")
        memory.clear()

        messages = memory.get_messages()
        assert len(messages) == 0


class TestAgentPlanning(TestCase):
    """Agent Planning 测试"""

    def test_simple_planner_keywords(self) -> None:
        """测试简单规划器关键词匹配"""
        from planning import SimplePlanner

        planner = SimplePlanner()
        tasks = planner.plan("查询北京天气")

        assert len(tasks) > 0
        assert any("天气" in task.description for task in tasks)

    def test_simple_planner_no_match(self) -> None:
        """测试简单规划器无匹配"""
        from planning import SimplePlanner

        planner = SimplePlanner()
        tasks = planner.plan("你好")

        assert len(tasks) == 0

    def test_task_status_transitions(self) -> None:
        """测试任务状态转换"""
        from planning import Task, TaskStatus

        task = Task(id="test_task", description="Test task")
        assert task.status == TaskStatus.PENDING

        task.mark_running()
        assert task.status == TaskStatus.RUNNING
        assert task.started_at is not None

        task.mark_completed(result="success")
        assert task.status == TaskStatus.COMPLETED
        assert task.result == "success"
        assert task.completed_at is not None


class TestAgentActions(TestCase):
    """Agent Actions 测试"""

    def test_action_result_creation(self) -> None:
        """测试行动结果创建"""
        from actions import ActionResult, ActionType

        result = ActionResult(
            action_type=ActionType.DIRECT_RESPONSE,
            success=True,
            content="Hello!",
        )

        assert result.action_type == ActionType.DIRECT_RESPONSE
        assert result.success is True
        assert result.content == "Hello!"

    def test_direct_response_action(self) -> None:
        """测试直接回复行动"""
        from actions import DirectResponseAction

        action = DirectResponseAction("Hello, World!")
        context = {"content": "", "tool_calls": []}

        assert action.should_execute(context) is True
        result = action.execute(context)

        assert result.success is True
        assert result.content == "Hello, World!"

    def test_tool_call_action_skip(self) -> None:
        """测试工具调用行动跳过"""
        from actions import ToolCallAction

        action = ToolCallAction()
        context = {"content": "", "tool_calls": []}

        assert action.should_execute(context) is False
