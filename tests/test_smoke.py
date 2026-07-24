# -*- coding: utf-8 -*-
"""Smoke tests for importable runtime entry points."""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


def test_agent_imports() -> None:
    """Test that Agent module can be imported."""
    from agent import Agent, AgentConfig, AgentResponse
    assert Agent is not None
    assert AgentConfig is not None
    assert AgentResponse is not None


def test_core_config_imports() -> None:
    """Test that core config can be imported."""
    from core.config import settings, AgentConfig
    assert settings is not None
    assert AgentConfig is not None


def test_memories_imports() -> None:
    """Test that memories module can be imported."""
    from memories import BaseMemory, ConversationHistoryMemory, MemoryManager
    assert BaseMemory is not None
    assert ConversationHistoryMemory is not None
    assert MemoryManager is not None


def test_planning_imports() -> None:
    """Test that planning module can be imported."""
    from planning import (
        BasePlanner,
        PlanningManager,
        Task,
        TaskExecutor,
        LLMPlanner,
        SimplePlanner,
    )
    assert BasePlanner is not None
    assert PlanningManager is not None
    assert Task is not None
    assert TaskExecutor is not None
    assert LLMPlanner is not None
    assert SimplePlanner is not None


def test_actions_imports() -> None:
    """Test that actions module can be imported."""
    from actions import (
        BaseAction,
        ActionDispatcher,
        ActionResult,
        ToolCallAction,
        DirectResponseAction,
    )
    assert BaseAction is not None
    assert ActionDispatcher is not None
    assert ActionResult is not None
    assert ToolCallAction is not None
    assert DirectResponseAction is not None


def test_llms_imports() -> None:
    """Test that llms module can be imported."""
    from llms import (
        create_chat_model,
        get_llm_provider,
        list_providers,
        DeepSeekProvider,
        DoubaoProvider,
        AliyunProvider,
        MockProvider,
    )
    assert create_chat_model is not None
    assert get_llm_provider is not None
    assert list_providers is not None
    assert "deepseek" in list_providers()
    assert "mock" in list_providers()


def test_tools_imports() -> None:
    """Test that tools module can be imported."""
    from tools import ToolRegistry, get_all_tools, discover_function_calling_tools
    assert ToolRegistry is not None
    assert get_all_tools is not None
    assert discover_function_calling_tools is not None


def test_core_imports() -> None:
    """Test that core module can be imported."""
    from core import settings, logger, container, XLangChainError
    assert settings is not None
    assert logger is not None
    assert container is not None
    assert XLangChainError is not None


def test_constants_imports() -> None:
    """Test that constants module can be imported."""
    from constants import AgentMode, StreamMode
    assert AgentMode is not None
    assert StreamMode is not None
    assert AgentMode.REACT is not None
