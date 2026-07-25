# -*- coding: utf-8 -*-
"""
LangChain Agent 封装

基于 LangGraph 的 create_react_agent API。
"""

from dataclasses import dataclass, field
from typing import Any, Generator, List, Optional, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langgraph.prebuilt import create_react_agent

from core.logger import logger
from llms import create_chat_model


@dataclass
class AgentResponse:
    """Agent 响应"""
    content: str
    success: bool
    tool_results: List[dict] = field(default_factory=list)
    iterations: int = 0
    metadata: dict = field(default_factory=dict)


class LCAgent:
    """基于 LangGraph create_react_agent 的 Agent 封装"""

    def __init__(
        self,
        config: Optional[Any] = None,
        llm: Optional[BaseChatModel] = None,
        tools: Optional[Sequence[Any]] = None,
        system_message: Optional[str] = None,
        state_schema: Optional[type] = None,
    ):
        if llm is None:
            from core.config import AgentConfig
            agent_config = config or AgentConfig()
            self._llm = create_chat_model(
                provider_name=agent_config.model_provider,
                model_name=agent_config.model_name,
                temperature=agent_config.temperature,
                max_tokens=agent_config.max_tokens,
            )
        else:
            self._llm = llm

        self._tools: Sequence[Any] = tools if tools is not None else self._get_default_tools()
        self._system_message = system_message or "你是一个智能助手，可以帮助用户完成各种任务。当需要外部信息时，使用工具来获取。不要编造信息。"
        self._state_schema = state_schema

        self._agent: Runnable = create_react_agent(
            model=self._llm,
            tools=self._tools,
            prompt=self._system_message,
            state_schema=self._state_schema,
        )

        logger.info(f"初始化 LCAgent, tools={len(self._tools)}")

    @staticmethod
    def _get_default_tools() -> Sequence[Any]:
        from tools import get_all_tools
        return get_all_tools()

    @property
    def llm(self) -> BaseChatModel:
        return self._llm

    @property
    def tools(self) -> Sequence[Any]:
        return self._tools

    def invoke(self, user_input: str, **kwargs) -> AgentResponse:
        """处理用户输入"""
        tool_results: List[dict] = []
        iterations = 0

        try:
            result = self._agent.invoke(
                {"messages": [("user", user_input)]},
                config=kwargs.get("config", {}),
            )

            content = self._extract_content(result)
            iterations = self._count_ai_messages(result)

            return AgentResponse(
                content=content,
                success=True,
                tool_results=tool_results,
                iterations=iterations,
                metadata={"agent_type": "langgraph"},
            )

        except Exception as e:
            logger.error(f"Agent 执行错误: {e}")
            return AgentResponse(
                content=f"处理请求时出错: {str(e)}",
                success=False,
                metadata={"error": str(e)},
            )

    def stream(self, user_input: str, **kwargs) -> Generator:
        """流式处理用户输入"""
        try:
            for event in self._agent.stream(
                {"messages": [("user", user_input)]},
                config=kwargs.get("config", {}),
            ):
                if isinstance(event, dict):
                    for node_output in event.values():
                        if isinstance(node_output, dict) and "messages" in node_output:
                            for msg in node_output["messages"]:
                                if hasattr(msg, "content") and msg.content:
                                    yield {"type": "message", "content": msg.content}
                                if hasattr(msg, "name") and msg.name:
                                    yield {"type": "tool", "name": msg.name}
                elif hasattr(event, "content"):
                    yield {"type": "message", "content": event.content}
        except Exception as e:
            logger.error(f"Agent 流式处理错误: {e}")
            yield {"type": "error", "error": str(e)}

    def _extract_content(self, result: Any) -> str:
        if isinstance(result, dict) and "messages" in result:
            messages = result["messages"]
            if messages:
                last = messages[-1]
                if hasattr(last, "content"):
                    return last.content
        return str(result)

    def _count_ai_messages(self, result: Any) -> int:
        if isinstance(result, dict) and "messages" in result:
            return sum(
                1 for m in result["messages"]
                if (isinstance(m, tuple) and m[0] == "ai")
                or (hasattr(m, "type") and m.type == "ai")
            )
        return 0

    def add_tools(self, tools: Sequence[Any]) -> None:
        """动态添加工具"""
        self._tools = list(self._tools) + list(tools)
        self._agent = create_react_agent(
            model=self._llm,
            tools=self._tools,
            prompt=self._system_message,
            state_schema=self._state_schema,
        )
        logger.info(f"添加 {len(tools)} 个工具，重新创建 Agent")

    def __repr__(self) -> str:
        return f"<LCAgent: tools={len(self._tools)}>"


__all__ = ["LCAgent", "AgentResponse"]
