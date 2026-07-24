# -*- coding: utf-8 -*-
"""
LangChain Agent 封装

基于 LangGraph 的 create_react_agent API，提供：
- ReAct Agent
- 工具调用 Agent
- 流式输出
- 记忆管理
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
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
    """
    LangChain Agent 封装

    基于 LangGraph 的 create_react_agent API。
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        llm: Optional[BaseChatModel] = None,
        tools: Optional[Sequence[Any]] = None,
        system_message: Optional[str] = None,
        state_schema: Optional[type] = None,
    ):
        """
        初始化 LangChain Agent

        Args:
            config: 配置对象
            llm: 语言模型
            tools: 工具列表
            system_message: 系统消息
            state_schema: 状态模式
        """
        self._config = config
        self._system_message = system_message
        self._state_schema = state_schema
        self._tool_results: List[dict] = []
        self._iteration_count = 0

        # 初始化 LLM
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

        # 初始化工具
        if tools is None:
            from tools import get_all_tools
            self._tools: Sequence[Any] = get_all_tools()
        else:
            self._tools = tools

        # 创建 Agent
        self._agent: Optional[Runnable] = None
        self._create_agent()

        logger.info(f"初始化 LangChain Agent (LangGraph), tools={len(self._tools)}")

    def _create_agent(self) -> None:
        """创建 Agent (LangGraph 标准实现)"""
        system_prompt = self._system_message or """你是一个智能助手，可以帮助用户完成各种任务。
当需要外部信息时，使用工具来获取。不要编造信息。"""

        # 使用 LangGraph 的 create_react_agent
        self._agent = create_react_agent(
            model=self._llm,
            tools=self._tools,
            prompt=system_prompt,
            state_schema=self._state_schema,
            checkpointer=None,
        )

    @property
    def llm(self) -> BaseChatModel:
        """获取 LLM"""
        return self._llm

    @property
    def tools(self) -> Sequence[Any]:
        """获取工具列表"""
        return self._tools

    @property
    def agent(self) -> Runnable:
        """获取 Agent"""
        if self._agent is None:
            self._create_agent()
        return self._agent

    def invoke(self, user_input: str, **kwargs) -> AgentResponse:
        """
        处理用户输入 (LangGraph 标准调用)

        Args:
            user_input: 用户输入
            **kwargs: 额外参数

        Returns:
            Agent 响应
        """
        self._tool_results = []
        self._iteration_count = 0

        try:
            # 构建输入 (LangGraph 格式)
            inputs = {"messages": [("user", user_input)]}

            # 调用 Agent (LangGraph 标准方式)
            config = kwargs.get("config", {})
            result = self.agent.invoke(inputs, config=config)

            # 解析结果 (LangGraph 输出格式)
            output = self._extract_output(result)
            self._iteration_count = self._count_iterations(result)

            return AgentResponse(
                content=output,
                success=True,
                tool_results=self._tool_results,
                iterations=self._iteration_count,
                metadata={"agent_type": "langgraph", "provider": "create_react_agent"},
            )

        except Exception as e:
            logger.error(f"Agent 执行错误: {e}")
            return AgentResponse(
                content=f"处理请求时出错: {str(e)}",
                success=False,
                tool_results=self._tool_results,
                metadata={"error": str(e)},
            )

    def stream(self, user_input: str, **kwargs) -> Generator:
        """
        流式处理用户输入 (LangGraph 标准方式)

        Args:
            user_input: 用户输入
            **kwargs: 额外参数

        Yields:
            流式响应块
        """
        self._tool_results = []
        self._iteration_count = 0

        try:
            inputs = {"messages": [("user", user_input)]}
            config = kwargs.get("config", {})

            for event in self.agent.stream(inputs, config=config):
                # LangGraph 事件格式处理
                if isinstance(event, dict):
                    for node_name, node_output in event.items():
                        if isinstance(node_output, dict) and "messages" in node_output:
                            messages = node_output["messages"]
                            for msg in messages:
                                if hasattr(msg, "content") and msg.content:
                                    yield {"type": "message", "content": msg.content, "node": node_name}
                                # 工具调用
                                if hasattr(msg, "name") and msg.name:
                                    yield {"type": "tool", "name": msg.name}
                                    self._tool_results.append({"tool": msg.name})
                elif hasattr(event, "content"):
                    yield {"type": "message", "content": event.content}

        except Exception as e:
            logger.error(f"Agent 流式处理错误: {e}")
            yield {"type": "error", "error": str(e)}

    def _extract_output(self, result: Any) -> str:
        """从 LangGraph 结果中提取输出"""
        if isinstance(result, dict):
            # LangGraph 返回格式: {"messages": [...], ...}
            if "messages" in result:
                messages = result["messages"]
                if messages:
                    last_msg = messages[-1]
                    if hasattr(last_msg, "content"):
                        return last_msg.content
                    elif isinstance(last_msg, tuple):
                        # (role, content) 格式
                        return last_msg[1] if len(last_msg) > 1 else str(last_msg)
            return str(result)
        return str(result)

    def _count_iterations(self, result: Any) -> int:
        """计算迭代次数 (LangGraph 方式)"""
        if isinstance(result, dict) and "messages" in result:
            messages = result["messages"]
            # 统计 AI 消息数量作为迭代次数
            ai_count = 0
            for m in messages:
                if isinstance(m, tuple):
                    if len(m) > 0 and m[0] == "ai":
                        ai_count += 1
                elif hasattr(m, "type") and m.type == "ai":
                    ai_count += 1
            return ai_count
        return self._iteration_count

    def add_tools(self, tools: Sequence[Any]) -> None:
        """动态添加工具"""
        self._tools = list(self._tools) + list(tools)
        self._create_agent()
        logger.info(f"添加 {len(tools)} 个工具，重新创建 Agent")

    def __repr__(self) -> str:
        return f"<LCAgent: tools={len(self._tools)}>"


__all__ = ["LCAgent", "AgentResponse"]
