# -*- coding: utf-8 -*-
"""
LangChain Agent 封装

基于 LangGraph 的 create_react_agent API，提供单 Agent 智能体。
"""

from dataclasses import dataclass, field
from typing import Any, Generator, List, Optional, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langgraph.prebuilt import create_react_agent

from agent.chat_history_service import ChatHistoryService, create_chat_history_service
from core.logger import logger
from core.middleware import DEFAULT_MIDDLEWARE_CHAIN, MiddlewareChain
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
    """基于 LangGraph create_react_agent 的单 Agent 封装"""

    def __init__(
        self,
        config: Optional[Any] = None,
        llm: Optional[BaseChatModel] = None,
        tools: Optional[Sequence[Any]] = None,
        system_message: Optional[str] = None,
        state_schema: Optional[type] = None,
        middleware_chain: Optional[MiddlewareChain] = None,
        session_id: Optional[str] = None,
        chat_history_service: Optional[ChatHistoryService] = None,
        auto_persist: bool = True,
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
        self._middleware: MiddlewareChain = middleware_chain or DEFAULT_MIDDLEWARE_CHAIN
        self._session_id: Optional[str] = session_id
        self._chat_history_service: Optional[ChatHistoryService] = chat_history_service
        self._auto_persist: bool = auto_persist

        self._agent: Runnable = create_react_agent(
            model=self._llm,
            tools=self._tools,
            prompt=self._system_message,
            state_schema=self._state_schema,
        )

        logger.info(
            f"初始化 LCAgent, tools={len(self._tools)}, "
            f"session_id={self._session_id}, auto_persist={self._auto_persist}, "
            f"middleware={self._middleware.middlewares}"
        )

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

    @property
    def session_id(self) -> Optional[str]:
        return self._session_id

    def _get_history_service(self) -> Optional[ChatHistoryService]:
        """获取历史服务实例（延迟创建）。"""
        if self._chat_history_service is not None:
            return self._chat_history_service
        if self._session_id is None:
            return None
        try:
            self._chat_history_service = create_chat_history_service()
            return self._chat_history_service
        except Exception as e:
            logger.warning(f"无法创建 ChatHistoryService: {e}")
            return None

    def _load_history(self, session_id: str) -> list:
        """加载历史消息并转换为 LangChain 格式。"""
        svc = self._get_history_service()
        if svc is None:
            return []
        messages = svc.get_messages(session_id)
        from langchain_core.messages import AIMessage, HumanMessage
        result = []
        for msg in messages:
            if msg.role == "user":
                result.append(("user", msg.content))
            else:
                result.append(("ai", msg.content))
        logger.debug(f"加载 {len(result)} 条历史消息到会话 {session_id}")
        return result

    def _save_message(self, session_id: str, role: str, content: str) -> None:
        """保存单条消息到 MySQL。"""
        svc = self._get_history_service()
        if svc is None:
            return
        try:
            model_provider = (
                self._llm.__class__.__name__
                if hasattr(self._llm, "__class__")
                else "unknown"
            )
            svc.add_message(
                session_id=session_id,
                role=role,
                content=content,
                model_provider=model_provider,
            )
        except Exception as e:
            logger.warning(f"保存消息失败: {e}")

    def invoke(self, user_input: str, **kwargs) -> AgentResponse:
        """处理用户输入（ReAct 推理模式）"""
        context = {
            "user_input": user_input,
            "iteration": 0,
            "_metrics": {},
            "_tool_call_log": [],
        }

        def _execute(ctx: dict) -> Any:
            session_id = self._session_id or kwargs.get("session_id")
            history = []

            if self._auto_persist and session_id:
                history = self._load_history(session_id)
                if history:
                    ctx["_history"] = history

            messages = [("user", ctx["user_input"])]
            if history:
                messages = history + messages

            result = self._agent.invoke(
                {"messages": messages},
                config=kwargs.get("config", {}),
            )
            ctx["_last_result"] = result
            ctx["_iterations"] = self._count_ai_messages(result)
            ctx["_content"] = self._extract_content(result)

            tool_results = self._extract_tool_results(result)
            ctx["_tool_results"] = tool_results
            return result

        wrapped_execute = self._middleware.wrap(_execute)

        try:
            wrapped_execute(context)
            result = context["_last_result"]
            content = context["_content"]

            if self._auto_persist and self._session_id:
                self._save_message(self._session_id, "user", user_input)
                self._save_message(self._session_id, "assistant", content)

            return AgentResponse(
                content=content,
                success=True,
                tool_results=context.get("_tool_results", []),
                iterations=context["_iterations"],
                metadata={
                    "agent_type": "langgraph.react",
                    "metrics": context.get("_metrics", {}),
                },
            )

        except Exception as e:
            logger.error(f"Agent 执行错误: {e}")
            return AgentResponse(
                content=f"处理请求时出错: {str(e)}",
                success=False,
                metadata={"error": str(e)},
            )

    def stream(self, user_input: str, **kwargs) -> Generator:
        """流式处理用户输入（带中间件支持和历史持久化）"""
        session_id = self._session_id or kwargs.get("session_id")
        history = []

        if self._auto_persist and session_id:
            history = self._load_history(session_id)

        messages = [("user", user_input)]
        if history:
            messages = history + messages

        context = {
            "user_input": user_input,
            "iteration": 0,
            "_metrics": {},
            "_tool_call_log": [],
        }

        context = self._middleware.before_invoke(context)

        full_response = ""
        try:
            for event in self._agent.stream(
                {"messages": messages},
                config=kwargs.get("config", {}),
            ):
                if isinstance(event, dict):
                    for node_name, node_output in event.items():
                        if isinstance(node_output, dict) and "messages" in node_output:
                            for msg in node_output["messages"]:
                                if hasattr(msg, "content") and msg.content:
                                    full_response += msg.content
                                    yield {"type": "message", "content": msg.content}
                                if hasattr(msg, "name") and msg.name:
                                    tool_context = {
                                        **context,
                                        "tool_name": msg.name,
                                        "tool_args": getattr(msg, "tool_call_id", {}),
                                    }
                                    self._middleware.before_invoke(tool_context)
                                    yield {"type": "tool", "name": msg.name}
                elif hasattr(event, "content"):
                    full_response += event.content
                    yield {"type": "message", "content": event.content}

            self._middleware.after_invoke(context, None)

            if self._auto_persist and session_id:
                self._save_message(session_id, "user", user_input)
                self._save_message(session_id, "assistant", full_response)

        except Exception as e:
            self._middleware.on_error(context, e)
            logger.error(f"Agent 流式处理错误: {e}")
            yield {"type": "error", "error": str(e)}

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

    def add_memory(self, content: str, memory_type: str = "general") -> None:
        """
        添加记忆到会话历史。

        Args:
            content: 记忆内容
            memory_type: 记忆类型（如 "general", "fact", "preference"）
        """
        if self._session_id:
            self._save_message(self._session_id, "memory", content)
            logger.debug(f"添加记忆 [{memory_type}]: {content[:50]}...")

    def search_memory(self, query: str, top_k: int = 5) -> list:
        """
        搜索会话历史中的相关记忆。

        Args:
            query: 搜索关键词
            top_k: 返回数量上限

        Returns:
            匹配的消息列表
        """
        svc = self._get_history_service()
        if svc is None or self._session_id is None:
            return []
        return svc.search_messages(self._session_id, query, limit=top_k)

    def clear_memory(self) -> bool:
        """
        清空当前会话的所有记忆（保留会话）。

        Returns:
            是否清空成功
        """
        svc = self._get_history_service()
        if svc is None or self._session_id is None:
            return False
        return svc.clear_conversation(self._session_id)

    def get_memory_summary(self) -> dict:
        """
        获取当前会话的记忆摘要。

        Returns:
            会话统计信息
        """
        svc = self._get_history_service()
        if svc is None or self._session_id is None:
            return {}
        return svc.get_session_summary(self._session_id)

    def _extract_content(self, result: Any) -> str:
        if isinstance(result, dict) and "messages" in result:
            messages = result["messages"]
            if messages:
                last = messages[-1]
                if hasattr(last, "content"):
                    return last.content
        return str(result)

    def _extract_tool_results(self, result: Any) -> List[dict]:
        """从 LangGraph result 中提取工具调用结果"""
        tool_results = []
        if isinstance(result, dict) and "messages" in result:
            for msg in result["messages"]:
                if hasattr(msg, "type") and msg.type == "tool":
                    tool_results.append({
                        "name": getattr(msg, "name", "unknown"),
                        "content": getattr(msg, "content", ""),
                    })
        return tool_results

    def _count_ai_messages(self, result: Any) -> int:
        if isinstance(result, dict) and "messages" in result:
            return sum(
                1 for m in result["messages"]
                if (isinstance(m, tuple) and m[0] == "ai")
                or (hasattr(m, "type") and m.type == "ai")
            )
        return 0

    def __repr__(self) -> str:
        return f"<LCAgent: tools={len(self._tools)}>"


__all__ = ["LCAgent", "AgentResponse"]
