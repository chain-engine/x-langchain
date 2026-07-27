# -*- coding: utf-8 -*-
"""
高级 Memory 模块

提供 LangChain 高级 Memory 组件封装：
- ConversationSummaryMemory: 摘要记忆（自动压缩历史）
- ConversationBufferWindowMemory: 窗口记忆（保留最近 N 条）
- ConversationEntityMemory: 实体记忆（提取实体关系）
- CombinedMemory: 组合记忆（多类型混合）
"""

from __future__ import annotations

from typing import Any, Optional

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable

from core.logger import logger


# =============================================================================
# 摘要记忆
# =============================================================================


class ConversationSummaryMemory:
    """
    摘要记忆

    自动将对话历史压缩为摘要，减少 token 消耗。
    保留对话的核心信息，同时大幅减少上下文长度。

    使用方式：
        ```python
        from memories import ConversationSummaryMemory
        from llms import create_chat_model

        llm = create_chat_model("deepseek")
        memory = ConversationSummaryMemory(
            llm=llm,
            max_summary_length=500,
        )

        # 添加消息
        memory.add_user_message("我想了解北京的历史")
        memory.add_ai_message("北京是中国的首都，有着三千多年的历史...")

        # 获取记忆变量（供 Chain 使用）
        vars = memory.load_memory_variables({})
        print(vars["history"])  # 可能已经被压缩为摘要
        ```
    """

    DEFAULT_SUMMARY_PROMPT = """请总结以下对话的要点：

当前对话摘要:
{summary}

新消息:
{new_lines}

请生成更新后的摘要，保留关键信息："""

    def __init__(
        self,
        llm: BaseLanguageModel,
        chat_memory: Optional[BaseChatMessageHistory] = None,
        max_summary_length: int = 500,
        prompt_template: Optional[str] = None,
        summary_message_limit: int = 10,
    ):
        """
        初始化摘要记忆

        Args:
            llm: 用于生成摘要的 LLM
            chat_memory: 底层聊天历史存储
            max_summary_length: 摘要最大长度（字符数）
            prompt_template: 自定义摘要提示模板
            summary_message_limit: 多少条消息后开始生成摘要
        """
        from langchain_core.chat_history import InMemoryChatMessageHistory

        self._llm = llm
        self._chat_memory = chat_memory or InMemoryChatMessageHistory()
        self._max_summary_length = max_summary_length
        self._summary_message_limit = summary_message_limit
        self._prompt_template = prompt_template or self.DEFAULT_SUMMARY_PROMPT
        self._summary: str = ""

        # 构建摘要 Chain
        self._summary_chain = self._build_summary_chain()

    def _build_summary_chain(self) -> Runnable:
        """构建摘要生成 Chain"""
        prompt = PromptTemplate.from_template(self._prompt_template)
        return prompt | self._llm | StrOutputParser()

    def add_user_message(self, message: str) -> None:
        """添加用户消息"""
        self._chat_memory.add_message(HumanMessage(content=message))
        self._maybe_summarize()

    def add_ai_message(self, message: str) -> None:
        """添加 AI 消息"""
        self._chat_memory.add_message(AIMessage(content=message))
        self._maybe_summarize()

    def add_message(self, role: str, content: str) -> None:
        """添加消息"""
        if role.lower() in ("user", "human"):
            self.add_user_message(content)
        else:
            self.add_ai_message(content)

    def _maybe_summarize(self) -> None:
        """检查是否需要生成摘要"""
        messages = self._chat_memory.messages
        if len(messages) >= self._summary_message_limit:
            self.generate_summary()

    def generate_summary(self) -> str:
        """
        生成当前对话的摘要

        Returns:
            生成的摘要内容
        """
        messages = self._chat_memory.messages
        if not messages:
            return ""

        # 将新消息转为文本
        new_lines = "\n".join(
            f"{'Human' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
            for m in messages
        )

        try:
            # 生成摘要
            result = self._summary_chain.invoke({
                "summary": self._summary or "（无）",
                "new_lines": new_lines,
            })

            self._summary = result[: self._max_summary_length]
            logger.debug(f"ConversationSummaryMemory: 生成摘要，长度={len(self._summary)}")

            return self._summary

        except Exception:
            logger.warning(f"摘要生成失败")
            return self._summary

    def get_summary(self) -> str:
        """获取当前摘要"""
        return self._summary

    def load_memory_variables(self, inputs: dict) -> dict:
        """加载记忆变量"""
        messages = self._chat_memory.messages
        if self._return_messages:
            return {"history": messages}
        else:
            return {
                "history": "\n".join(
                    f"{'Human' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
                    for m in messages
                )
            }

    def get_messages(self) -> list[BaseMessage]:
        """获取所有消息"""
        return list(self._chat_memory.messages)

    def clear(self) -> None:
        """清空记忆"""
        self._chat_memory.clear()
        self._summary = ""


# =============================================================================
# 窗口记忆
# =============================================================================


class ConversationBufferWindowMemory:
    """
    窗口记忆

    只保留最近 N 条对话消息，自动丢弃更早的消息。
    适用于有固定上下文长度限制的场景。

    使用方式：
        ```python
        from memories import ConversationBufferWindowMemory

        memory = ConversationBufferWindowMemory(
            k=5,  # 只保留最近 5 条消息
            return_messages=True,
        )

        memory.add_user_message("你好")
        memory.add_ai_message("你好！")
        # ... 更多消息

        # 只有最近 5 条消息
        vars = memory.load_memory_variables({})
        ```
    """

    def __init__(
        self,
        k: int = 5,
        chat_memory: Optional[BaseChatMessageHistory] = None,
        return_messages: bool = True,
    ):
        """
        初始化窗口记忆

        Args:
            k: 保留的消息数量
            chat_memory: 底层聊天历史存储
            return_messages: 返回消息列表还是字符串
        """
        from langchain_core.chat_history import InMemoryChatMessageHistory

        self._k = k
        self._chat_memory = chat_memory or InMemoryChatMessageHistory()
        self._return_messages = return_messages

    @property
    def k(self) -> int:
        """窗口大小"""
        return self._k

    def set_k(self, k: int) -> None:
        """动态调整窗口大小"""
        self._k = k

    def add_user_message(self, message: str) -> None:
        """添加用户消息"""
        self._chat_memory.add_message(HumanMessage(content=message))
        self._trim_history()

    def add_ai_message(self, message: str) -> None:
        """添加 AI 消息"""
        self._chat_memory.add_message(AIMessage(content=message))
        self._trim_history()

    def add_message(self, role: str, content: str) -> None:
        """添加消息"""
        if role.lower() in ("user", "human"):
            self.add_user_message(content)
        else:
            self.add_ai_message(content)

    def _trim_history(self) -> None:
        """裁剪超出窗口的历史"""
        messages = self._chat_memory.messages
        if len(messages) > self._k:
            # 保留最近的 k 条消息
            self._chat_memory.messages = messages[-self._k :]

    def load_memory_variables(self, inputs: dict) -> dict:
        """加载记忆变量"""
        messages = self._chat_memory.messages
        if self._return_messages:
            return {"history": messages}
        else:
            return {
                "history": "\n".join(
                    f"{'Human' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
                    for m in messages
                )
            }

    def get_messages(self) -> list[BaseMessage]:
        """获取当前窗口内的消息"""
        return list(self._chat_memory.messages)

    def clear(self) -> None:
        """清空记忆"""
        self._chat_memory.clear()


# =============================================================================
# 实体记忆
# =============================================================================


class ConversationEntityMemory:
    """
    实体记忆

    从对话中提取并跟踪实体（人名、地点、组织等）及其关系。
    适用于需要记住实体信息的场景。

    使用方式：
        ```python
        from memories import ConversationEntityMemory

        memory = ConversationEntityMemory(llm=llm)

        memory.add_user_message("我的朋友张三住在上海，他在阿里巴巴工作")
        memory.add_ai_message("好的，我已经记住张三的信息了。")

        # 获取实体记忆
        entities = memory.get_entities()
        print(entities)
        # {'张三': {'location': '上海', 'employer': '阿里巴巴'}, ...}
        ```
    """

    ENTITY_EXTRACTION_PROMPT = """从以下对话中提取实体及其属性。

对话:
{history}

请提取所有实体及其关系，输出为 JSON 格式：
{{
  "实体名称": {{
    "属性1": "值1",
    "属性2": "值2"
  }}
}}

只提取明确提到的信息，不要推测。"""

    def __init__(
        self,
        llm: BaseLanguageModel,
        chat_memory: Optional[BaseChatMessageHistory] = None,
        prompt_template: Optional[str] = None,
    ):
        """
        初始化实体记忆

        Args:
            llm: 用于提取实体的 LLM
            chat_memory: 底层聊天历史存储
            prompt_template: 自定义实体提取提示
        """
        from langchain_core.chat_history import InMemoryChatMessageHistory

        self._llm = llm
        self._chat_memory = chat_memory or InMemoryChatMessageHistory()
        self._prompt_template = prompt_template or self.ENTITY_EXTRACTION_PROMPT
        self._entities: dict[str, dict] = {}

        self._extraction_chain = self._build_extraction_chain()

    def _build_extraction_chain(self) -> Runnable:
        """构建实体提取 Chain"""
        prompt = PromptTemplate.from_template(self._prompt_template)
        return prompt | self._llm | StrOutputParser()

    def add_user_message(self, message: str) -> None:
        """添加用户消息"""
        self._chat_memory.add_message(HumanMessage(content=message))

    def add_ai_message(self, message: str) -> None:
        """添加 AI 消息"""
        self._chat_memory.add_message(AIMessage(content=message))

    def add_message(self, role: str, content: str) -> None:
        """添加消息"""
        if role.lower() in ("user", "human"):
            self.add_user_message(content)
        else:
            self.add_ai_message(content)

    def extract_entities(self) -> dict[str, dict]:
        """
        从当前对话中提取实体

        Returns:
            实体字典 {实体名: {属性: 值}}
        """
        messages = self._chat_memory.messages
        if not messages:
            return {}

        history_text = "\n".join(
            f"{'Human' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
            for m in messages
        )

        try:
            result = self._extraction_chain.invoke({"history": history_text})
            self._entities = self._parse_entity_result(result)
            return self._entities
        except Exception:
            logger.warning(f"实体提取失败")
            return self._entities

    def _parse_entity_result(self, result: str) -> dict[str, dict]:
        """解析实体提取结果"""
        import json
        import re

        # 尝试提取 JSON
        json_match = re.search(r"\{[\s\S]*\}", result)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        return {}

    def get_entities(self) -> dict[str, dict]:
        """获取当前实体记忆"""
        return dict(self._entities)

    def update_entity(self, entity: str, **kwargs: Any) -> None:
        """手动更新实体信息"""
        if entity not in self._entities:
            self._entities[entity] = {}
        self._entities[entity].update(kwargs)

    def load_memory_variables(self, inputs: dict) -> dict:
        """加载记忆变量"""
        # 先提取实体
        self.extract_entities()

        # 构建实体描述
        entity_lines = []
        for entity, attrs in self._entities.items():
            attr_str = ", ".join(f"{k}: {v}" for k, v in attrs.items())
            entity_lines.append(f"{entity}: {attr_str}")

        entity_text = "\n".join(entity_lines) if entity_lines else "（无已知实体）"

        return {"history": self._chat_memory.messages, "entities": entity_text}

    def get_messages(self) -> list[BaseMessage]:
        """获取所有消息"""
        return list(self._chat_memory.messages)

    def clear(self) -> None:
        """清空记忆"""
        self._chat_memory.clear()
        self._entities = {}


# =============================================================================
# 组合记忆
# =============================================================================


class CombinedMemory:
    """
    组合记忆

    将多种记忆类型组合使用，取长补短。
    例如：摘要记忆 + 窗口记忆。

    使用方式：
        ```python
        from memories import CombinedMemory, ConversationSummaryMemory, ConversationBufferWindowMemory

        summary_mem = ConversationSummaryMemory(llm=llm)
        window_mem = ConversationBufferWindowMemory(k=5)

        combined = CombinedMemory(
            memories=[summary_mem, window_mem],
            memory_key="history",
        )

        combined.add_user_message("你好")
        vars = combined.load_memory_variables({})
        ```
    """

    def __init__(
        self,
        memories: list[Any],
        memory_key: str = "history",
        default_memory_class: type = ConversationBufferWindowMemory,
    ):
        """
        初始化组合记忆

        Args:
            memories: 记忆实例列表
            memory_key: 记忆变量在输出字典中的键名
            default_memory_class: 默认记忆类（用于兼容旧接口）
        """
        self._memories = memories
        self._memory_key = memory_key
        self._default_memory_class = default_memory_class

    def add_user_message(self, message: str) -> None:
        """向所有记忆添加用户消息"""
        for memory in self._memories:
            if hasattr(memory, "add_user_message"):
                memory.add_user_message(message)

    def add_ai_message(self, message: str) -> None:
        """向所有记忆添加 AI 消息"""
        for memory in self._memories:
            if hasattr(memory, "add_ai_message"):
                memory.add_ai_message(message)

    def add_message(self, role: str, content: str) -> None:
        """向所有记忆添加消息"""
        for memory in self._memories:
            if hasattr(memory, "add_message"):
                memory.add_message(role, content)
            elif hasattr(memory, "add_user_message") and role.lower() in ("user", "human"):
                memory.add_user_message(content)
            elif hasattr(memory, "add_ai_message"):
                memory.add_ai_message(content)

    def load_memory_variables(self, inputs: dict) -> dict:
        """加载所有记忆变量"""
        result = {}
        for memory in self._memories:
            if hasattr(memory, "load_memory_variables"):
                mem_vars = memory.load_memory_variables(inputs)
                result.update(mem_vars)

        return result

    def get_messages(self) -> list[BaseMessage]:
        """从第一个支持的记忆获取消息"""
        for memory in self._memories:
            if hasattr(memory, "get_messages"):
                return memory.get_messages()
        return []

    def clear(self) -> None:
        """清空所有记忆"""
        for memory in self._memories:
            if hasattr(memory, "clear"):
                memory.clear()


# =============================================================================
# 高级记忆工厂类
# =============================================================================


class AdvancedMemory:
    """
    高级记忆工厂

    根据类型创建不同的高级记忆实例。

    使用方式：
        ```python
        from memories import AdvancedMemory

        factory = AdvancedMemory(llm=llm)

        # 创建摘要记忆
        mem = factory.create(memory_type="summary", k=5)

        # 创建实体记忆
        mem = factory.create(memory_type="entity")
        ```
    """

    SUPPORTED_TYPES = ["summary", "window", "entity", "combined"]

    def __init__(
        self,
        llm: Optional[BaseLanguageModel] = None,
        chat_memory: Optional[BaseChatMessageHistory] = None,
    ):
        """
        初始化高级记忆工厂

        Args:
            llm: 用于生成摘要/实体的 LLM
            chat_memory: 底层聊天历史存储
        """
        self._llm = llm
        self._chat_memory = chat_memory

    def create(
        self,
        memory_type: str = "window",
        **kwargs: Any,
    ) -> Any:
        """
        创建指定类型的高级记忆

        Args:
            memory_type: 记忆类型
                - "summary": 摘要记忆
                - "window": 窗口记忆
                - "entity": 实体记忆
                - "combined": 组合记忆
            **kwargs: 透传给记忆构造函数

        Returns:
            记忆实例
        """
        if memory_type not in self.SUPPORTED_TYPES:
            raise ValueError(
                f"不支持的记忆类型: {memory_type}，"
                f"支持的: {self.SUPPORTED_TYPES}"
            )

        if memory_type == "summary":
            if not self._llm:
                raise ValueError("摘要记忆需要提供 llm 参数")
            return ConversationSummaryMemory(
                llm=self._llm,
                chat_memory=self._chat_memory,
                **kwargs,
            )

        elif memory_type == "window":
            return ConversationBufferWindowMemory(
                chat_memory=self._chat_memory,
                **kwargs,
            )

        elif memory_type == "entity":
            if not self._llm:
                raise ValueError("实体记忆需要提供 llm 参数")
            return ConversationEntityMemory(
                llm=self._llm,
                chat_memory=self._chat_memory,
                **kwargs,
            )

        elif memory_type == "combined":
            memories = kwargs.pop("memories", [])
            return CombinedMemory(memories=memories, **kwargs)

        raise ValueError(f"未知的记忆类型: {memory_type}")


# =============================================================================
# 工厂函数
# =============================================================================


def create_advanced_memory(
    memory_type: str = "window",
    llm: Optional[BaseLanguageModel] = None,
    **kwargs: Any,
) -> Any:
    """
    工厂函数：创建高级记忆

    Args:
        memory_type: 记忆类型 (summary / window / entity / combined)
        llm: 用于摘要/实体的 LLM
        **kwargs: 透传给对应记忆构造函数

    Returns:
        记忆实例

    Example:
        ```python
        # 摘要记忆
        memory = create_advanced_memory("summary", llm=llm)

        # 窗口记忆
        memory = create_advanced_memory("window", k=10)

        # 组合记忆
        memory = create_advanced_memory("combined", memories=[mem1, mem2])
        ```
    """
    factory = AdvancedMemory(llm=llm)
    return factory.create(memory_type=memory_type, **kwargs)


def create_summary_memory(
    llm: BaseLanguageModel,
    **kwargs: Any,
) -> ConversationSummaryMemory:
    """
    工厂函数：创建摘要记忆

    Args:
        llm: 用于生成摘要的 LLM
        **kwargs: 透传给 ConversationSummaryMemory

    Returns:
        ConversationSummaryMemory 实例
    """
    return ConversationSummaryMemory(llm=llm, **kwargs)


def create_window_memory(
    k: int = 5,
    **kwargs: Any,
) -> ConversationBufferWindowMemory:
    """
    工厂函数：创建窗口记忆

    Args:
        k: 窗口大小
        **kwargs: 透传给 ConversationBufferWindowMemory

    Returns:
        ConversationBufferWindowMemory 实例
    """
    return ConversationBufferWindowMemory(k=k, **kwargs)


def create_entity_memory(
    llm: BaseLanguageModel,
    **kwargs: Any,
) -> ConversationEntityMemory:
    """
    工厂函数：创建实体记忆

    Args:
        llm: 用于提取实体的 LLM
        **kwargs: 透传给 ConversationEntityMemory

    Returns:
        ConversationEntityMemory 实例
    """
    return ConversationEntityMemory(llm=llm, **kwargs)


__all__ = [
    "ConversationSummaryMemory",
    "ConversationBufferWindowMemory",
    "ConversationEntityMemory",
    "CombinedMemory",
    "AdvancedMemory",
    "create_advanced_memory",
    "create_summary_memory",
    "create_window_memory",
    "create_entity_memory",
]
