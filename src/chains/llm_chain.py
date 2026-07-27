# -*- coding: utf-8 -*-
"""
LLMChain - Prompt + LLM 最简链

封装 LangChain 的 LLMChain 模式。
"""

from __future__ import annotations

from typing import Any, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from core.logger import logger


class LLMChain:
    """
    LLMChain - Prompt + LLM 最简链

    封装 LangChain 的经典 LLMChain 模式：
    Input -> [Prompt] -> [LLM] -> [OutputParser] -> Output

    与 LCAgent 的区别：
    - LLMChain: 单一 LLM 调用，无 Agent 循环，无工具调用
    - LCAgent: 包含 ReAct 循环和工具调用

    使用方式：
        ```python
        from llms import create_chat_model
        from chains import LLMChain, create_llm_chain

        llm = create_chat_model()
        chain = create_llm_chain(
            llm=llm,
            system_message="你是一个助手。",
            user_template="{question}",
        )

        result = chain.invoke({"question": "你好"})
        # -> "你好！有什么可以帮你的吗？"
        ```
    """

    def __init__(
        self,
        llm: BaseChatModel,
        prompt: ChatPromptTemplate,
        output_parser: Optional[Runnable] = None,
        *,
        verbose: bool = False,
        temperature: Optional[float] = None,
    ):
        """
        初始化 LLMChain

        Args:
            llm: 聊天模型
            prompt: 提示模板
            output_parser: 输出解析器（默认为 StrOutputParser）
            verbose: 是否输出详细信息
            temperature: LLM 温度参数（None 使用默认）
        """
        self._llm = llm
        self._prompt = prompt
        self._output_parser = output_parser or StrOutputParser()
        self._verbose = verbose
        self._temperature = temperature

        # 构建 LCEL 链
        self._chain: Runnable = self._build_chain()

        logger.info(f"LLMChain 初始化: prompt_variables={prompt.input_variables}")

    def _build_chain(self) -> Runnable:
        """构建 LCEL 链"""
        chain: Runnable = self._prompt | self._llm
        if self._output_parser:
            chain = chain | self._output_parser
        return chain

    @property
    def chain(self) -> Runnable:
        """获取底层 LCEL Runnable"""
        return self._chain

    def invoke(self, input: dict[str, Any], **kwargs: Any) -> Any:
        """
        同步调用链

        Args:
            input: 输入字典（必须包含 prompt 中的所有变量）
            **kwargs: 其他参数

        Returns:
            链输出
        """
        return self._chain.invoke(input, **kwargs)

    async def ainvoke(self, input: dict[str, Any], **kwargs: Any) -> Any:
        """
        异步调用链

        Args:
            input: 输入字典
            **kwargs: 其他参数

        Returns:
            异步链输出
        """
        return await self._chain.ainvoke(input, **kwargs)

    def stream(self, input: dict[str, Any], **kwargs: Any):
        """
        流式调用

        Args:
            input: 输入字典
            **kwargs: 其他参数

        Yields:
            流式输出块
        """
        return self._chain.stream(input, **kwargs)

    async def astream(self, input: dict[str, Any], **kwargs: Any):
        """
        异步流式调用

        Yields:
            异步流式输出块
        """
        return self._chain.astream(input, **kwargs)

    def run(self, **kwargs: Any) -> str:
        """
        便捷方法：直接运行并返回字符串

        Args:
            **kwargs: 输入变量

        Returns:
            输出字符串
        """
        result = self._chain.invoke(kwargs)
        return str(result) if result is not None else ""

    def __or__(self, other: Runnable) -> Runnable:
        """支持 LCEL: chain | next"""
        return self._chain | other

    def __repr__(self) -> str:
        return f"<LLMChain: variables={self._prompt.input_variables}>"


# =============================================================================
# 工厂函数
# =============================================================================


def create_llm_chain(
    llm: BaseChatModel,
    *,
    system_message: Optional[str] = None,
    user_template: Optional[str] = None,
    prompt_template: Optional[str] = None,
    messages: Optional[list[tuple[str, str]]] = None,
    output_parser: Optional[Runnable] = None,
    temperature: Optional[float] = None,
) -> LLMChain:
    """
    工厂函数：创建 LLMChain

    支持三种 prompt 定义方式（优先级从高到低）：
    1. messages: 消息列表 [("system", "..."), ("user", "{input}")]
    2. prompt_template: 完整模板字符串
    3. system_message + user_template: 分离定义

    Args:
        llm: 聊天模型
        system_message: 系统消息
        user_template: 用户模板（支持 {变量} 占位符）
        prompt_template: 完整提示模板字符串
        messages: 消息列表（优先级最高）
        output_parser: 输出解析器
        temperature: LLM 温度参数

    Returns:
        LLMChain 实例

    示例：
        ```python
        chain = create_llm_chain(
            llm=llm,
            system_message="你是一个翻译助手",
            user_template="将以下中文翻译为英文: {text}",
        )
        result = chain.invoke({"text": "你好世界"})
        ```
    """
    # 构建 prompt
    if messages:
        prompt = ChatPromptTemplate.from_messages(messages)
    elif prompt_template:
        prompt = ChatPromptTemplate.from_template(prompt_template)
    elif user_template:
        if system_message:
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_message),
                ("user", user_template),
            ])
        else:
            prompt = ChatPromptTemplate.from_messages([
                ("user", user_template),
            ])
    else:
        raise ValueError("必须提供 messages、prompt_template 或 user_template 之一")

    return LLMChain(
        llm=llm,
        prompt=prompt,
        output_parser=output_parser,
        temperature=temperature,
    )


__all__ = ["LLMChain", "create_llm_chain"]
