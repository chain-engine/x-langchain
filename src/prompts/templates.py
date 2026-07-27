# -*- coding: utf-8 -*-
"""
提示模板模块

在 langchain_core.prompts 基础上提供：
- PromptTemplate: 标准文本提示模板
- ChatPromptTemplate: 对话型提示模板
- create_tool_prompt: 工具调用提示工厂
- create_qa_prompt: 问答场景提示工厂
- create_summarization_prompt: 文本摘要提示工厂

所有类同时实现 Runnable 接口，可以与 LangChain LCEL 流水线无缝衔接。
"""

from __future__ import annotations

from typing import Any, List, Optional

from langchain_core.prompts import ChatPromptTemplate as LCChatPromptTemplate
from langchain_core.prompts import PromptTemplate as LCPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda

from core.logger import logger


# =============================================================================
# 文本提示模板
# =============================================================================

class PromptTemplate:
    """
    文本提示模板

    在 langchain_core.prompts.PromptTemplate 基础上提供：
    - from_template(): 类方法工厂
    - format(**kwargs): 渲染文本
    - 实现 Runnable 接口（通过内部 Runnable 组合）

    使用方式：
        ```python
        template = PromptTemplate.from_template("你好，{name}！今天是 {day}。")
        print(template.format(name="小明", day="周一"))
        ```
    """

    def __init__(
        self,
        template: str,
        input_variables: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        """
        初始化文本提示模板

        Args:
            template: 模板字符串，使用 {变量名} 占位
            input_variables: 输入变量名列表；为 None 时由 langchain 自动推断
            **kwargs: 透传给 langchain PromptTemplate 的其他参数
        """
        if input_variables is None:
            self._template = LCPromptTemplate.from_template(template, **kwargs)
        else:
            self._template = LCPromptTemplate(
                template=template,
                input_variables=input_variables,
                **kwargs,
            )

        # 使用 RunnableLambda 暴露 Runnable 能力
        # 注意：RunnableLambda 会以 func(input, **kwargs) 形式调用，
        # 这里包一层以同时支持 dict 输入与 kwargs 输入
        self._runnable: Runnable = RunnableLambda(self._invoke_runnable)

    # ------------------------------------------------------------------ #
    # 工厂
    # ------------------------------------------------------------------ #
    @classmethod
    def from_template(cls, template: str, **kwargs: Any) -> "PromptTemplate":
        """
        从模板字符串创建实例

        Args:
            template: 模板字符串
            **kwargs: 透传给 langchain PromptTemplate

        Returns:
            PromptTemplate 实例
        """
        return cls(template=template, input_variables=None, **kwargs)

    # ------------------------------------------------------------------ #
    # 渲染
    # ------------------------------------------------------------------ #
    def format(self, **kwargs: Any) -> str:
        """
        渲染提示词

        Args:
            **kwargs: 模板变量

        Returns:
            渲染后的字符串
        """
        return self._template.format(**kwargs)

    # ------------------------------------------------------------------ #
    # Runnable 接口
    # ------------------------------------------------------------------ #
    def _invoke_runnable(self, input: Any = None, **kwargs: Any) -> str:
        """RunnableLambda 入口适配"""
        if isinstance(input, dict):
            return self.format(**input)
        if input is not None and not kwargs:
            return self.format(input)
        return self.format(**kwargs)

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> str:
        """
        Runnable 接口入口

        Args:
            input: dict 或其他可解包对象
            config: Runnable 配置
            **kwargs: 其他参数

        Returns:
            渲染后的字符串
        """
        if isinstance(input, dict):
            return self._runnable.invoke(input, config=config, **kwargs)
        return self.format(**kwargs)

    def __or__(self, other: Any) -> Any:
        """
        支持 LCEL：template | next_runnable

        Args:
            other: 下一个 Runnable

        Returns:
            组合后的 RunnableSequence
        """
        return self._runnable.__or__(other)

    # ------------------------------------------------------------------ #
    # 透传
    # ------------------------------------------------------------------ #
    @property
    def input_variables(self) -> List[str]:
        """模板输入变量列表"""
        return list(self._template.input_variables)

    def to_langchain_template(self) -> LCPromptTemplate:
        """导出原始 langchain PromptTemplate"""
        return self._template


# =============================================================================
# 对话提示模板
# =============================================================================

class ChatPromptTemplate:
    """
    对话型提示模板

    在 langchain_core.prompts.ChatPromptTemplate 基础上提供：
    - from_messages(): 类方法工厂
    - add_message(): 动态追加消息
    - format(**kwargs): 渲染为字符串
    - format_messages(**kwargs): 渲染为消息列表

    使用方式：
        ```python
        template = ChatPromptTemplate.from_messages([
            ("system", "你是一个助手"),
            ("user", "{question}"),
        ])
        print(template.format(question="你好"))
        ```
    """

    def __init__(self, template: LCChatPromptTemplate):
        """
        初始化对话提示模板

        Args:
            template: 已经构建好的 langchain ChatPromptTemplate 实例
        """
        self._template = template
        self._runnable: Runnable = RunnableLambda(self._invoke_runnable)

    # ------------------------------------------------------------------ #
    # 工厂
    # ------------------------------------------------------------------ #
    @classmethod
    def from_messages(cls, messages: List[Any]) -> "ChatPromptTemplate":
        """
        从消息列表创建实例

        支持两种格式：
        1. langchain 风格：`[("system", "..."), ("user", "{q}")]`
        2. 字典列表：`[{"role": "system", "content": "..."}]`

        Args:
            messages: 消息列表

        Returns:
            ChatPromptTemplate 实例
        """
        if not messages:
            raise ValueError("messages 不能为空")

        # 兼容 dict 形式
        normalized: List[Any] = []
        for msg in messages:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                normalized.append((msg["role"], msg["content"]))
            else:
                normalized.append(msg)

        template = LCChatPromptTemplate.from_messages(normalized)
        return cls(template)

    # ------------------------------------------------------------------ #
    # 动态追加消息
    # ------------------------------------------------------------------ #
    def add_message(self, role: str, content: str) -> None:
        """
        动态追加一条消息到模板末尾

        Args:
            role: 角色名称（system / user / assistant）
            content: 消息内容

        注意：该方法会替换底层 langchain ChatPromptTemplate，
        已有变量关系会保留。
        """
        try:
            self._template = self._template + LCChatPromptTemplate.from_messages(
                [(role, content)]
            )
            logger.debug(f"已向 ChatPromptTemplate 追加消息: role={role}")
        except Exception as e:
            logger.error(f"追加消息失败: {e}")
            raise

    # ------------------------------------------------------------------ #
    # 渲染
    # ------------------------------------------------------------------ #
    def format(self, **kwargs: Any) -> str:
        """
        渲染为字符串

        Args:
            **kwargs: 模板变量

        Returns:
            渲染好的字符串（多消息用换行连接）
        """
        return self._template.format(**kwargs)

    def format_messages(self, **kwargs: Any) -> List[Any]:
        """
        渲染为消息对象列表（BaseMessage 子类）

        Args:
            **kwargs: 模板变量

        Returns:
            LangChain BaseMessage 列表
        """
        return self._template.format_messages(**kwargs)

    # ------------------------------------------------------------------ #
    # Runnable 接口
    # ------------------------------------------------------------------ #
    def _invoke_runnable(self, input: Any = None, **kwargs: Any) -> str:
        """RunnableLambda 入口适配"""
        if isinstance(input, dict):
            return self.format(**input)
        if input is not None and not kwargs:
            return self.format(input)
        return self.format(**kwargs)

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> str:
        """
        Runnable 接口入口

        Args:
            input: dict 或其他可解包对象
            config: Runnable 配置
            **kwargs: 其他参数

        Returns:
            渲染后的字符串
        """
        if isinstance(input, dict):
            return self._runnable.invoke(input, config=config, **kwargs)
        return self.format(**kwargs)

    def __or__(self, other: Any) -> Any:
        """支持 LCEL 组合"""
        return self._runnable.__or__(other)

    # ------------------------------------------------------------------ #
    # 透传
    # ------------------------------------------------------------------ #
    @property
    def input_variables(self) -> List[str]:
        """模板输入变量列表"""
        return list(self._template.input_variables)

    @property
    def messages(self) -> List[Any]:
        """内部消息列表"""
        return list(self._template.messages)

    def to_langchain_template(self) -> LCChatPromptTemplate:
        """导出原始 langchain ChatPromptTemplate"""
        return self._template


# =============================================================================
# 工厂函数
# =============================================================================

def create_tool_prompt(
    tool_name: str,
    tool_description: str,
    input_descriptions: dict,
) -> PromptTemplate:
    """
    工厂函数：为单个工具调用创建提示模板

    生成如下形式的提示：
        工具名称: xxx
        工具描述: xxx
        参数说明:
        - arg1: 描述1
        - arg2: 描述2
        请按 JSON 格式输出调用参数。

    Args:
        tool_name: 工具名称
        tool_description: 工具描述
        input_descriptions: 参数名 -> 描述 的字典

    Returns:
        PromptTemplate 实例

    使用示例：
        ```python
        prompt = create_tool_prompt(
            tool_name="get_weather",
            tool_description="查询指定城市的天气",
            input_descriptions={"city": "城市名称", "date": "日期，YYYY-MM-DD"},
        )
        print(prompt.format(input='{"city": "北京"}'))
        ```
    """
    if not tool_name:
        raise ValueError("tool_name 不能为空")

    param_lines = "\n".join(
        f"- {k}: {v}" for k, v in (input_descriptions or {}).items()
    )
    if not param_lines:
        param_lines = "（无参数）"

    template_str = (
        f"工具名称: {tool_name}\n"
        f"工具描述: {tool_description}\n"
        f"参数说明:\n{param_lines}\n\n"
        "请根据用户输入，按 JSON 格式输出该工具的调用参数。\n"
        "用户输入: {input}\n"
        "工具调用参数(JSON):"
    )

    return PromptTemplate.from_template(template_str)


def create_qa_prompt(
    context_template: str = "{context}",
    question_template: str = "{question}",
) -> PromptTemplate:
    """
    工厂函数：为问答场景（QA / RAG）创建提示模板

    生成如下形式的提示：
        已知信息:
        {context}

        问题: {question}
        请基于已知信息回答问题。如果已知信息不足以回答，请明确说明。

    Args:
        context_template: 上下文模板，默认使用 {context}
        question_template: 问题模板，默认使用 {question}

    Returns:
        PromptTemplate 实例

    使用示例：
        ```python
        prompt = create_qa_prompt()
        print(prompt.format(context="北京是中国的首都。", question="中国的首都是哪里？"))
        ```
    """
    template_str = (
        "已知信息:\n"
        f"{context_template}\n\n"
        "问题:\n"
        f"{question_template}\n\n"
        "请基于已知信息回答问题。如果已知信息不足以回答，请明确说明。"
    )
    return PromptTemplate.from_template(template_str)


def create_summarization_prompt(
    summarize_template: str = "请总结以下内容:\n{text}",
) -> PromptTemplate:
    """
    工厂函数：为文本摘要任务创建提示模板

    Args:
        summarize_template: 摘要模板字符串，必须包含 {text} 占位符

    Returns:
        PromptTemplate 实例

    使用示例：
        ```python
        prompt = create_summarization_prompt()
        print(prompt.format(text="长文本内容..."))
        ```
    """
    if "{text}" not in summarize_template:
        raise ValueError("summarize_template 必须包含 {text} 占位符")

    return PromptTemplate.from_template(summarize_template)


__all__ = [
    "PromptTemplate",
    "ChatPromptTemplate",
    "create_tool_prompt",
    "create_qa_prompt",
    "create_summarization_prompt",
]