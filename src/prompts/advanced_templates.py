# -*- coding: utf-8 -*-
"""
高级提示模板模块

在 langchain_core.prompts 基础上提供：
- PipelinePromptTemplate: 多级 prompt 管道模板
- ChatMessagePromptTemplate: 消息级别模板
- FewShotChatMessagePromptTemplate: 少样本聊天模板
- DynamicPipelinePromptTemplate: 动态管道提示模板
- create_pipeline_prompt: 快速创建管道提示
- create_chat_message_prompt: 快速创建消息模板
"""

from __future__ import annotations

from typing import Any, List, Optional

from langchain_core.messages import BaseMessage
from langchain_core.prompts import (
    ChatMessagePromptTemplate as LCChatMessagePromptTemplate,
    FewShotChatMessagePromptTemplate as LCFewShotChatMessagePromptTemplate,
    PromptTemplate as LCPromptTemplate,
)
from langchain_core.runnables import RunnableLambda

from core.logger import logger


# =============================================================================
# 管道提示模板
# =============================================================================


class PipelinePromptTemplate:
    """
    多级管道提示模板

    由多个子模板组成，按顺序执行并拼接最终提示。
    适用于需要多步骤处理的复杂提示场景。

    使用方式：
        ```python
        from prompts import PipelinePromptTemplate

        template = PipelinePromptTemplate.from_prompts(
            pipeline_prompts=[
                ("introduction", "你是一个{role}，名叫{name}。"),
                ("context", "当前上下文：{context}"),
                ("question", "用户问题：{question}"),
            ],
            final_prompt="{introduction}\\n{context}\\n{question}\\n请回答：",
        )

        result = template.format(
            role="助手",
            name="小明",
            context="用户正在学习 Python",
            question="什么是列表推导式？"
        )
        ```
    """

    def __init__(
        self,
        pipeline_prompts: List[tuple[str, Any]],
        final_prompt: Any,
    ):
        """
        初始化管道提示模板

        Args:
            pipeline_prompts: 管道中的提示列表，每项为 (变量名, 模板) 元组
            final_prompt: 最终提示模板，使用前面所有变量的组合
        """
        self._pipeline_prompts = pipeline_prompts
        self._final_prompt = final_prompt

    @classmethod
    def from_prompts(
        cls,
        pipeline_prompts: List[tuple[str, str]],
        final_prompt: str,
    ) -> "PipelinePromptTemplate":
        """
        从提示列表创建管道模板

        Args:
            pipeline_prompts: [(变量名, 模板), ...]
            final_prompt: 最终模板字符串

        Returns:
            PipelinePromptTemplate 实例
        """
        normalized_prompts = [
            (name, LCPromptTemplate.from_template(template))
            for name, template in pipeline_prompts
        ]
        final = LCPromptTemplate.from_template(final_prompt)
        return cls(pipeline_prompts=normalized_prompts, final_prompt=final)

    def format(self, **kwargs: Any) -> str:
        """
        渲染最终提示词

        Args:
            **kwargs: 各子模板的变量

        Returns:
            渲染好的字符串
        """
        context = {}
        for name, template in self._pipeline_prompts:
            if hasattr(template, "format"):
                context[name] = template.format(**kwargs)
            else:
                context[name] = str(template)

        if hasattr(self._final_prompt, "format"):
            return self._final_prompt.format(**context)
        return str(self._final_prompt).format(**context)

    def invoke(self, input_data: Any, config: Any = None, **kwargs: Any) -> str:
        """Runnable 接口"""
        if isinstance(input_data, dict):
            kwargs.update(input_data)
        return self.format(**kwargs)

    def __or__(self, other: Any) -> Any:
        """支持 LCEL"""
        return RunnableLambda(self.invoke).__or__(other)

    @property
    def input_variables(self) -> List[str]:
        """获取所有输入变量"""
        variables = set()
        for _, tmpl in self._pipeline_prompts:
            if hasattr(tmpl, "input_variables"):
                variables.update(tmpl.input_variables)
        if hasattr(self._final_prompt, "input_variables"):
            variables.update(self._final_prompt.input_variables)
        return list(variables)


# =============================================================================
# 聊天消息提示模板
# =============================================================================


class ChatMessagePromptTemplate:
    """
    聊天消息提示模板

    使用显式的角色类型创建消息模板。
    支持自定义角色（如 "assistant", "system", "user"）。

    使用方式：
        ```python
        from prompts import ChatMessagePromptTemplate

        template = ChatMessagePromptTemplate.from_messages([
            ("system", "你是一个{personality}助手"),
            ("user", "{question}"),
        ])

        result = template.format(personality="专业", question="如何学习编程？")
        ```
    """

    def __init__(self, template: LCChatMessagePromptTemplate):
        """初始化聊天消息提示模板"""
        self._template = template

    @classmethod
    def from_messages(
        cls,
        messages: List[tuple[str, str]],
    ) -> "ChatMessagePromptTemplate":
        """
        从消息列表创建模板

        Args:
            messages: [(角色, 内容), ...] 元组列表

        Returns:
            ChatMessagePromptTemplate 实例
        """
        lc_template = LCChatMessagePromptTemplate.from_messages(messages)
        return cls(lc_template)

    @classmethod
    def from_role(
        cls,
        role: str,
        template: str,
    ) -> "ChatMessagePromptTemplate":
        """
        从单个角色创建模板

        Args:
            role: 角色名
            template: 模板内容

        Returns:
            ChatMessagePromptTemplate 实例
        """
        lc_template = LCChatMessagePromptTemplate.from_messages([(role, template)])
        return cls(lc_template)

    def format(self, **kwargs: Any) -> str:
        """渲染模板"""
        return self._template.format(**kwargs)

    def format_messages(self, **kwargs: Any) -> List[BaseMessage]:
        """渲染为消息对象"""
        return self._template.format_messages(**kwargs)

    def invoke(self, input_data: Any, config: Any = None, **kwargs: Any) -> str:
        """Runnable 接口"""
        if isinstance(input_data, dict):
            kwargs.update(input_data)
        return self.format(**kwargs)

    def __or__(self, other: Any) -> Any:
        """支持 LCEL"""
        return self._template.__or__(other)

    @property
    def input_variables(self) -> List[str]:
        """获取输入变量"""
        return list(self._template.input_variables)


# =============================================================================
# 少样本聊天提示模板
# =============================================================================


class FewShotChatMessagePromptTemplate:
    """
    少样本聊天提示模板

    在对话场景下使用少样本学习的提示模板。
    自动将示例格式化为对话格式。

    使用方式：
        ```python
        from prompts import FewShotChatMessagePromptTemplate, ChatMessagePromptTemplate

        examples = [
            {"question": "1+1=?", "answer": "2"},
            {"question": "2+2=?", "answer": "4"},
        ]

        example_template = ChatMessagePromptTemplate.from_messages([
            ("human", "{question}"),
            ("ai", "{answer}"),
        ])

        template = FewShotChatMessagePromptTemplate.from_examples(
            examples=examples,
            example_prompt=example_template,
            k=2,
        )

        result = template.format(question="3+3=?")
        ```
    """

    def __init__(
        self,
        template: LCFewShotChatMessagePromptTemplate,
    ):
        """初始化少样本聊天提示模板"""
        self._template = template

    @classmethod
    def from_examples(
        cls,
        examples: List[dict],
        example_prompt: Any,
        k: int = 4,
        **kwargs: Any,
    ) -> "FewShotChatMessagePromptTemplate":
        """
        从示例列表创建模板

        Args:
            examples: 示例字典列表
            example_prompt: 示例模板（ChatMessagePromptTemplate）
            k: 使用最近 k 个示例
            **kwargs: 其他参数

        Returns:
            FewShotChatMessagePromptTemplate 实例
        """
        lc_example_prompt = (
            example_prompt._template
            if hasattr(example_prompt, "_template")
            else example_prompt
        )

        lc_template = LCFewShotChatMessagePromptTemplate.from_examples(
            examples=examples,
            example_prompt=lc_example_prompt,
            k=k,
            **kwargs,
        )
        return cls(lc_template)

    def format(self, **kwargs: Any) -> str:
        """渲染模板"""
        return self._template.format(**kwargs)

    def format_messages(self, **kwargs: Any) -> List[BaseMessage]:
        """渲染为消息对象"""
        return self._template.format_messages(**kwargs)

    def invoke(self, input_data: Any, config: Any = None, **kwargs: Any) -> str:
        """Runnable 接口"""
        if isinstance(input_data, dict):
            kwargs.update(input_data)
        return self.format(**kwargs)

    def __or__(self, other: Any) -> Any:
        """支持 LCEL"""
        return self._template.__or__(other)

    @property
    def input_variables(self) -> List[str]:
        """获取输入变量"""
        return list(self._template.input_variables)


# =============================================================================
# 动态管道提示模板
# =============================================================================


class DynamicPipelinePromptTemplate:
    """
    动态管道提示模板

    运行时根据输入动态决定使用哪些子模板。
    适用于条件化提示场景。

    使用方式：
        ```python
        from prompts import DynamicPipelinePromptTemplate

        template = DynamicPipelinePromptTemplate(
            selectors={
                "formal": "请用正式的语气回答：{question}",
                "casual": "随意点回答：{question}",
            },
            default_key="casual",
        )

        result = template.format(question="你好", tone="formal")
        ```
    """

    def __init__(
        self,
        selectors: dict[str, Any],
        default_key: str = "default",
        context_template: Optional[Any] = None,
        final_template: Optional[Any] = None,
    ):
        """
        初始化动态管道提示模板

        Args:
            selectors: 模板选择器 {名称: 模板}
            default_key: 默认模板键名
            context_template: 上下文模板
            final_template: 最终输出模板
        """
        self._selectors = selectors
        self._default_key = default_key
        self._context_template = context_template
        self._final_template = final_template

    def format(self, **kwargs: Any) -> str:
        """根据输入渲染模板"""
        selector_key = kwargs.pop("tone", self._default_key)
        if selector_key not in self._selectors:
            selector_key = self._default_key

        selected_template = self._selectors[selector_key]
        if hasattr(selected_template, "_template"):
            selected_template = selected_template._template
        if hasattr(selected_template, "format"):
            selected_template = selected_template.format(**kwargs)
        else:
            selected_template = str(selected_template).format(**kwargs)

        parts = []
        if self._context_template:
            ctx = self._context_template
            if hasattr(ctx, "format"):
                ctx = ctx.format(**kwargs)
            else:
                ctx = str(ctx).format(**kwargs)
            parts.append(ctx)

        parts.append(selected_template)

        if self._final_template:
            final = self._final_template
            if hasattr(final, "format"):
                final = final.format(context=parts[0] if parts else "", prompt=parts[1] if len(parts) > 1 else "")
            else:
                final = str(final).format(context=parts[0] if parts else "", prompt=parts[1] if len(parts) > 1 else "")
            return final

        return "\n".join(parts)

    def invoke(self, input_data: Any, config: Any = None, **kwargs: Any) -> str:
        """Runnable 接口"""
        if isinstance(input_data, dict):
            kwargs.update(input_data)
        return self.format(**kwargs)

    def __or__(self, other: Any) -> Any:
        """支持 LCEL"""
        return RunnableLambda(self.invoke).__or__(other)

    @property
    def input_variables(self) -> List[str]:
        """获取所有输入变量"""
        variables = set()
        for template in self._selectors.values():
            if hasattr(template, "input_variables"):
                variables.update(template.input_variables)
        return list(variables)


# =============================================================================
# 工厂函数
# =============================================================================


def create_pipeline_prompt(
    pipeline_prompts: List[tuple[str, str]],
    final_prompt: str,
) -> PipelinePromptTemplate:
    """
    工厂函数：创建管道提示模板

    Args:
        pipeline_prompts: [(变量名, 模板字符串), ...]
        final_prompt: 最终模板字符串

    Returns:
        PipelinePromptTemplate 实例

    Example:
        ```python
        template = create_pipeline_prompt(
            pipeline_prompts=[
                ("intro", "你是{role}。"),
                ("context", "当前主题：{topic}"),
            ],
            final_prompt="{intro}\\n{context}\\n用户问题：{question}",
        )
        ```
    """
    return PipelinePromptTemplate.from_prompts(
        pipeline_prompts=pipeline_prompts,
        final_prompt=final_prompt,
    )


def create_chat_message_prompt(
    messages: List[tuple[str, str]],
) -> ChatMessagePromptTemplate:
    """
    工厂函数：创建聊天消息提示模板

    Args:
        messages: [(角色, 内容), ...]

    Returns:
        ChatMessagePromptTemplate 实例

    Example:
        ```python
        template = create_chat_message_prompt([
            ("system", "你是一个{personality}助手"),
            ("user", "{question}"),
        ])
        ```
    """
    return ChatMessagePromptTemplate.from_messages(messages)


def create_few_shot_chat_prompt(
    examples: List[dict],
    example_template: Any,
    k: int = 4,
    **kwargs: Any,
) -> FewShotChatMessagePromptTemplate:
    """
    工厂函数：创建少样本聊天提示模板

    Args:
        examples: 示例列表
        example_template: 示例模板
        k: 使用示例数量
        **kwargs: 其他参数

    Returns:
        FewShotChatMessagePromptTemplate 实例
    """
    return FewShotChatMessagePromptTemplate.from_examples(
        examples=examples,
        example_prompt=example_template,
        k=k,
        **kwargs,
    )


__all__ = [
    "PipelinePromptTemplate",
    "ChatMessagePromptTemplate",
    "FewShotChatMessagePromptTemplate",
    "DynamicPipelinePromptTemplate",
    "create_pipeline_prompt",
    "create_chat_message_prompt",
    "create_few_shot_chat_prompt",
]
