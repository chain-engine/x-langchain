# -*- coding: utf-8 -*-
"""
Prompts 模块 - 提示词模板

提供 LangChain 提示模板封装：
|- PromptTemplate: 标准文本提示模板
|- ChatPromptTemplate: 对话型提示模板
|- FewShotPromptTemplate: 少样本提示模板
|- DynamicFewShotPromptTemplate: 动态少样本提示模板
|- PipelinePromptTemplate: 多级管道提示模板
|- ChatMessagePromptTemplate: 消息级别提示模板
|- FewShotChatMessagePromptTemplate: 少样本聊天提示模板
|- DynamicPipelinePromptTemplate: 动态管道提示模板
|- create_tool_prompt: 工具调用提示工厂
|- create_qa_prompt: 问答场景提示工厂
|- create_summarization_prompt: 文本摘要提示工厂
|- create_few_shot_prompt: 少样本提示工厂
|- create_dynamic_few_shot_prompt: 动态少样本提示工厂
|- create_pipeline_prompt: 管道提示工厂
|- create_chat_message_prompt: 聊天消息提示工厂
|- create_few_shot_chat_prompt: 少样本聊天提示工厂
"""

from .templates import (
    PromptTemplate,
    ChatPromptTemplate,
    create_tool_prompt,
    create_qa_prompt,
    create_summarization_prompt,
)
from .few_shot import (
    FewShotPromptTemplate,
    DynamicFewShotPromptTemplate,
    create_few_shot_prompt,
    create_dynamic_few_shot_prompt,
)
from .advanced_templates import (
    PipelinePromptTemplate,
    ChatMessagePromptTemplate,
    FewShotChatMessagePromptTemplate,
    DynamicPipelinePromptTemplate,
    LangChainPromptTemplates,
    create_pipeline_prompt,
    create_chat_message_prompt,
    create_few_shot_chat_prompt,
)

__all__ = [
    # Basic Templates
    "PromptTemplate",
    "ChatPromptTemplate",
    # Few-shot Templates
    "FewShotPromptTemplate",
    "DynamicFewShotPromptTemplate",
    # Advanced Templates
    "PipelinePromptTemplate",
    "ChatMessagePromptTemplate",
    "FewShotChatMessagePromptTemplate",
    "DynamicPipelinePromptTemplate",
    "LangChainPromptTemplates",
    # Factory Functions
    "create_tool_prompt",
    "create_qa_prompt",
    "create_summarization_prompt",
    "create_few_shot_prompt",
    "create_dynamic_few_shot_prompt",
    "create_pipeline_prompt",
    "create_chat_message_prompt",
    "create_few_shot_chat_prompt",
]
