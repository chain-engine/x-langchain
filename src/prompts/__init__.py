# -*- coding: utf-8 -*-
"""
Prompts 模块 - 提示词模板

提供 LangChain 标准提示词模板封装：
- FewShotPromptTemplate: 少样本提示模板
- DynamicFewShotPromptTemplate: 动态少样本模板
- PromptTemplate: 标准提示模板
- ChatPromptTemplate: 对话提示模板
- create_tool_prompt: 为工具调用创建提示模板
- create_qa_prompt: 为问答创建提示模板
"""

from .few_shot import (
    FewShotPromptTemplate,
    DynamicFewShotPromptTemplate,
    create_few_shot_prompt,
    create_dynamic_few_shot_prompt,
)
from .templates import (
    PromptTemplate,
    ChatPromptTemplate,
    create_tool_prompt,
    create_qa_prompt,
    create_summarization_prompt,
)

__all__ = [
    "FewShotPromptTemplate",
    "DynamicFewShotPromptTemplate",
    "create_few_shot_prompt",
    "create_dynamic_few_shot_prompt",
    "PromptTemplate",
    "ChatPromptTemplate",
    "create_tool_prompt",
    "create_qa_prompt",
    "create_summarization_prompt",
]