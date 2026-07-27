# -*- coding: utf-8 -*-
"""带对话记忆的 LCEL 链封装。"""

from typing import Any

from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough

from memories import BufferMemory


_DEFAULT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个有帮助的 AI 助手。"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)


class ConversationChain:
    """将用户输入、对话历史和 LLM 组合为连续对话链。"""

    def __init__(
        self,
        llm: BaseLanguageModel | Runnable,
        prompt_template: Any = None,
        memory: Any = None,
    ) -> None:
        """初始化对话链；memory 需实现记忆变量加载与上下文保存接口。"""
        self.llm = llm
        self.prompt = prompt_template or _DEFAULT_PROMPT
        self.memory = memory or BufferMemory()
        if not hasattr(self.memory, "load_memory_variables") or not hasattr(
            self.memory, "save_context"
        ):
            raise TypeError("memory 必须实现 load_memory_variables 和 save_context")

        load_history = RunnableLambda(
            lambda _: self.memory.load_memory_variables({}).get("history", [])
        )
        self._chain: Runnable = (
            {"history": load_history, "input": RunnablePassthrough()}
            | self.prompt
            | llm
            | StrOutputParser()
        )

    def invoke(self, user_input: str, **kwargs: Any) -> str:
        """调用对话链并保存本轮上下文。"""
        result = self._chain.invoke(user_input, config=kwargs.pop("config", None))
        self.memory.save_context({"input": user_input}, {"output": result})
        return result

    async def ainvoke(self, user_input: str, **kwargs: Any) -> str:
        """异步调用对话链并保存本轮上下文。"""
        result = await self._chain.ainvoke(user_input, config=kwargs.pop("config", None))
        self.memory.save_context({"input": user_input}, {"output": result})
        return result

    def stream(self, user_input: str, **kwargs: Any):
        """流式调用对话链，并在流结束后保存本轮上下文。"""
        chunks = []
        for chunk in self._chain.stream(user_input, config=kwargs.pop("config", None)):
            chunks.append(chunk)
            yield chunk
        self.memory.save_context(
            {"input": user_input}, {"output": "".join(map(str, chunks))}
        )


__all__ = ["ConversationChain"]
