# -*- coding: utf-8 -*-
"""兼容 LangGraph 的 Agent 定义。"""

from langchain.agents import create_agent

from core.config import settings
from tools import get_all_tools


def get_default_model():
    """创建默认模型；配置不完整时自动回退到 mock。"""
    from models import create_chat_model

    provider = settings.MODEL_NAME
    if not settings.validate_model_config(provider):
        provider = "mock"
    return create_chat_model(provider)


SYSTEM_PROMPT = """你是一个可以使用工具的智能助手。

当用户需要实时信息或外部数据时，优先调用工具。使用工具后，请清晰总结工具结果，
不要编造事实。遇到数据库问题时，请遵循 TextToSQL 流程：改写问题、查看表结构、
生成 SQL、校验 SQL、执行 SQL，然后用自然语言解释结果。
"""

TOOLS = get_all_tools()

agent = create_agent(
    model=get_default_model(),
    tools=TOOLS,
    system_prompt=SYSTEM_PROMPT,
    debug=False,
)


if __name__ == "__main__":
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "你能做什么？"}]}
    )
    print(result)
