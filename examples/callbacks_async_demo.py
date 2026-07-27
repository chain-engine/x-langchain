# -*- coding: utf-8 -*-
"""
Callbacks & Async 示例

演示 Callbacks 可观测性和异步接口。
"""

import time
from llms import create_chat_model
from callbacks import (
    TokenCountCallbackHandler,
    TimingCallbackHandler,
    MultiCallbackHandler,
    CallbackConfig,
    get_default_callbacks,
)


def demo_token_counter():
    """Token 计数"""
    print("=" * 60)
    print("[Callbacks] Token 计数器")

    handler = TokenCountCallbackHandler()
    llm = create_chat_model("mock")

    print("[调用 1]")
    llm.invoke([{"role": "user", "content": "你好"}], callbacks=[handler])
    print(f"  Token 统计: {handler.get_summary()}")

    print("[调用 2]")
    llm.invoke([{"role": "user", "content": "今天天气如何"}], callbacks=[handler])
    print(f"  Token 统计: {handler.get_summary()}")


def demo_timing():
    """耗时统计"""
    print("\n" + "=" * 60)
    print("[Callbacks] 耗时统计")

    handler = TimingCallbackHandler()
    llm = create_chat_model("mock")

    print("[执行 LLM 调用]")
    start = time.perf_counter()
    llm.invoke([{"role": "user", "content": "测试"}], callbacks=[handler])
    elapsed = time.perf_counter() - start

    stats = handler.get_summary()
    print(f"  实际耗时: {elapsed*1000:.2f}ms")
    print(f"  Callback 统计: {stats}")


def demo_multi_callbacks():
    """多处理器组合"""
    print("\n" + "=" * 60)
    print("[Callbacks] MultiCallbackHandler")

    multi = MultiCallbackHandler()
    multi.add_handler(TokenCountCallbackHandler())
    multi.add_handler(TimingCallbackHandler())

    llm = create_chat_model("mock")

    llm.invoke(
        [{"role": "user", "content": "你好"}],
        callbacks=multi.get_handlers(),
    )

    token_handler = multi.get_handler(TokenCountCallbackHandler)
    timing_handler = multi.get_handler(TimingCallbackHandler)

    if token_handler:
        print(f"[Token] {token_handler.get_summary()}")
    if timing_handler:
        print(f"[Timing] {timing_handler.get_summary()}")


def demo_async_invoke():
    """异步调用"""
    print("\n" + "=" * 60)
    print("[Async] 异步调用")

    from langchain_core.language_models import BaseChatModel

    # 注意：实际异步需要 LLM 支持 ainvoke
    # 这里演示模式，实际运行时需确认 LLM 支持异步
    llm = create_chat_model("mock")

    print(f"[LLM 类型] {type(llm).__name__}")
    print(f"[支持 ainvoke] {hasattr(llm, 'ainvoke')}")

    # 模拟异步调用
    import asyncio

    async def run_async():
        try:
            result = await llm.ainvoke([{"role": "user", "content": "你好"}])
            print(f"[异步结果] {getattr(result, 'content', str(result)[:50])}")
        except Exception as e:
            print(f"[异步调用] 当前 LLM 不支持 ainvoke: {type(e).__name__}")

    asyncio.run(run_async())


if __name__ == "__main__":
    demo_token_counter()
    demo_timing()
    demo_multi_callbacks()
    demo_async_invoke()
