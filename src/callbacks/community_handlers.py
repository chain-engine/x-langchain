# -*- coding: utf-8 -*-
"""
LangChain 标准 Callback Handlers

提供 LangChain 官方和社区的 Callback Handler 封装：
- StdOutCallbackHandler: 标准输出回调
- AimCallbackHandler: AIM 监控回调
- FileCallbackHandler: 文件日志回调
- CustomCallbackHandler: 自定义回调基类
- SensitiveInfoCallbackHandler: 敏感信息过滤回调
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from core.logger import logger


# =============================================================================
# 标准输出回调处理器
# =============================================================================


class StdOutCallbackHandler(BaseCallbackHandler):
    """
    标准输出 Callback Handler

    将 LLM 调用信息打印到标准输出，便于调试和快速查看。
    轻量级实现，适合开发和测试环境。
    """

    def __init__(
        self,
        *,
        color: Optional[str] = None,
        verbose: bool = True,
        include_prompt: bool = False,
    ):
        """
        初始化标准输出处理器

        Args:
            color: ANSI 颜色代码（如 "green", "blue"）
            verbose: 是否输出详细信息
            include_prompt: 是否包含 prompt 内容
        """
        super().__init__()
        self._color = color or ""
        self._verbose = verbose
        self._include_prompt = include_prompt
        self._colors = {
            "red": "\033[91m",
            "green": "\033[92m",
            "yellow": "\033[93m",
            "blue": "\033[94m",
            "magenta": "\033[95m",
            "cyan": "\033[96m",
            "reset": "\033[0m",
        }

    def _colored(self, text: str) -> str:
        """添加颜色"""
        color_code = self._colors.get(self._color, "")
        reset = self._colors["reset"]
        return f"{color_code}{text}{reset}"

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        **kwargs: Any,
    ) -> None:
        """LLM 调用开始"""
        name = serialized.get("name", "Unknown")
        if self._verbose:
            print(self._colored(f"[LLM Start] {name}"))
            if self._include_prompt and prompts:
                for i, prompt in enumerate(prompts):
                    preview = prompt[:200] + "..." if len(prompt) > 200 else prompt
                    print(f"  Prompt {i}: {preview}")

    def on_llm_end(
        self,
        response: LLMResult,
        **kwargs: Any,
    ) -> None:
        """LLM 调用结束"""
        if not self._verbose:
            return

        for generation_list in response.generations:
            for generation in generation_list:
                content = getattr(generation, "text", "") or getattr(generation, "content", "")
                if content:
                    preview = content[:200] + "..." if len(content) > 200 else content
                    print(self._colored(f"[LLM End] {preview}"))

                # Token 使用信息
                if hasattr(generation, "generation_info") and generation.generation_info:
                    info = generation.generation_info
                    if "token_usage" in info:
                        print(f"  Token Usage: {info['token_usage']}")

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        inputs: str,
        **kwargs: Any,
    ) -> None:
        """工具调用开始"""
        name = serialized.get("name", "Unknown")
        if self._verbose:
            print(self._colored(f"[Tool Start] {name}"))
            if self._include_prompt:
                preview = str(inputs)[:200] + "..." if len(str(inputs)) > 200 else str(inputs)
                print(f"  Input: {preview}")

    def on_tool_end(
        self,
        output: str,
        **kwargs: Any,
    ) -> None:
        """工具调用结束"""
        if self._verbose:
            preview = str(output)[:200] + "..." if len(str(output)) > 200 else str(output)
            print(self._colored(f"[Tool End] {preview}"))


# =============================================================================
# Aim 监控回调处理器
# =============================================================================


class AimCallbackHandler(BaseCallbackHandler):
    """
    AIM 监控 Callback Handler

    将 LLM 调用追踪数据发送到 Aim 平台。
    Aim 是一个开源的 ML 模型追踪和可视化工具。

    使用方式：
        ```python
        from langchain.callbacks import get_callback_manager

        # 需要安装 aim: pip install aim
        handler = AimCallbackHandler(repo="./aim_logs")
        llm.invoke(..., callbacks=[handler])
        ```
    """

    def __init__(
        self,
        repo: str = "./aim_repo",
        experiment_name: str = "x-langchain",
    ):
        """
        初始化 AIM 处理器

        Args:
            repo: Aim 仓库路径
            experiment_name: 实验名称
        """
        super().__init__()
        self._repo = repo
        self._experiment_name = experiment_name
        self._aim_run: Optional[Any] = None
        self._setup_aim()

    def _setup_aim(self) -> None:
        """初始化 Aim"""
        try:
            from aim import Run

            self._aim_run = Run(
                repo=self._repo,
                experiment=self._experiment_name,
            )
            logger.info(f"AimCallbackHandler: 已初始化, repo={self._repo}")
        except ImportError:
            logger.warning("AimCallbackHandler: 未安装 aim，请运行 pip install aim")
        except Exception as e:
            logger.warning(f"AimCallbackHandler: 初始化失败: {e}")

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        **kwargs: Any,
    ) -> None:
        """记录 LLM 调用开始"""
        if not self._aim_run:
            return

        try:
            name = serialized.get("name", "Unknown")
            self._aim_run.track(
                {"event": "llm_start", "model": name},
                name="llm_events",
            )
        except Exception as e:
            logger.warning(f"AimCallbackHandler: 追踪失败: {e}")

    def on_llm_end(
        self,
        response: LLMResult,
        **kwargs: Any,
    ) -> None:
        """记录 LLM 调用结束"""
        if not self._aim_run:
            return

        try:
            total_tokens = 0
            for generation_list in response.generations:
                for generation in generation_list:
                    if hasattr(generation, "usage_metadata"):
                        usage = generation.usage_metadata
                        total_tokens = usage.get("total_tokens", 0)

            self._aim_run.track(
                {"event": "llm_end", "total_tokens": total_tokens},
                name="llm_events",
            )
        except Exception as e:
            logger.warning(f"AimCallbackHandler: 追踪失败: {e}")

    def __del__(self) -> None:
        """关闭 Aim Run"""
        if self._aim_run:
            try:
                self._aim_run.close()
            except Exception:
                pass


# =============================================================================
# 文件日志回调处理器
# =============================================================================


class FileCallbackHandler(BaseCallbackHandler):
    """
    文件日志 Callback Handler

    将 LLM 调用记录写入本地 JSON 文件，便于后续分析和调试。
    支持追加模式，不会覆盖历史记录。

    使用方式：
        ```python
        handler = FileCallbackHandler(
            log_file="./logs/llm_calls.jsonl",
        )
        llm.invoke(..., callbacks=[handler])
        ```
    """

    def __init__(
        self,
        log_file: str = "./logs/llm_calls.jsonl",
        auto_mkdir: bool = True,
    ):
        """
        初始化文件日志处理器

        Args:
            log_file: 日志文件路径（支持 .jsonl 格式）
            auto_mkdir: 是否自动创建目录
        """
        super().__init__()
        self._log_file = Path(log_file)
        self._call_stack: list[dict] = []

        if auto_mkdir:
            self._log_file.parent.mkdir(parents=True, exist_ok=True)

        self._current_call: Optional[dict] = None

    def _write_log(self, record: dict) -> None:
        """写入日志"""
        try:
            with open(self._log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"FileCallbackHandler: 写入失败: {e}")

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        **kwargs: Any,
    ) -> None:
        """记录 LLM 调用开始"""
        self._current_call = {
            "event": "llm_start",
            "timestamp": datetime.now().isoformat(),
            "model": serialized.get("name", "Unknown"),
            "prompt_count": len(prompts),
            "prompts": prompts if len(prompts) <= 3 else prompts[:3],
        }

    def on_llm_end(
        self,
        response: LLMResult,
        **kwargs: Any,
    ) -> None:
        """记录 LLM 调用结束"""
        if not self._current_call:
            return

        try:
            self._current_call["event"] = "llm_end"
            self._current_call["end_timestamp"] = datetime.now().isoformat()

            # 提取输出
            outputs = []
            total_tokens = 0
            for generation_list in response.generations:
                for generation in generation_list:
                    text = getattr(generation, "text", "") or getattr(generation, "content", "")
                    outputs.append(text[:500] if text else "")

                    if hasattr(generation, "usage_metadata"):
                        total_tokens = generation.usage_metadata.get("total_tokens", 0)

            self._current_call["outputs"] = outputs
            self._current_call["total_tokens"] = total_tokens

            # 计算耗时
            start = datetime.fromisoformat(self._current_call["timestamp"])
            end = datetime.fromisoformat(self._current_call["end_timestamp"])
            self._current_call["duration_ms"] = (end - start).total_seconds() * 1000

            self._write_log(self._current_call)

        except Exception as e:
            logger.warning(f"FileCallbackHandler: 记录失败: {e}")
        finally:
            self._current_call = None

    def on_llm_error(
        self,
        error: Exception,
        **kwargs: Any,
    ) -> None:
        """记录 LLM 调用错误"""
        if self._current_call:
            self._current_call["event"] = "llm_error"
            self._current_call["error"] = str(error)
            self._write_log(self._current_call)
            self._current_call = None


# =============================================================================
# 敏感信息过滤回调处理器
# =============================================================================


class SensitiveInfoCallbackHandler(BaseCallbackHandler):
    """
    敏感信息过滤 Callback Handler

    自动检测并过滤 LLM 输入/输出中的敏感信息，
    防止敏感数据泄露到日志或监控系统中。

    支持的敏感信息类型：
    - API Key
    - 邮箱地址
    - 手机号
    - 身份证号
    - 信用卡号
    """

    PATTERNS = {
        "api_key": [
            (r"api[_-]?key['\"]?\s*[:=]\s*['\"]?([a-zA-Z0-9_\-]{20,})", r"api_key='***\1'[-]'***"),
            (r"token['\"]?\s*[:=]\s*['\"]?([a-zA-Z0-9_\-\.]{20,})", r"token='***\1'[-]'***"),
        ],
        "email": (r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "[EMAIL_REDACTED]"),
        "phone": (r"1[3-9]\d{9}", "[PHONE_REDACTED]"),
        "id_card": (r"\d{17}[\dXx]", "[ID_REDACTED]"),
        "credit_card": (r"\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}", "[CARD_REDACTED]"),
    }

    def __init__(
        self,
        patterns: Optional[dict[str, Any]] = None,
        replace_with: str = "[REDACTED]",
        verbose: bool = False,
    ):
        """
        初始化敏感信息过滤处理器

        Args:
            patterns: 自定义敏感信息模式
            replace_with: 替换文本
            verbose: 是否输出过滤日志
        """
        super().__init__()
        self._patterns = patterns or self.PATTERNS
        self._replace_with = replace_with
        self._verbose = verbose
        self._stats = {"total_calls": 0, "redacted_count": 0}

    def _filter_text(self, text: str) -> tuple[str, list[str]]:
        """
        过滤敏感信息

        Returns:
            (过滤后的文本, 被过滤的信息类型列表)
        """
        import re

        redacted_types = []
        result = text

        for info_type, pattern_info in self._patterns.items():
            if isinstance(pattern_info, tuple) and len(pattern_info) == 2:
                # 可替换模式
                pattern, replacement = pattern_info
                new_result, count = re.subn(pattern, replacement, result, flags=re.IGNORECASE)
                if count > 0:
                    result = new_result
                    redacted_types.append(info_type)
                    self._stats["redacted_count"] += count
            else:
                # 简单替换模式
        else:
            pattern = pattern_info
            if re.search(pattern, result, re.IGNORECASE):
                result = re.sub(pattern, self._replace_with, result, flags=re.IGNORECASE)
                redacted_types.append(info_type)
                self._stats["redacted_count"] += 1

        return result, redacted_types

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        **kwargs: Any,
    ) -> None:
        """过滤 LLM 输入"""
        self._stats["total_calls"] += 1

        if self._verbose:
            for i, prompt in enumerate(prompts):
                filtered, types = self._filter_text(prompt)
                if types:
                    logger.debug(f"LLM Input [{i}]: 过滤了 {types}")

    def on_llm_end(
        self,
        response: LLMResult,
        **kwargs: Any,
    ) -> None:
        """过滤 LLM 输出"""
        if self._verbose:
            for generation_list in response.generations:
                for generation in generation_list:
                    text = getattr(generation, "text", "") or getattr(generation, "content", "")
                    if text:
                        filtered, types = self._filter_text(text)
                        if types:
                            logger.debug(f"LLM Output: 过滤了 {types}")

    def get_stats(self) -> dict[str, int]:
        """获取统计信息"""
        return dict(self._stats)


# =============================================================================
# 自定义回调基类
# =============================================================================


class CustomCallbackHandler(BaseCallbackHandler):
    """
    自定义 Callback Handler 基类

    提供常用的回调方法和统计功能，
    方便快速创建自定义的 Handler。

    使用方式：
        ```python
        class MyCallbackHandler(CustomCallbackHandler):
            def on_llm_generation_complete(self, prompt, completion, **kwargs):
                print(f"Generated: {completion}")

        handler = MyCallbackHandler()
        ```
    """

    def __init__(self):
        super().__init__()
        self._stats = {
            "llm_calls": 0,
            "tool_calls": 0,
            "chain_calls": 0,
            "errors": 0,
            "total_llm_time_ms": 0.0,
        }
        self._llm_start_times: dict[int, float] = {}

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        **kwargs: Any,
    ) -> None:
        """LLM 调用开始"""
        self._stats["llm_calls"] += 1
        import time

        call_id = id(prompts)
        self._llm_start_times[call_id] = time.perf_counter()

    def on_llm_end(
        self,
        response: LLMResult,
        **kwargs: Any,
    ) -> None:
        """LLM 调用结束"""
        import time

        # 找到对应的 start 时间
        for generation_list in response.generations:
            call_id = id(generation_list)
            if call_id in self._llm_start_times:
                elapsed = (time.perf_counter() - self._llm_start_times.pop(call_id)) * 1000
                self._stats["total_llm_time_ms"] += elapsed
                break

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        inputs: str,
        **kwargs: Any,
    ) -> None:
        """工具调用开始"""
        self._stats["tool_calls"] += 1

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """Chain 调用开始"""
        self._stats["chain_calls"] += 1

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """Chain 调用结束"""
        pass

    def on_llm_error(
        self,
        error: Exception,
        **kwargs: Any,
    ) -> None:
        """LLM 调用错误"""
        self._stats["errors"] += 1

    def get_stats(self) -> dict[str, Any]:
        """获取统计信息"""
        avg_time = (
            self._stats["total_llm_time_ms"] / self._stats["llm_calls"]
            if self._stats["llm_calls"] > 0
            else 0
        )
        return {
            **self._stats,
            "avg_llm_time_ms": round(avg_time, 2),
        }

    def reset_stats(self) -> None:
        """重置统计"""
        self._stats = {
            "llm_calls": 0,
            "tool_calls": 0,
            "chain_calls": 0,
            "errors": 0,
            "total_llm_time_ms": 0.0,
        }
        self._llm_start_times.clear()

    # 子类可重写的方法
    def on_llm_generation_complete(
        self,
        prompt: str,
        completion: str,
        **kwargs: Any,
    ) -> None:
        """LLM 生成完成（子类可重写）"""
        pass

    def on_tool_execution_complete(
        self,
        tool_name: str,
        input_data: str,
        output_data: str,
        **kwargs: Any,
    ) -> None:
        """工具执行完成（子类可重写）"""
        pass


# =============================================================================
# 事件日志回调处理器
# =============================================================================


class EventLogCallbackHandler(BaseCallbackHandler):
    """
    事件日志 Callback Handler

    记录所有回调事件到内存中的列表，
    方便后续分析和调试。

    使用方式：
        ```python
        handler = EventLogCallbackHandler()
        llm.invoke(..., callbacks=[handler])

        # 获取所有事件
        events = handler.get_events()
        print(json.dumps(events, indent=2))
        ```
    """

    def __init__(self, max_events: int = 1000):
        """
        初始化事件日志处理器

        Args:
            max_events: 最大保留事件数
        """
        super().__init__()
        self._events: list[dict] = []
        self._max_events = max_events
        self._current_chain: Optional[str] = None

    def _add_event(self, event: dict) -> None:
        """添加事件"""
        event["timestamp"] = datetime.now().isoformat()
        self._events.append(event)
        if len(self._events) > self._max_events:
            self._events.pop(0)

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        **kwargs: Any,
    ) -> None:
        """记录 LLM 开始"""
        self._add_event({
            "type": "llm_start",
            "name": serialized.get("name", "Unknown"),
        })

    def on_llm_end(
        self,
        response: LLMResult,
        **kwargs: Any,
    ) -> None:
        """记录 LLM 结束"""
        outputs = []
        for generation_list in response.generations:
            for generation in generation_list:
                text = getattr(generation, "text", "") or getattr(generation, "content", "")
                outputs.append(text[:200] if text else "")

        self._add_event({
            "type": "llm_end",
            "outputs": outputs,
        })

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        inputs: str,
        **kwargs: Any,
    ) -> None:
        """记录工具开始"""
        self._add_event({
            "type": "tool_start",
            "name": serialized.get("name", "Unknown"),
            "inputs": str(inputs)[:200],
        })

    def on_tool_end(
        self,
        output: str,
        **kwargs: Any,
    ) -> None:
        """记录工具结束"""
        self._add_event({
            "type": "tool_end",
            "output": str(output)[:200],
        })

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """记录 Chain 开始"""
        name = serialized.get("name", "Unknown")
        self._current_chain = name
        self._add_event({
            "type": "chain_start",
            "name": name,
        })

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """记录 Chain 结束"""
        self._add_event({
            "type": "chain_end",
            "name": self._current_chain,
        })
        self._current_chain = None

    def get_events(self) -> list[dict]:
        """获取所有事件"""
        return list(self._events)

    def clear_events(self) -> None:
        """清空事件"""
        self._events.clear()

    def get_summary(self) -> dict[str, int]:
        """获取事件统计摘要"""
        summary: dict[str, int] = {}
        for event in self._events:
            event_type = event.get("type", "unknown")
            summary[event_type] = summary.get(event_type, 0) + 1
        return summary


__all__ = [
    "StdOutCallbackHandler",
    "AimCallbackHandler",
    "FileCallbackHandler",
    "SensitiveInfoCallbackHandler",
    "CustomCallbackHandler",
    "EventLogCallbackHandler",
]
