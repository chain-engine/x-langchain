# -*- coding: utf-8 -*-
"""Token 用量统计器。"""

from __future__ import annotations

from collections import defaultdict
from threading import Lock
from typing import Any


class TokenUsageTracker:
    """按模型提供商聚合 Token 用量的单例统计器。"""

    _instance: "TokenUsageTracker | None" = None
    _instance_lock = Lock()
    _stats: defaultdict[str, dict[str, int]]
    _lock: Lock

    def __new__(cls) -> "TokenUsageTracker":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._stats = defaultdict(cls._empty_stats)
                    cls._instance._lock = Lock()
        return cls._instance

    @staticmethod
    def _empty_stats() -> dict[str, int]:
        return {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def record(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model_name: str,
        provider: str,
    ) -> dict[str, Any]:
        """记录一次模型调用，并返回本次调用信息。"""
        prompt = max(0, int(prompt_tokens or 0))
        completion = max(0, int(completion_tokens or 0))
        total = prompt + completion
        with self._lock:
            item = self._stats[provider.lower()]
            item["prompt_tokens"] += prompt
            item["completion_tokens"] += completion
            item["total_tokens"] += total
        return {
            "model_name": model_name,
            "provider": provider,
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": total,
        }

    def get_total(self) -> dict[str, Any]:
        """返回所有提供商的汇总统计。"""
        with self._lock:
            result = self._empty_stats()
            for item in self._stats.values():
                for key in result:
                    result[key] += item[key]
            return result

    def get_by_provider(self, provider: str) -> dict[str, int]:
        """返回指定提供商的统计。"""
        with self._lock:
            return dict(self._stats[provider.lower()])

    def reset(self) -> None:
        """清空所有统计。"""
        with self._lock:
            self._stats.clear()

    @staticmethod
    def estimate_cost(provider: str, prompt_tokens: int, completion_tokens: int) -> float:
        """按粗略公开价格估算美元成本。"""
        prices = {
            "deepseek": (0.001, 0.001),
            "doubao": (0.001, 0.001),
            "aliyun": (0.002, 0.006),
            "tongyi": (0.002, 0.006),
            "openai": (0.005, 0.015),
            "anthropic": (0.003, 0.015),
        }
        prompt_price, completion_price = prices.get(provider.lower(), (0.001, 0.001))
        return (max(0, prompt_tokens) * prompt_price + max(0, completion_tokens) * completion_price) / 1000


__all__ = ["TokenUsageTracker"]
