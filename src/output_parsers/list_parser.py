# -*- coding: utf-8 -*-
"""
列表输出解析器

将 LLM 返回的逗号分隔字符串解析为 Python list[str]。
"""

from __future__ import annotations

from typing import Any, List

from langchain_core.output_parsers import CommaSeparatedListOutputParser as LCListParser
from langchain_core.runnables import Runnable, RunnableLambda

from core.logger import logger


class CommaSeparatedListOutputParser:
    """
    逗号分隔列表解析器

    将 LLM 返回的逗号分隔字符串解析为 Python 列表。
    例如："苹果, 香蕉, 橙子" -> ["苹果", "香蕉", "橙子"]

    特性：
    - 自动处理多种分隔符（逗号、顿号、分号、换行）
    - 自动去除每项首尾空白
    - 提供 Runnable 接口

    使用方式：
        ```python
        parser = CommaSeparatedListOutputParser()
        result = parser.invoke("苹果, 香蕉, 橙子")
        # -> ["苹果", "香蕉", "橙子"]
        ```
    """

    # 支持的分隔符
    _separators: tuple[str, ...] = ("，", ",", "、", ";", "；", "\n")

    def __init__(
        self,
        *,
        min_items: int = 0,
        max_items: int | None = None,
        strip: bool = True,
        dedup: bool = False,
    ):
        """
        初始化列表解析器

        Args:
            min_items: 最少项目数，不满足则抛出异常
            max_items: 最多项目数，超出则截断
            strip: 是否去除每项首尾空白
            dedup: 是否去重
        """
        self._min_items = min_items
        self._max_items = max_items
        self._strip = strip
        self._dedup = dedup
        self._parser = LCListParser()

    def parse(self, text: str) -> List[str]:
        """
        解析文本为字符串列表

        Args:
            text: LLM 返回的原始文本

        Returns:
            解析后的字符串列表

        Raises:
            ValueError: 项目数不满足限制
        """
        text = text.strip()

        # 尝试使用 langchain 解析器
        try:
            result: List[str] = self._parser.parse(text)
        except Exception:
            # 回退：手动解析
            result = self._split手动(text)

        # 处理
        if self._strip:
            result = [item.strip() for item in result]

        if self._dedup:
            seen = set()
            deduped = []
            for item in result:
                if item and item not in seen:
                    seen.add(item)
                    deduped.append(item)
            result = deduped

        # 过滤空项
        result = [item for item in result if item]

        # 截断
        if self._max_items is not None and len(result) > self._max_items:
            result = result[: self._max_items]

        # 验证
        if len(result) < self._min_items:
            raise ValueError(
                f"解析结果项目数不足：期望至少 {self._min_items} 项，实际 {len(result)} 项"
            )

        return result

    def _split手动(self, text: str) -> List[str]:
        """手动分割文本"""
        for sep in self._separators:
            if sep in text:
                return [s.strip() for s in text.split(sep) if s.strip()]
        # 无分隔符，整体作为一个元素
        return [text.strip()] if text.strip() else []

    # ------------------------------------------------------------------ #
    # Runnable 接口
    # ------------------------------------------------------------------ #
    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> List[str]:
        """
        Runnable 接口入口
        """
        text = self._extract_text(input)
        return self.parse(text)

    def _extract_text(self, input: Any) -> str:
        if isinstance(input, str):
            return input
        if isinstance(input, dict):
            for key in ("text", "content", "output"):
                if key in input:
                    return str(input[key])
            return str(input)
        if hasattr(input, "content"):
            return str(input.content)
        return str(input)

    def __or__(self, other: Any) -> Any:
        def _parse_wrapper(text: Any) -> List[str]:
            return self.parse(text)

        return RunnableLambda(_parse_wrapper) | other

    def __rrshift__(self, other: Any) -> Any:
        def _parse_wrapper(text: Any) -> List[str]:
            return self.parse(text)

        return other | RunnableLambda(_parse_wrapper)

    def get_format_instructions(self) -> str:
        """获取格式说明"""
        return (
            "Your response should be a list of comma-separated values, "
            "e.g.: `foo, bar, baz`"
        )


def create_list_parser(
    *,
    min_items: int = 0,
    max_items: int | None = None,
    strip: bool = True,
    dedup: bool = False,
) -> CommaSeparatedListOutputParser:
    """
    工厂函数：创建列表解析器

    Args:
        min_items: 最少项目数
        max_items: 最多项目数
        strip: 是否去除空白
        dedup: 是否去重

    Returns:
        CommaSeparatedListOutputParser 实例
    """
    return CommaSeparatedListOutputParser(
        min_items=min_items,
        max_items=max_items,
        strip=strip,
        dedup=dedup,
    )


__all__ = ["CommaSeparatedListOutputParser", "create_list_parser"]
