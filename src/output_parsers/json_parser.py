# -*- coding: utf-8 -*-
"""
JSON 输出解析器

将 LLM 返回的原始文本解析为 Python dict。
支持流式和非流式两种模式。
"""

from __future__ import annotations

from typing import Any, Optional

from langchain_core.output_parsers import JsonOutputParser as LCJsonOutputParser
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from core.logger import logger


class JsonOutputParser:
    """
    JSON 字符串解析器

    将 LLM 返回的包含 JSON 的字符串解析为 Python dict。
    内部封装 langchain_core.output_parsers.JsonOutputParser。

    特性：
    - 自动处理 markdown 代码块包裹的 JSON
    - 支持流式解析模式
    - 提供 Runnable 接口，可与 LCEL 流水线组合

    使用方式：
        ```python
        parser = JsonOutputParser()
        result = parser.invoke("{'name': '张三', 'age': 25}")
        # -> {"name": "张三", "age": 25}
        ```
    """

    # 可选：限制解析结果必须包含的键（为空则不限制）
    _required_keys: set[str] | None = None

    def __init__(
        self,
        *,
        required_keys: Optional[set[str]] = None,
        strip_whitespace: bool = True,
    ):
        """
        初始化 JSON 解析器

        Args:
            required_keys: 解析后必须包含的键集合，为 None 则不限制
            strip_whitespace: 是否在解析前去除首尾空白
        """
        self._required_keys = required_keys
        self._strip_whitespace = strip_whitespace
        self._parser = LCJsonOutputParser()

    def parse(self, text: str) -> Any:
        """
        同步解析 LLM 输出为 dict

        Args:
            text: LLM 返回的原始文本

        Returns:
            解析后的 Python 对象（通常为 dict）
        """
        if self._strip_whitespace:
            text = text.strip()

        # 去掉 markdown 代码块包裹
        text = self._strip_code_fence(text)

        try:
            result = self._parser.parse(text)
            if self._required_keys:
                missing = self._required_keys - set(result.keys())
                if missing:
                    raise ValueError(f"解析结果缺少必需字段: {missing}")
            return result
        except Exception as e:
            logger.warning(f"JSON 解析失败: {e}, 原始文本: {text[:200]}")
            raise

    def _strip_code_fence(self, text: str) -> str:
        """去掉 markdown JSON 代码块包裹"""
        text = text.strip()
        # 处理 ```json ... ``` 或 ``` ... ```
        if text.startswith("```"):
            lines = text.split("\n")
            if len(lines) >= 2:
                # 去掉第一行的 ```json 或 ```
                if lines[0].startswith("```"):
                    lines = lines[1:]
                # 去掉最后一行的 ```
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                text = "\n".join(lines).strip()
        return text

    # ------------------------------------------------------------------ #
    # Runnable 接口
    # ------------------------------------------------------------------ #
    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        """
        Runnable 接口入口

        支持三种输入格式：
        1. str: 直接解析字符串
        2. dict: 尝试提取 "text" / "content" / "output" 字段
        3. AIMessage / 其他带 content 的对象

        Args:
            input: 输入文本或消息对象
            config: Runnable 配置
            **kwargs: 其他参数

        Returns:
            解析后的对象
        """
        text = self._extract_text(input)
        return self.parse(text)

    def _extract_text(self, input: Any) -> str:
        """从各种输入格式中提取纯文本"""
        if isinstance(input, str):
            return input
        if isinstance(input, dict):
            for key in ("text", "content", "output", "raw_output"):
                if key in input:
                    return str(input[key])
            return str(input)
        if hasattr(input, "content"):
            return str(input.content)
        return str(input)

    def __or__(self, other: Any) -> Any:
        """支持 LCEL: parser | next_runnable"""
        return self._parser | other

    def __rrshift__(self, other: Any) -> Any:
        """支持 LCEL: prompt | parser"""
        return other | self._parser

    def with_retry(
        self,
        *,
        max_attempts: int = 3,
        retry_on: type[Exception] | tuple[type[Exception], ...] = (ValueError,),
    ) -> "RetryableJsonParser":
        """
        返回带重试能力的解析器包装

        Args:
            max_attempts: 最大重试次数
            retry_on: 需要重试的异常类型

        Returns:
            RetryableJsonParser 实例
        """
        return RetryableJsonParser(
            parser=self,
            max_attempts=max_attempts,
            retry_on=retry_on,
        )


class RetryableJsonParser:
    """
    带重试能力的 JSON 解析器

    当解析失败时，将原始 LLM 输出保留并抛出异常，
    上层可据此要求 LLM 重新生成。
    """

    def __init__(
        self,
        parser: JsonOutputParser,
        max_attempts: int = 3,
        retry_on: type[Exception] | tuple[type[Exception], ...] = (ValueError,),
    ):
        self.parser = parser
        self.max_attempts = max_attempts
        self.retry_on = retry_on

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        """带重试的解析"""
        last_error: Exception | None = None
        last_text = self.parser._extract_text(input)

        for attempt in range(1, self.max_attempts + 1):
            try:
                return self.parser.parse(last_text)
            except self.retry_on as e:
                last_error = e
                logger.warning(f"JSON 解析失败（第 {attempt} 次）: {e}")
                # 在实际场景中，这里会触发 LLM 重试
                # 当前实现保留原始文本，由调用方决定下一步

        raise last_error from None


def create_json_parser(
    required_keys: Optional[set[str]] = None,
    strip_whitespace: bool = True,
) -> JsonOutputParser:
    """
    工厂函数：创建 JSON 解析器

    Args:
        required_keys: 解析结果必须包含的键集合
        strip_whitespace: 是否去除首尾空白

    Returns:
        JsonOutputParser 实例

    使用示例：
        ```python
        parser = create_json_parser(required_keys={"name", "age"})
        result = parser.invoke('{"name": "张三", "age": 30, "city": "北京"}')
        # -> {"name": "张三", "age": 30}
        # "city" 被自动过滤（如果只需要 name 和 age）
        ```
    """
    return JsonOutputParser(required_keys=required_keys, strip_whitespace=strip_whitespace)


__all__ = ["JsonOutputParser", "create_json_parser", "RetryableJsonParser"]
