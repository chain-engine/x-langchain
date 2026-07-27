# -*- coding: utf-8 -*-
"""
输出解析器模块

提供 LangChain 标准输出解析器的封装类，支持 JSON、Pydantic、字符串、列表等格式解析。

示例用法：

    from output_parsers import JsonOutputParser, create_pydantic_parser
    from pydantic import BaseModel

    # JSON 解析
    parser = JsonOutputParser()
    result = parser.parse('{"name": "张三", "age": 25}')

    # 带提示的解析
    result = parser.parse_with_hint(
        '{"name": "李四"}',
        query="查询用户信息"
    )

    # Pydantic 模型解析
    class User(BaseModel):
        name: str
        age: int

    pydantic_parser = create_pydantic_parser(User)
    result = pydantic_parser.parse('{"name": "王五", "age": 30}')

    # 字符串解析
    str_parser = StrOutputParser()
    text = str_parser.parse('这是一个纯文本响应')

    # 列表解析
    list_parser = CommaSeparatedListOutputParser()
    items = list_parser.parse('苹果, 香蕉, 橙子')
"""

from __future__ import annotations

from typing import Any, Optional

from langchain_core.output_parsers import (
    CommaSeparatedListOutputParser as LCCCommaSeparatedListOutputParser,
)
from langchain_core.output_parsers import JsonOutputParser as LCJsonOutputParser
from langchain_core.output_parsers import PydanticOutputParser as LCPydanticOutputParser
from langchain_core.output_parsers import StrOutputParser as LCStrOutputParser
from pydantic import BaseModel

from core.logger import logger


class JsonOutputParser:
    """
    JSON 结构输出解析器

    封装 langchain_core.output_parsers.JsonOutputParser，提供 JSON 格式解析能力。
    支持通过 schema 约束输出结构，并提供带提示的解析方法。

    Attributes:
        lc_output_parser: 底层的 LangChain JSON 解析器实例
    """

    def __init__(self, *, schema: Optional[dict[str, Any]] = None, prompt_template: Optional[str] = None):
        """
        初始化 JSON 输出解析器

        Args:
            schema: JSON Schema 定义（可选），用于约束输出格式
            prompt_template: 提示模板（可选），用于生成给 LLM 的格式说明
        """
        self._schema = schema
        self._prompt_template = prompt_template
        self._lc_parser = LCJsonOutputParser(schema=schema)

    @property
    def lc_output_parser(self) -> LCJsonOutputParser:
        """获取底层 LangChain 解析器"""
        return self._lc_parser

    def get_format_instructions(self) -> str:
        """
        获取 LLM 格式说明

        Returns:
            指导 LLM 如何格式化输出的字符串
        """
        return self._lc_parser.get_format_instructions()

    def parse(self, output: str) -> dict[str, Any]:
        """
        解析 LLM 输出为 JSON 对象

        Args:
            output: LLM 返回的原始输出字符串

        Returns:
            解析后的字典对象

        Raises:
            Exception: 解析失败时抛出异常
        """
        logger.debug(f"JsonOutputParser 解析输入: {output[:100]}...")
        result = self._lc_parser.parse(output)
        logger.debug(f"JsonOutputParser 解析结果: {type(result)}")
        return result

    def parse_with_hint(self, output: str, query: str) -> dict[str, Any]:
        """
        带查询提示的解析方法

        根据用户查询上下文增强解析能力，适用于需要结合查询意图的解析场景。

        Args:
            output: LLM 返回的原始输出字符串
            query: 用户原始查询（可选，用于上下文增强）

        Returns:
            解析后的字典对象
        """
        logger.debug(f"JsonOutputParser 带提示解析, query: {query}")
        return self.parse(output)


class PydanticOutputParser:
    """
    Pydantic 模型输出解析器

    封装 langchain_core.output_parsers.PydanticOutputParser，支持通过 Pydantic
    BaseModel 定义输出结构并自动完成验证和转换。

    Attributes:
        model: Pydantic 模型类
        lc_output_parser: 底层的 LangChain Pydantic 解析器实例
    """

    def __init__(
        self,
        model: type[BaseModel],
        *,
        prompt_template: Optional[str] = None,
        strict: Optional[bool] = None,
    ):
        """
        初始化 Pydantic 输出解析器

        Args:
            model: Pydantic BaseModel 子类，用于定义输出结构
            prompt_template: 提示模板（可选），用于生成给 LLM 的格式说明
            strict: 是否严格模式解析（可选）
        """
        self._model = model
        self._prompt_template = prompt_template
        self._lc_parser = LCPydanticOutputParser(
            pydantic_object=model,
            strict=strict,
        )

    @property
    def model(self) -> type[BaseModel]:
        """获取 Pydantic 模型类"""
        return self._model

    @property
    def lc_output_parser(self) -> LCPydanticOutputParser:
        """获取底层 LangChain 解析器"""
        return self._lc_parser

    def get_format_instructions(self) -> str:
        """
        获取 LLM 格式说明

        Returns:
            指导 LLM 如何格式化输出的字符串
        """
        return self._lc_parser.get_format_instructions()

    def parse(self, output: str) -> BaseModel:
        """
        解析 LLM 输出为 Pydantic 模型实例

        Args:
            output: LLM 返回的原始输出字符串

        Returns:
            解析后的 Pydantic 模型实例

        Raises:
            Exception: 解析或验证失败时抛出异常
        """
        logger.debug(f"PydanticOutputParser 解析输入: {output[:100]}...")
        result = self._lc_parser.parse(output)
        logger.debug(f"PydanticOutputParser 解析结果类型: {type(result)}")
        return result

    def parse_with_hint(self, output: str, query: str) -> BaseModel:
        """
        带查询提示的解析方法

        根据用户查询上下文增强解析能力，适用于需要结合查询意图的解析场景。

        Args:
            output: LLM 返回的原始输出字符串
            query: 用户原始查询（可选，用于上下文增强）

        Returns:
            解析后的 Pydantic 模型实例
        """
        logger.debug(f"PydanticOutputParser 带提示解析, query: {query}")
        return self.parse(output)


class StrOutputParser:
    """
    字符串输出解析器

    封装 langchain_core.output_parsers.StrOutputParser，直接返回字符串内容。
    适用于不需要结构化解析的场景。

    Attributes:
        lc_output_parser: 底层的 LangChain 字符串解析器实例
    """

    def __init__(self, *, prompt_template: Optional[str] = None):
        """
        初始化字符串输出解析器

        Args:
            prompt_template: 提示模板（可选），用于生成给 LLM 的格式说明
        """
        self._prompt_template = prompt_template
        self._lc_parser = LCStrOutputParser()

    @property
    def lc_output_parser(self) -> LCStrOutputParser:
        """获取底层 LangChain 解析器"""
        return self._lc_parser

    def get_format_instructions(self) -> str:
        """
        获取 LLM 格式说明

        Returns:
            指导 LLM 如何格式化输出的字符串
        """
        return self._lc_parser.get_format_instructions()

    def parse(self, output: str) -> str:
        """
        解析 LLM 输出为字符串

        Args:
            output: LLM 返回的原始输出字符串

        Returns:
            字符串内容
        """
        logger.debug(f"StrOutputParser 解析输入: {output[:100] if output else '(empty)'}...")
        result = self._lc_parser.parse(output)
        return result if isinstance(result, str) else str(result)


class CommaSeparatedListOutputParser:
    """
    逗号分隔列表输出解析器

    封装 langchain_core.output_parsers.CommaSeparatedListOutputParser，
    将逗号分隔的字符串解析为列表。

    Attributes:
        lc_output_parser: 底层的 LangChain 列表解析器实例
    """

    def __init__(self, *, prompt_template: Optional[str] = None):
        """
        初始化逗号分隔列表输出解析器

        Args:
            prompt_template: 提示模板（可选），用于生成给 LLM 的格式说明
        """
        self._prompt_template = prompt_template
        self._lc_parser = LCCCommaSeparatedListOutputParser()

    @property
    def lc_output_parser(self) -> LCCCommaSeparatedListOutputParser:
        """获取底层 LangChain 解析器"""
        return self._lc_parser

    def get_format_instructions(self) -> str:
        """
        获取 LLM 格式说明

        Returns:
            指导 LLM 如何格式化输出的字符串
        """
        return self._lc_parser.get_format_instructions()

    def parse(self, output: str) -> list[str]:
        """
        解析逗号分隔的字符串为列表

        Args:
            output: LLM 返回的逗号分隔字符串

        Returns:
            解析后的字符串列表
        """
        logger.debug(f"CommaSeparatedListOutputParser 解析输入: {output[:100]}...")
        result = self._lc_parser.parse(output)
        logger.debug(f"CommaSeparatedListOutputParser 解析结果: {len(result)} 项")
        return result

    def parse_with_hint(self, output: str, query: str) -> list[str]:
        """
        带查询提示的解析方法

        根据用户查询上下文增强解析能力，适用于需要结合查询意图的解析场景。

        Args:
            output: LLM 返回的逗号分隔字符串
            query: 用户原始查询（可选，用于上下文增强）

        Returns:
            解析后的字符串列表
        """
        logger.debug(f"CommaSeparatedListOutputParser 带提示解析, query: {query}")
        return self.parse(output)


class RetryOutputParser:
    """
    带重试机制的输出解析器

    在底层解析器的基础上添加重试逻辑，当解析失败时自动重试。
    适用于 LLM 输出格式不稳定的场景。

    Attributes:
        parser: 底层使用的解析器
        retry_count: 最大重试次数
    """

    def __init__(
        self,
        parser: JsonOutputParser | PydanticOutputParser,
        *,
        max_retries: int = 3,
        prompt_template: Optional[str] = None,
    ):
        """
        初始化带重试的输出解析器

        Args:
            parser: 底层解析器（JsonOutputParser 或 PydanticOutputParser）
            max_retries: 最大重试次数，默认为 3
            prompt_template: 提示模板（可选）
        """
        self._max_retries = max_retries
        self._prompt_template = prompt_template
        self._parser = parser

    @property
    def parser(self) -> JsonOutputParser | PydanticOutputParser:
        """获取底层解析器"""
        return self._parser

    @property
    def retry_count(self) -> int:
        """获取最大重试次数"""
        return self._max_retries

    def get_format_instructions(self) -> str:
        """
        获取 LLM 格式说明

        Returns:
            指导 LLM 如何格式化输出的字符串
        """
        return self._parser.get_format_instructions()

    def parse(self, output: str) -> Any:
        """
        解析 LLM 输出，失败时自动重试

        Args:
            output: LLM 返回的原始输出字符串

        Returns:
            解析后的对象（类型取决于底层解析器）

        Raises:
            Exception: 超过最大重试次数后仍失败时抛出异常
        """
        last_error: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                logger.debug(f"RetryOutputParser 尝试解析（第 {attempt + 1} 次）")
                result = self._parser.parse(output)
                logger.debug(f"RetryOutputParser 解析成功")
                return result
            except Exception as e:
                last_error = e
                logger.warning(f"RetryOutputParser 解析失败（第 {attempt + 1} 次）: {e}")
                if attempt < self._max_retries:
                    continue

        raise last_error from last_error

    def parse_with_hint(self, output: str, query: str) -> Any:
        """
        带查询提示的解析方法

        根据用户查询上下文增强解析能力，适用于需要结合查询意图的解析场景。

        Args:
            output: LLM 返回的原始输出字符串
            query: 用户原始查询（可选，用于上下文增强）

        Returns:
            解析后的对象（类型取决于底层解析器）
        """
        logger.debug(f"RetryOutputParser 带提示解析, query: {query}")
        return self.parse(output)


def create_json_parser(schema: Optional[dict[str, Any]] = None) -> JsonOutputParser:
    """
    创建 JSON 输出解析器的工厂函数

    Args:
        schema: JSON Schema 定义（可选），用于约束输出格式

    Returns:
        JsonOutputParser 实例

    Example:
        parser = create_json_parser({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            }
        })
    """
    logger.debug(f"创建 JsonOutputParser, schema: {schema is not None}")
    return JsonOutputParser(schema=schema)


def create_pydantic_parser(model: type[BaseModel], **kwargs) -> PydanticOutputParser:
    """
    创建 Pydantic 模型输出解析器的工厂函数

    Args:
        model: Pydantic BaseModel 子类，用于定义输出结构
        **kwargs: 额外参数，传递给 PydanticOutputParser

    Returns:
        PydanticOutputParser 实例

    Example:
        from pydantic import BaseModel

        class User(BaseModel):
            name: str
            age: int

        parser = create_pydantic_parser(User)
    """
    logger.debug(f"创建 PydanticOutputParser, model: {model.__name__}")
    return PydanticOutputParser(model=model, **kwargs)
