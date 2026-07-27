# -*- coding: utf-8 -*-
"""
Output Parsers 模块 - LLM 结构化输出解析

提供 LangChain 标准 Output Parser 封装：
|- JsonOutputParser: JSON 字符串解析
|- PydanticOutputParser: Pydantic 模型解析（支持 with_structured_output）
|- StrOutputParser: 字符串解析
|- CommaSeparatedListOutputParser: 逗号分隔列表解析
|- RetryOutputParser: 带重试的健壮解析
|- StructuredOutputParser: 通用结构化输出解析器
|- XmlOutputParser: XML 格式输出解析器
|- DatetimeOutputParser: 日期时间解析器
|- create_json_parser: 快速创建 JSON 解析器
|- create_pydantic_parser: 快速创建 Pydantic 解析器
|- create_structured_parser: 快速创建结构化解析器
|- create_xml_parser: 快速创建 XML 解析器
|- create_datetime_parser: 快速创建日期时间解析器
"""

from .json_parser import JsonOutputParser, create_json_parser
from .pydantic_parser import PydanticOutputParser, create_pydantic_parser
from .list_parser import CommaSeparatedListOutputParser, create_list_parser
from .retry_parser import RetryOutputParser, create_retry_parser
from .structured_parser import (
    StructuredOutputParser,
    XmlOutputParser,
    DatetimeOutputParser,
    create_structured_parser,
    create_xml_parser,
    create_datetime_parser,
)

__all__ = [
    # Json
    "JsonOutputParser",
    "create_json_parser",
    # Pydantic
    "PydanticOutputParser",
    "create_pydantic_parser",
    # List
    "CommaSeparatedListOutputParser",
    "create_list_parser",
    # Retry
    "RetryOutputParser",
    "create_retry_parser",
    # Structured
    "StructuredOutputParser",
    "create_structured_parser",
    # XML
    "XmlOutputParser",
    "create_xml_parser",
    # Datetime
    "DatetimeOutputParser",
    "create_datetime_parser",
]
