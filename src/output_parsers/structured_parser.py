# -*- coding: utf-8 -*-
"""
通用结构化输出解析器

提供 LangChain 中常用的结构化输出解析器封装：
- StructuredOutputParser: 通用结构化输出解析器
- XmlOutputParser: XML 格式输出解析器
- DatetimeOutputParser: 日期时间解析器
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Optional, TypeVar

from langchain_core.output_parsers import (
    DatetimeOutputParser as LCDatetimeOutputParser,
    XmlOutputParser as LCXmlOutputParser,
)
from pydantic import BaseModel

from core.logger import logger


T = TypeVar("T", bound=BaseModel)


# =============================================================================
# 通用结构化输出解析器
# =============================================================================


class StructuredOutputParser:
    """
    通用结构化输出解析器

    基于 Pydantic 模型和提示模板，生成格式说明并解析 LLM 输出。
    支持动态生成 JSON Schema，提供清晰的格式指导。

    使用方式：
        ```python
        from output_parsers import StructuredOutputParser

        parser = StructuredOutputParser.from_response_format(
            response_format={
                "name": "用户姓名",
                "age": "用户年龄（整数）",
                "city": "用户所在城市"
            }
        )

        # 在 prompt 中使用格式说明
        prompt = f"""请从文本中提取信息:
        {parser.get_format_instructions()}

        文本: 张三今年25岁，住在上海
        """

        result = parser.parse(llm_output)
        # {'name': '张三', 'age': 25, 'city': '上海'}
        ```
    """

    def __init__(
        self,
        response_format: dict[str, str],
        *,
        strict: bool = False,
        enforce_order: bool = False,
    ):
        """
        初始化结构化输出解析器

        Args:
            response_format: 响应格式定义，key 为字段名，value 为字段描述
            strict: 是否严格解析（解析失败时抛出异常）
            enforce_order: 是否强制按字段顺序解析
        """
        self._response_format = response_format
        self._strict = strict
        self._enforce_order = enforce_order
        self._format_template = self._build_format_template()

    @classmethod
    def from_response_format(
        cls,
        response_format: dict[str, str],
        **kwargs: Any,
    ) -> "StructuredOutputParser":
        """
        从响应格式定义创建解析器

        Args:
            response_format: 字段名 -> 字段描述 的字典
            **kwargs: 透传给构造函数

        Returns:
            StructuredOutputParser 实例
        """
        return cls(response_format=response_format, **kwargs)

    @classmethod
    def from_pydantic(
        cls,
        model: type[T],
        **kwargs: Any,
    ) -> "StructuredOutputParser":
        """
        从 Pydantic 模型创建解析器

        从 Pydantic BaseModel 的字段和注释自动推断格式定义。

        Args:
            model: Pydantic BaseModel 子类
            **kwargs: 透传给构造函数

        Returns:
            StructuredOutputParser 实例
        """
        import inspect

        response_format: dict[str, str] = {}
        for field_name, field_info in model.model_fields.items():
            description = field_info.description or field_name
            # 添加类型信息
            type_hint = field_info.annotation
            if type_hint is not None:
                type_str = self._type_to_str(type_hint)
                if type_str:
                    description = f"{description}（{type_str}）"
            response_format[field_name] = description

        parser = cls(response_format=response_format, **kwargs)
        parser._pydantic_model = model
        return parser

    @staticmethod
    def _type_to_str(annotation: Any) -> str:
        """将类型注解转换为可读字符串"""
        type_map = {
            str: "字符串",
            int: "整数",
            float: "浮点数",
            bool: "布尔值",
            list: "列表",
            dict: "字典",
        }

        origin = getattr(annotation, "__origin__", annotation)
        if origin in type_map:
            return type_map[origin]

        # 处理 Optional
        args = getattr(annotation, "__args__", [])
        if len(args) == 2 and type(None) in args:
            inner = next(a for a in args if a is not type(None))
            inner_str = self._type_to_str(inner)
            if inner_str:
                return f"可选的{inner_str}，可为空"

        return ""

    def _build_format_template(self) -> str:
        """构建格式说明模板"""
        lines = ["请按以下 JSON 格式输出：", "```json", "{"]
        for i, (field, desc) in enumerate(self._response_format.items()):
            comma = "," if i < len(self._response_format) - 1 else ""
            lines.append(f'  "{field}": "<{desc}>"{comma}')
        lines.extend(["}", "```"])
        return "\n".join(lines)

    def get_format_instructions(self) -> str:
        """
        获取格式说明

        Returns:
            指导 LLM 如何格式化输出的字符串
        """
        return self._format_template

    def get_json_schema(self) -> dict[str, Any]:
        """
        获取 JSON Schema

        Returns:
            JSON Schema 字典
        """
        properties = {}
        required = []

        for field, desc in self._response_format.items():
            properties[field] = {
                "type": "string",
                "description": desc,
            }
            required.append(field)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    def parse(self, output: str) -> dict[str, Any]:
        """
        解析 LLM 输出为结构化字典

        Args:
            output: LLM 返回的原始输出字符串

        Returns:
            解析后的字典

        Raises:
            ValueError: 解析失败时抛出
        """
        logger.debug(f"StructuredOutputParser 解析输入: {output[:100]}...")

        # 尝试提取 JSON
        json_str = self._extract_json(output)

        if not json_str:
            if self._strict:
                raise ValueError("无法从输出中提取 JSON")
            return {}

        # 解析 JSON
        try:
            import json

            result = json.loads(json_str)
            logger.debug(f"StructuredOutputParser 解析成功: {list(result.keys())}")
            return result
        except json.JSONDecodeError as e:
            if self._strict:
                raise ValueError(f"JSON 解析失败: {e}") from e
            # 尝试智能修复
            return self._smart_parse(json_str)

    def _extract_json(self, text: str) -> Optional[str]:
        """从文本中提取 JSON 字符串"""
        # 优先匹配 ```json ... ``` 块
        json_block = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if json_block:
            return json_block.group(1).strip()

        # 尝试匹配 ``` ... ``` 块
        code_block = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
        if code_block:
            content = code_block.group(1).strip()
            if content.startswith("{") or content.startswith("["):
                return content

        # 尝试直接匹配 JSON 对象
        json_obj = re.search(r"\{[\s\S]*\}", text)
        if json_obj:
            return json_obj.group(0)

        return None

    def _smart_parse(self, json_str: str) -> dict[str, Any]:
        """
        智能解析：处理不完整或有瑕疵的 JSON

        Args:
            json_str: 部分 JSON 字符串

        Returns:
            尽可能解析的字典
        """
        import json

        result: dict[str, Any] = {}

        # 尝试逐行解析
        for line in json_str.split("\n"):
            line = line.strip()
            if not line or line == "{" or line == "}" or line == "}," or line == "}":
                continue

            # 匹配 "field": "value" 或 "field": value
            match = re.match(r'^"([^"]+)":\s*(.+?)\s*[,}]?\s*$', line)
            if match:
                key = match.group(1)
                value_str = match.group(2).rstrip(",").strip()

                # 尝试解析值
                if value_str.startswith('"') and value_str.endswith('"'):
                    result[key] = value_str[1:-1]
                elif value_str == "null":
                    result[key] = None
                elif value_str == "true":
                    result[key] = True
                elif value_str == "false":
                    result[key] = False
                elif value_str.isdigit():
                    result[key] = int(value_str)
                elif self._is_float(value_str):
                    result[key] = float(value_str)

        return result

    @staticmethod
    def _is_float(s: str) -> bool:
        """判断字符串是否可转为浮点数"""
        try:
            float(s)
            return "." in s
        except ValueError:
            return False

    def parse_with_validation(
        self,
        output: str,
        model: type[T],
    ) -> T:
        """
        解析并验证为 Pydantic 模型

        Args:
            output: LLM 输出
            model: Pydantic 模型类

        Returns:
            验证后的模型实例

        Raises:
            ValidationError: 验证失败时抛出
        """
        data = self.parse(output)
        return model.model_validate(data)


# =============================================================================
# XML 输出解析器
# =============================================================================


class XmlOutputParser:
    """
    XML 格式输出解析器

    封装 langchain_core.output_parsers.XmlOutputParser，
    解析 LLM 返回的 XML 格式内容。

    使用方式：
        ```python
        from output_parsers import XmlOutputParser

        parser = XmlOutputParser()

        output = '''<root>
            <name>张三</name>
            <age>25</age>
        </root>'''

        result = parser.parse(output)
        # {'name': '张三', 'age': '25'}
        ```
    """

    def __init__(
        self,
        *,
        tags: Optional[list[str]] = None,
        decoder: Optional[callable] = None,
    ):
        """
        初始化 XML 解析器

        Args:
            tags: 期望的顶级标签列表（用于验证）
            decoder: 自定义解码函数，接收标签名和内容，返回解析后的值
        """
        self._tags = tags
        self._decoder = decoder
        self._lc_parser = LCXmlOutputParser()

    @property
    def lc_output_parser(self) -> LCXmlOutputParser:
        """获取底层 LangChain 解析器"""
        return self._lc_parser

    def get_format_instructions(self) -> str:
        """
        获取格式说明

        Returns:
            指导 LLM 输出 XML 格式的字符串
        """
        return self._lc_parser.get_format_instructions()

    def parse(self, output: str) -> dict[str, Any]:
        """
        解析 XML 输出

        Args:
            output: LLM 返回的 XML 字符串

        Returns:
            解析后的字典
        """
        logger.debug(f"XmlOutputParser 解析输入: {output[:100]}...")

        try:
            result = self._lc_parser.parse(output)
            logger.debug(f"XmlOutputParser 解析成功: {result}")
            return self._post_process(result)
        except Exception as e:
            logger.warning(f"XmlOutputParser 解析失败: {e}")
            # 尝试手动解析
            return self._manual_parse(output)

    def _post_process(self, result: Any) -> dict[str, Any]:
        """后处理解析结果"""
        if isinstance(result, dict):
            for key, value in result.items():
                if self._decoder:
                    result[key] = self._decoder(key, value)
        return result

    def _manual_parse(self, xml_str: str) -> dict[str, Any]:
        """
        手动解析 XML（当 LangChain 解析器失败时）

        Args:
            xml_str: XML 字符串

        Returns:
            解析后的字典
        """
        result: dict[str, Any] = {}

        # 去除 XML 声明和注释
        xml_str = re.sub(r"<\?[^>]+\?>", "", xml_str)
        xml_str = re.sub(r"<!--.*?-->", "", xml_str, flags=re.DOTALL)

        # 匹配标签内容
        pattern = r"<(\w+)>([^<]*)</\1>"
        for match in re.finditer(pattern, xml_str):
            tag = match.group(1)
            content = match.group(2).strip()

            # 处理嵌套标签（转为字符串）
            if "<" in content:
                content = self._parse_nested(content)

            # 类型转换
            content = self._convert_type(content)

            if tag not in result:
                result[tag] = content
            elif isinstance(result[tag], list):
                result[tag].append(content)
            else:
                result[tag] = [result[tag], content]

        return result

    def _parse_nested(self, content: str) -> str:
        """解析嵌套标签为字符串表示"""
        parts = []
        for match in re.finditer(r"<(\w+)>([^<]*)</\1>", content):
            parts.append(f"{match.group(1)}: {match.group(2).strip()}")
        return "; ".join(parts) if parts else content

    @staticmethod
    def _convert_type(value: str) -> Any:
        """尝试转换值类型"""
        value = value.strip()

        # 布尔值
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False

        # 数字
        if value.isdigit():
            return int(value)
        if re.match(r"^-?\d+\.?\d*$", value):
            return float(value)

        # 列表格式 [a, b, c]
        if value.startswith("[") and value.endswith("]"):
            inner = value[1:-1]
            items = [item.strip().strip('"').strip("'") for item in inner.split(",")]
            return [XmlOutputParser._convert_type(item) for item in items if item]

        return value


# =============================================================================
# 日期时间输出解析器
# =============================================================================


class DatetimeOutputParser:
    """
    日期时间输出解析器

    封装 langchain_core.output_parsers.DatetimeOutputParser，
    解析 LLM 返回的日期时间字符串。

    使用方式：
        ```python
        from output_parsers import DatetimeOutputParser

        parser = DatetimeOutputParser()

        result = parser.parse("2024-01-15T10:30:00")
        print(result)  # datetime.datetime(2024, 1, 15, 10, 30, 0)
        ```
    """

    def __init__(
        self,
        *,
        format: str = "%Y-%m-%dT%H:%M:%S",
        expected_format: Optional[str] = None,
    ):
        """
        初始化日期时间解析器

        Args:
            format: datetime.strftime 格式字符串
            expected_format: 期望的人类可读格式说明
        """
        self._format = format
        self._expected_format = expected_format
        self._lc_parser = LCDatetimeOutputParser(format=format)

    @property
    def lc_output_parser(self) -> LCDatetimeOutputParser:
        """获取底层 LangChain 解析器"""
        return self._lc_parser

    def get_format_instructions(self) -> str:
        """
        获取格式说明

        Returns:
            指导 LLM 输出日期时间格式的字符串
        """
        instructions = self._lc_parser.get_format_instructions()
        if self._expected_format:
            instructions += f"\n期望格式: {self._expected_format}"
        return instructions

    def parse(self, output: str) -> datetime:
        """
        解析日期时间字符串

        Args:
            output: LLM 返回的日期时间字符串

        Returns:
            datetime 对象

        Raises:
            ValueError: 解析失败时抛出
        """
        logger.debug(f"DatetimeOutputParser 解析输入: {output}")

        try:
            result = self._lc_parser.parse(output)
            logger.debug(f"DatetimeOutputParser 解析成功: {result}")
            return result
        except Exception as e:
            # 尝试多种常见格式
            return self._try_multiple_formats(output)

    def _try_multiple_formats(self, output: str) -> datetime:
        """尝试多种常见日期时间格式"""
        formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y/%m/%d %H:%M:%S",
            "%Y-%m-%d",
            "%Y/%m/%d",
            "%Y年%m月%d日 %H时%M分%S秒",
            "%Y年%m月%d日",
            "%d/%m/%Y %H:%M:%S",
            "%m/%d/%Y %H:%M:%S",
        ]

        for fmt in formats:
            try:
                return datetime.strptime(output.strip(), fmt)
            except ValueError:
                continue

        raise ValueError(f"无法解析日期时间字符串: {output}")


# =============================================================================
# 工厂函数
# =============================================================================


def create_structured_parser(
    response_format: Optional[dict[str, str]] = None,
    **kwargs: Any,
) -> StructuredOutputParser:
    """
    工厂函数：创建通用结构化输出解析器

    Args:
        response_format: 响应格式定义
        **kwargs: 透传给构造函数

    Returns:
        StructuredOutputParser 实例
    """
    if not response_format:
        response_format = {}
    return StructuredOutputParser(response_format=response_format, **kwargs)


def create_xml_parser(
    tags: Optional[list[str]] = None,
    **kwargs: Any,
) -> XmlOutputParser:
    """
    工厂函数：创建 XML 输出解析器

    Args:
        tags: 期望的顶级标签
        **kwargs: 透传给构造函数

    Returns:
        XmlOutputParser 实例
    """
    return XmlOutputParser(tags=tags, **kwargs)


def create_datetime_parser(
    format: str = "%Y-%m-%dT%H:%M:%S",
    **kwargs: Any,
) -> DatetimeOutputParser:
    """
    工厂函数：创建日期时间输出解析器

    Args:
        format: 日期时间格式
        **kwargs: 透传给构造函数

    Returns:
        DatetimeOutputParser 实例
    """
    return DatetimeOutputParser(format=format, **kwargs)


__all__ = [
    "StructuredOutputParser",
    "XmlOutputParser",
    "DatetimeOutputParser",
    "create_structured_parser",
    "create_xml_parser",
    "create_datetime_parser",
]
