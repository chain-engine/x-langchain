# -*- coding: utf-8 -*-
"""
结构化结果提取

从工具执行结果中提取关键字段，存入记忆供后续使用。
"""

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Callable


@dataclass
class FieldExtractor:
    """字段提取器"""
    name: str  # 字段名
    extractor: Callable[[Any], Any]  # 提取函数
    description: str = ""  # 字段描述


@dataclass
class StructuredResult:
    """结构化结果"""
    tool_name: str
    raw_result: Any
    fields: dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "tool_name": self.tool_name,
            "raw_result": str(self.raw_result)[:500] if self.raw_result else None,
            "fields": self.fields,
            "summary": self.summary,
            "metadata": self.metadata,
        }

    def get_field(self, name: str, default: Any = None) -> Any:
        """获取字段值"""
        return self.fields.get(name, default)


class BaseResultExtractor(ABC):
    """结果提取器基类"""

    @abstractmethod
    def extract(self, result: Any) -> StructuredResult:
        """提取结构化结果"""
        pass


class JSONResultExtractor(BaseResultExtractor):
    """JSON 结果提取器"""

    def __init__(self, tool_name: str, field_schemas: list[dict]):
        """
        Args:
            tool_name: 工具名称
            field_schemas: 字段定义列表，格式为 [{"name": "字段名", "path": "JSON路径"}, ...]
        """
        self._tool_name = tool_name
        self._field_schemas = field_schemas

    def extract(self, result: Any) -> StructuredResult:
        data = result
        if isinstance(result, str):
            try:
                data = json.loads(result)
            except json.JSONDecodeError:
                pass

        fields = {}
        for schema in self._field_schemas:
            name = schema["name"]
            path = schema.get("path", "")
            default = schema.get("default")

            if not path:
                fields[name] = data
            else:
                fields[name] = self._extract_by_path(data, path, default)

        return StructuredResult(
            tool_name=self._tool_name,
            raw_result=result,
            fields=fields,
        )

    def _extract_by_path(self, data: Any, path: str, default: Any = None) -> Any:
        """根据路径提取值"""
        if not data:
            return default

        parts = path.split(".")
        current = data

        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            elif isinstance(current, list) and part.isdigit():
                idx = int(part)
                current = current[idx] if idx < len(current) else None
            else:
                return default

            if current is None:
                return default

        return current


class SQLResultExtractor(BaseResultExtractor):
    """SQL 查询结果提取器"""

    def __init__(
        self,
        tool_name: str = "sql_execute",
        extract_numeric: bool = True,
        extract_first_row: bool = True,
    ):
        self._tool_name = tool_name
        self._extract_numeric = extract_numeric
        self._extract_first_row = extract_first_row

    def extract(self, result: Any) -> StructuredResult:
        fields = {"raw_rows": [], "row_count": 0, "columns": []}

        if isinstance(result, str):
            result = self._parse_sql_result(result)

        if isinstance(result, dict):
            # 标准 SQL 结果格式
            fields["row_count"] = result.get("row_count", 0)
            fields["columns"] = result.get("columns", [])
            fields["raw_rows"] = result.get("rows", [])

            # 提取数值字段（用于聚合查询）
            if self._extract_numeric:
                numeric_fields = self._extract_numeric_fields(fields["raw_rows"], fields["columns"])
                fields.update(numeric_fields)

            # 提取第一行（用于详情查询）
            if self._extract_first_row and fields["raw_rows"]:
                fields["first_row"] = dict(zip(fields["columns"], fields["raw_rows"][0]))

            # 生成摘要
            fields["summary"] = self._generate_summary(fields)

        return StructuredResult(
            tool_name=self._tool_name,
            raw_result=result,
            fields=fields,
        )

    def _parse_sql_result(self, result: str) -> dict:
        """解析 SQL 结果字符串"""
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            # 尝试解析为行格式
            lines = result.strip().split("\n")
            if not lines:
                return {}

            # 尝试解析为 "key: value" 格式
            data = {}
            for line in lines:
                if ":" in line:
                    key, value = line.split(":", 1)
                    data[key.strip()] = value.strip()
            return data

    def _extract_numeric_fields(self, rows: list, columns: list) -> dict:
        """提取数值字段"""
        numeric_fields = {}
        if not rows or not columns:
            return numeric_fields

        for i, col in enumerate(columns):
            col_lower = col.lower()
            if any(kw in col_lower for kw in ["amount", "total", "price", "count", "num", "sum"]):
                values = [row[i] for row in rows if i < len(row) and row[i] is not None]
                try:
                    numeric_values = [float(v) for v in values]
                    if numeric_values:
                        numeric_fields[col] = {
                            "sum": sum(numeric_values),
                            "avg": sum(numeric_values) / len(numeric_values),
                            "min": min(numeric_values),
                            "max": max(numeric_values),
                            "count": len(numeric_values),
                        }
                except (ValueError, TypeError):
                    pass

        return numeric_fields

    def _generate_summary(self, fields: dict) -> str:
        """生成结果摘要"""
        row_count = fields.get("row_count", 0)
        columns = fields.get("columns", [])

        summary = f"查询返回 {row_count} 条记录"
        if columns:
            summary += f"，字段: {', '.join(columns[:5])}"
            if len(columns) > 5:
                summary += f" 等 {len(columns)} 个字段"

        return summary


class TextResultExtractor(BaseResultExtractor):
    """文本结果提取器 - 使用正则表达式"""

    def __init__(
        self,
        tool_name: str,
        patterns: list[dict[str, str]],
    ):
        """
        Args:
            tool_name: 工具名称
            patterns: 正则模式列表，格式为 [{"name": "字段名", "pattern": "正则", "type": "int|float|str"}, ...]
        """
        self._tool_name = tool_name
        self._patterns = patterns

    def extract(self, result: Any) -> StructuredResult:
        text = str(result) if result else ""
        fields = {}

        for p in self._patterns:
            name = p["name"]
            pattern = p["pattern"]
            result_type = p.get("type", "str")

            match = re.search(pattern, text)
            if match:
                value = match.group(1) if match.groups() else match.group(0)
                if result_type == "int":
                    value = int(value.replace(",", ""))
                elif result_type == "float":
                    value = float(value.replace(",", ""))
                fields[name] = value

        return StructuredResult(
            tool_name=self._tool_name,
            raw_result=result,
            fields=fields,
        )


class WeatherResultExtractor(BaseResultExtractor):
    """天气结果提取器"""

    def extract(self, result: Any) -> StructuredResult:
        fields = {}

        if isinstance(result, str):
            result = self._parse_weather_result(result)

        if isinstance(result, dict):
            fields["location"] = result.get("location", result.get("city", ""))
            fields["temperature"] = result.get("temperature", result.get("temp"))
            fields["weather"] = result.get("weather", result.get("condition", ""))
            fields["humidity"] = result.get("humidity", result.get("rh"))
            fields["wind"] = result.get("wind", "")
            fields["suggestion"] = result.get("suggestion", result.get("suggest", ""))

        # 生成穿衣建议
        temp = fields.get("temperature")
        if temp:
            try:
                temp_val = float(temp) if isinstance(temp, (int, float, str)) else 0
                if temp_val < 10:
                    fields["clothing_advice"] = "建议穿羽绒服或厚外套"
                elif temp_val < 20:
                    fields["clothing_advice"] = "建议穿外套或毛衣"
                elif temp_val < 30:
                    fields["clothing_advice"] = "建议穿轻薄衣物"
                else:
                    fields["clothing_advice"] = "建议穿短袖或清凉衣物"
            except (ValueError, TypeError):
                pass

        return StructuredResult(
            tool_name="weather",
            raw_result=result,
            fields=fields,
        )

    def _parse_weather_result(self, result: str) -> dict:
        """解析天气结果"""
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            # 尝试从文本提取
            data = {}
            patterns = [
                (r"城市[：:]\s*(\S+)", "location"),
                (r"温度[：:]\s*(\d+)", "temperature"),
                (r"天气[：:]\s*(\S+)", "weather"),
                (r"湿度[：:]\s*(\d+)", "humidity"),
            ]
            for pattern, key in patterns:
                match = re.search(pattern, result)
                if match:
                    data[key] = match.group(1)
            return data


# 预定义提取器注册表
EXTRACTOR_REGISTRY: dict[str, BaseResultExtractor] = {}


def register_extractor(name: str, extractor: BaseResultExtractor) -> None:
    """注册提取器"""
    EXTRACTOR_REGISTRY[name] = extractor


def get_extractor(name: str) -> Optional[BaseResultExtractor]:
    """获取提取器"""
    return EXTRACTOR_REGISTRY.get(name)


def extract_structured_result(tool_name: str, result: Any) -> StructuredResult:
    """
    便捷函数：根据工具名提取结构化结果

    Args:
        tool_name: 工具名称
        result: 工具执行结果

    Returns:
        结构化结果
    """
    # 查找注册的提取器
    extractor = get_extractor(tool_name)
    if extractor:
        return extractor.extract(result)

    # 使用默认提取器
    if tool_name in ("sql_execute", "execute_sql", "text_to_sql_execute"):
        extractor = SQLResultExtractor(tool_name)
    elif tool_name in ("weather", "weather_search"):
        extractor = WeatherResultExtractor()
    else:
        # 通用 JSON 提取器
        return StructuredResult(
            tool_name=tool_name,
            raw_result=result,
            fields={"raw": str(result)[:200]},
        )

    return extractor.extract(result)


__all__ = [
    "FieldExtractor",
    "StructuredResult",
    "BaseResultExtractor",
    "JSONResultExtractor",
    "SQLResultExtractor",
    "TextResultExtractor",
    "WeatherResultExtractor",
    "register_extractor",
    "get_extractor",
    "extract_structured_result",
]
