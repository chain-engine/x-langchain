# -*- coding: utf-8 -*-
"""
Pydantic 模型输出解析器

将 LLM 返回的文本解析为 Pydantic BaseModel 实例。
支持两种模式：
1. with_structured_output: 直接使用 LLM 的结构化输出能力（推荐）
2. 解析模式: 先让 LLM 输出 JSON，再手动解析
"""

from __future__ import annotations

from typing import Any, Generic, Type, TypeVar, get_type_hints

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser as LCJsonOutputParser
from langchain_core.runnables import Runnable
from pydantic import BaseModel, ValidationError

from core.logger import logger

T = TypeVar("T", bound=BaseModel)


class PydanticOutputParser(Generic[T]):
    """
    Pydantic 模型输出解析器

    提供类型安全的 LLM 输出解析，将文本解析为 Pydantic 实例。
    支持两种工作模式：

    模式一（推荐）：with_structured_output 模式
        - LLM 原生支持结构化输出，准确性最高
        - 通过 bind() 将 schema 注入 LLM
        - 直接解析为 Pydantic 对象

    模式二（兼容）：解析 JSON 模式
        - 先让 LLM 输出 JSON 字符串
        - 再将字符串解析为 Pydantic 实例
        - 适用于不支持结构化输出的模型

    使用方式：
        ```python
        from pydantic import BaseModel

        class UserInfo(BaseModel):
            name: str
            age: int
            email: Optional[str] = None

        parser = PydanticOutputParser(pydantic_model=UserInfo)
        result = parser.invoke('{"name": "张三", "age": 25}')
        # -> UserInfo(name='张三', age=25, email=None)
        ```
    """

    def __init__(
        self,
        pydantic_model: Type[T],
        *,
        encoding: str = "utf-8",
        strict: bool = False,
    ):
        """
        初始化 Pydantic 解析器

        Args:
            pydantic_model: Pydantic 模型类
            encoding: 文本编码，默认 utf-8
            strict: 是否使用严格模式（不忽略额外字段）
        """
        if not issubclass(pydantic_model, BaseModel):
            raise TypeError(f"pydantic_model 必须是 BaseModel 子类，实际: {type(pydantic_model)}")

        self._model: Type[T] = pydantic_model
        self._encoding = encoding
        self._strict = strict
        self._json_parser = LCJsonOutputParser()

    @property
    def model(self) -> Type[T]:
        """获取 Pydantic 模型类"""
        return self._model

    def get_format_instructions(self) -> str:
        """
        获取格式说明（用于注入 LLM prompt）

        返回类似：
        "Please respond in JSON format conforming to the following schema:
        {'properties': {'name': {'title': 'Name', 'type': 'string'}, ...}, ...}"

        Returns:
            格式说明字符串
        """
        schema = self._model.model_json_schema()
        import json

        return (
            "Please respond in JSON format conforming to the following schema:\n"
            f"{json.dumps(schema, indent=2, ensure_ascii=False)}"
        )

    def parse(self, text: str) -> T:
        """
        同步解析文本为 Pydantic 实例

        Args:
            text: LLM 返回的原始文本

        Returns:
            Pydantic 模型实例

        Raises:
            ValidationError: 解析或验证失败
            ValueError: JSON 格式错误
        """
        text = text.strip()
        text = self._strip_code_fence(text)

        try:
            # 先尝试解析为 dict
            raw = self._json_parser.parse(text)

            # 构造 Pydantic 实例
            if self._strict:
                return self._model.model_validate(raw)
            else:
                return self._model.model_validate(raw)

        except ValidationError as e:
            logger.warning(f"Pydantic 验证失败: {e}")
            raise
        except Exception as e:
            logger.warning(f"解析失败: {e}")
            raise ValueError(f"无法解析为 {self._model.__name__}: {e}") from e

    def _strip_code_fence(self, text: str) -> str:
        """去掉 markdown JSON 代码块"""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            if len(lines) >= 2:
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                text = "\n".join(lines).strip()
        return text

    # ------------------------------------------------------------------ #
    # Runnable 接口
    # ------------------------------------------------------------------ #
    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> T:
        """
        Runnable 接口入口

        Args:
            input: LLM 原始输出（str / dict / AIMessage）
            config: Runnable 配置
            **kwargs: 其他参数

        Returns:
            Pydantic 模型实例
        """
        text = self._extract_text(input)
        return self.parse(text)

    def _extract_text(self, input: Any) -> str:
        """从各种输入中提取纯文本"""
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
        """支持 LCEL: parser | next"""
        from langchain_core.runnables import RunnableLambda

        def _parse_wrapper(text: Any) -> T:
            return self.parse(text)

        return RunnableLambda(_parse_wrapper) | other

    def __rrshift__(self, other: Any) -> Any:
        """支持 LCEL: prompt | parser"""
        from langchain_core.runnables import RunnableLambda

        def _parse_wrapper(text: Any) -> T:
            return self.parse(text)

        return other | RunnableLambda(_parse_wrapper)

    # ------------------------------------------------------------------ #
    # LLM 绑定（推荐方式）
    # ------------------------------------------------------------------ #
    def bind_to_model(self, llm: BaseChatModel) -> Runnable:
        """
        将解析器绑定到 LLM，返回支持结构化输出的 Runnable

        这是推荐的使用方式，让 LLM 原生输出符合 schema 的结果。

        Args:
            llm: BaseChatModel 实例

        Returns:
            绑定结构化输出后的 LLM Runnable

        使用示例：
            ```python
            from llms import create_chat_model

            class Weather(BaseModel):
                city: str
                temperature: float
                condition: str

            parser = PydanticOutputParser(pydantic_model=Weather)
            llm = create_chat_model()
            structured_llm = parser.bind_to_model(llm)

            # structured_llm.invoke([("user", "北京今天天气如何？")])
            # -> Weather(city="北京", temperature=25.0, condition="晴")
            ```
        """
        return llm.with_structured_output(self._model)

    def __repr__(self) -> str:
        return f"PydanticOutputParser(model={self._model.__name__})"


def create_pydantic_parser(
    model: Type[T],
    *,
    encoding: str = "utf-8",
    strict: bool = False,
) -> PydanticOutputParser[T]:
    """
    工厂函数：创建 Pydantic 解析器

    Args:
        model: Pydantic 模型类
        encoding: 文本编码
        strict: 是否严格验证

    Returns:
        PydanticOutputParser 实例

    使用示例：
        ```python
        from pydantic import BaseModel

        class Response(BaseModel):
            answer: str
            confidence: float

        parser = create_pydantic_parser(Response)
        result = parser.invoke('{"answer": "太阳从东边升起", "confidence": 0.95}')
        ```
    """
    return PydanticOutputParser(pydantic_model=model, encoding=encoding, strict=strict)


__all__ = ["PydanticOutputParser", "create_pydantic_parser"]
