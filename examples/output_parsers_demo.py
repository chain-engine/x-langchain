# -*- coding: utf-8 -*-
"""
Output Parsers 示例

演示如何使用 output_parsers 模块进行 LLM 结构化输出。
"""

from pydantic import BaseModel
from llms import create_chat_model
from output_parsers import (
    JsonOutputParser,
    PydanticOutputParser,
    CommaSeparatedListOutputParser,
)


class WeatherResponse(BaseModel):
    city: str
    temperature: float
    condition: str
    humidity: int = 0


def demo_json_parser():
    """JSON 解析器"""
    parser = JsonOutputParser()
    text = '{"name": "张三", "age": 30, "city": "北京"}'
    result = parser.parse(text)
    print(f"[JsonOutputParser] 输入: {text}")
    print(f"[JsonOutputParser] 输出: {result}")
    print()


def demo_pydantic_parser():
    """Pydantic 解析器"""
    parser = PydanticOutputParser(pydantic_model=WeatherResponse)
    print(f"[PydanticOutputParser] 格式说明:\n{parser.get_format_instructions()}\n")

    # 模拟 LLM 输出
    llm_text = '{"city": "上海", "temperature": 25.5, "condition": "晴朗", "humidity": 60}'
    result = parser.parse(llm_text)
    print(f"[PydanticOutputParser] 输入: {llm_text}")
    print(f"[PydanticOutputParser] 输出: {result}")
    print(f"[PydanticOutputParser] 类型: {type(result).__name__}")
    print()


def demo_list_parser():
    """列表解析器"""
    parser = CommaSeparatedListOutputParser(min_items=1, max_items=5)
    texts = [
        "苹果，香蕉，橙子",
        "北京；上海；广州；深圳",
        "第一项\n第二项\n第三项",
    ]
    for text in texts:
        result = parser.parse(text)
        print(f"[ListParser] 输入: {text}")
        print(f"[ListParser] 输出: {result}")
    print()


def demo_lcel_composition():
    """LCEL 组合"""
    llm = create_chat_model("mock")
    parser = PydanticOutputParser(pydantic_model=WeatherResponse)
    structured_llm = parser.bind_to_model(llm)
    print(f"[LCEL] 绑定后的类型: {type(structured_llm).__name__}")


if __name__ == "__main__":
    print("=" * 60)
    print("Output Parsers 示例")
    print("=" * 60)

    demo_json_parser()
    demo_pydantic_parser()
    demo_list_parser()
    demo_lcel_composition()
