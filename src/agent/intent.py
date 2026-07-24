# -*- coding: utf-8 -*-
"""
意图识别模块

识别用户输入的意图，进行参数标准化。
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Callable


class Intent(str, Enum):
    """意图类型"""
    UNKNOWN = "unknown"
    GREETING = "greeting"
    QUERY = "query"  # 一般查询
    DATABASE_QUERY = "database_query"  # 数据库查询
    WEATHER_QUERY = "weather_query"  # 天气查询
    CALENDAR_QUERY = "calendar_query"  # 日历/日程查询
    CALCULATION = "calculation"  # 计算
    TEXT_GENERATION = "text_generation"  # 文本生成
    TRANSLATION = "translation"  # 翻译
    SUMMARY = "summary"  # 摘要
    ANALYSIS = "analysis"  # 分析
    ACTION = "action"  # 执行动作
    HELP = "help"  # 求助


@dataclass
class IntentResult:
    """意图识别结果"""
    intent: Intent
    confidence: float = 0.0  # 置信度 0-1
    entities: dict[str, Any] = field(default_factory=dict)  # 提取的实体
    normalized_input: str = ""  # 标准化后的输入
    suggested_tools: list[str] = field(default_factory=list)  # 建议使用的工具
    metadata: dict = field(default_factory=dict)


class IntentRecognizer:
    """
    意图识别器

    基于规则的意图识别，支持：
    - 关键词匹配
    - 正则表达式提取
    - 参数标准化
    """

    # 意图定义：关键词、正则模式、推荐工具
    INTENT_PATTERNS = {
        Intent.GREETING: {
            "keywords": ["你好", "hi", "hello", "嗨", "早上好", "晚上好", "hi", "hey"],
            "patterns": [],
            "tools": [],
        },
        Intent.WEATHER_QUERY: {
            "keywords": ["天气", "weather", "气温", "温度", "下雨", "晴", "多云", "雨"],
            "patterns": [
                (r"(.+?)天气", "location"),
                (r"(\w+)的天气", "location"),
            ],
            "tools": ["weather_search", "weather"],
        },
        Intent.CALENDAR_QUERY: {
            "keywords": ["日程", "calendar", "会议", "schedule", "安排", "今天", "明天", "这周"],
            "patterns": [
                (r"今天(.+?)安排", "date"),
                (r"明天的(.+?)", "date"),
            ],
            "tools": ["calendar_tool", "get_schedule"],
        },
        Intent.DATABASE_QUERY: {
            "keywords": ["数据库", "db", "表", "查询", "sql", "订单", "用户", "统计", "金额", "总额", "数量"],
            "patterns": [
                (r"统计(.+?)总额", "query_type"),
                (r"查询(.+?)订单", "query_type"),
            ],
            "tools": ["text_to_sql"],
        },
        Intent.CALCULATION: {
            "keywords": ["计算", "等于", "加起来", "求和", "+", "-", "*", "/", "乘", "除"],
            "patterns": [
                (r"(\d+)\s*[\+\-]\s*(\d+)", "expression"),
            ],
            "tools": [],
        },
        Intent.TRANSLATION: {
            "keywords": ["翻译", "translate", "成英文", "成中文", "翻译成"],
            "patterns": [
                (r"把(.+?)翻译成(.+)", "text_and_target"),
            ],
            "tools": ["translate"],
        },
        Intent.SUMMARY: {
            "keywords": ["总结", "摘要", "概括", "总结一下", "要点"],
            "patterns": [
                (r"总结(.+)", "content"),
            ],
            "tools": [],
        },
        Intent.HELP: {
            "keywords": ["帮助", "help", "怎么用", "如何使用", "能做什么", "功能"],
            "patterns": [],
            "tools": [],
        },
        Intent.QUERY: {
            "keywords": ["什么", "怎么", "如何", "为什么", "who", "what", "how", "why"],
            "patterns": [],
            "tools": [],
        },
    }

    def __init__(self, custom_patterns: Optional[dict[Intent, dict]] = None):
        """
        初始化意图识别器

        Args:
            custom_patterns: 自定义意图模式
        """
        self._patterns = {**self.INTENT_PATTERNS}
        if custom_patterns:
            self._patterns.update(custom_patterns)

    def recognize(self, user_input: str) -> IntentResult:
        """
        识别用户意图

        Args:
            user_input: 用户输入

        Returns:
            意图识别结果
        """
        user_input = user_input.strip()
        best_match = IntentResult(
            intent=Intent.UNKNOWN,
            confidence=0.0,
            normalized_input=user_input,
        )

        # 预处理
        normalized = self._normalize_input(user_input)
        best_match.normalized_input = normalized

        for intent, patterns in self._patterns.items():
            score = self._calculate_score(normalized, patterns)
            if score > best_match.confidence:
                best_match.intent = intent
                best_match.confidence = score

        # 提取实体
        best_match.entities = self._extract_entities(normalized, best_match.intent)

        # 获取建议工具
        if best_match.intent in self._patterns:
            best_match.suggested_tools = self._patterns[best_match.intent].get("tools", [])

        return best_match

    def _normalize_input(self, text: str) -> str:
        """标准化输入"""
        # 转为小写
        text = text.lower()
        # 去除多余空格
        text = re.sub(r"\s+", " ", text)
        # 去除标点
        text = re.sub(r"[^\w\s\u4e00-\u9fff]", "", text)
        return text

    def _calculate_score(self, text: str, patterns: dict) -> float:
        """计算匹配分数"""
        score = 0.0

        # 关键词匹配
        keywords = patterns.get("keywords", [])
        for kw in keywords:
            if kw.lower() in text:
                score += 0.4

        # 正则匹配
        regex_patterns = patterns.get("patterns", [])
        for _, _ in regex_patterns:
            score += 0.2

        # 归一化
        return min(score, 1.0)

    def _extract_entities(self, text: str, intent: Intent) -> dict[str, Any]:
        """提取实体"""
        entities = {}

        if intent not in self._patterns:
            return entities

        patterns = self._patterns[intent].get("patterns", [])

        for pattern, entity_name in patterns:
            match = re.search(pattern, text)
            if match:
                if len(match.groups()) == 1:
                    entities[entity_name] = match.group(1)
                else:
                    entities[entity_name] = match.groups()

        # 特定意图的实体提取
        if intent == Intent.WEATHER_QUERY:
            # 提取地点
            location_match = re.search(r"(.+?)(?:天气|的天气)", text)
            if location_match:
                entities["location"] = location_match.group(1).strip()

        elif intent == Intent.CALCULATION:
            # 提取数学表达式
            expr_match = re.search(r"(\d+\.?\d*)\s*([+\-*/])\s*(\d+\.?\d*)", text)
            if expr_match:
                entities["left"] = float(expr_match.group(1))
                entities["operator"] = expr_match.group(2)
                entities["right"] = float(expr_match.group(3))

        elif intent == Intent.DATABASE_QUERY:
            # 提取时间范围
            time_match = re.search(r"(上月|本月|本周|今年|去年|\d+年\d+月)", text)
            if time_match:
                entities["time_range"] = time_match.group(1)

            # 提取统计类型
            if "总额" in text or "总金额" in text:
                entities["aggregation"] = "sum"
            elif "平均" in text:
                entities["aggregation"] = "avg"
            elif "数量" in text or "多少" in text:
                entities["aggregation"] = "count"

        return entities


def parse_parameters(intent_result: IntentResult, user_input: str) -> dict[str, Any]:
    """
    根据意图标准化参数

    Args:
        intent_result: 意图识别结果
        user_input: 原始输入

    Returns:
        标准化后的参数字典
    """
    params = intent_result.entities.copy()

    # 添加原始输入
    params["_raw_input"] = user_input

    # 根据意图添加默认值
    intent = intent_result.intent

    if intent == Intent.WEATHER_QUERY:
        params.setdefault("location", params.get("location", "北京"))

    elif intent == Intent.DATABASE_QUERY:
        params.setdefault("time_range", "本月")
        params.setdefault("aggregation", "sum")

    return params


__all__ = [
    "Intent",
    "IntentResult",
    "IntentRecognizer",
    "parse_parameters",
]
