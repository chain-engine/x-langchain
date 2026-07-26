# -*- coding: utf-8 -*-
"""
天气工具模块

提供获取天气信息的工具函数，供各个模型文件共同使用。
"""

from typing import Any, Dict

import requests
from langchain.tools import tool
from core.config import settings


def _search_weather_core(city: str) -> str:
    """
    查询指定城市的天气信息的核心逻辑

    Args:
        city: 城市名称

    Returns:
        天气信息字符串
    """
    try:
        # 验证参数
        if not city or not isinstance(city, str):
            return "错误：城市名称不能为空"

        # 集成高德地图天气 API
        api_key: str = settings.AMAP_API_KEY
        if not api_key:
            return "错误：请在 .env 文件中配置 AMAP_API_KEY"

        # 1. 先通过地理编码 API 获取城市的 adcode
        geo_url: str = "https://restapi.amap.com/v3/geocode/geo"
        geo_params: Dict[str, str] = {"key": api_key, "address": city, "output": "json"}

        geo_response: requests.Response = requests.get(geo_url, params=geo_params, timeout=5)
        geo_data: Dict[str, Any] = geo_response.json()

        if geo_data.get("status") != "1" or not geo_data.get("geocodes"):
            return f"错误：无法获取 {city} 的地理位置信息"

        adcode: str = geo_data["geocodes"][0]["adcode"]

        # 2. 使用 adcode 获取天气信息
        weather_url: str = "https://restapi.amap.com/v3/weather/weatherInfo"
        weather_params: Dict[str, str] = {
            "key": api_key,
            "city": adcode,
            "extensions": "base",  # base: 基础天气信息, all: 详细天气信息
            "output": "json",
        }

        weather_response: requests.Response = requests.get(weather_url, params=weather_params, timeout=5)
        weather_data: Dict[str, Any] = weather_response.json()

        if weather_data.get("status") != "1" or not weather_data.get("lives"):
            return f"错误：无法获取 {city} 的天气信息"

        # 3. 处理天气数据
        live_weather: Dict[str, str] = weather_data["lives"][0]
        weather: str = live_weather["weather"]
        temperature: str = live_weather["temperature"]
        winddirection: str = live_weather["winddirection"]
        windpower: str = live_weather["windpower"]
        humidity: str = live_weather["humidity"]
        reporttime: str = live_weather["reporttime"]

        # 4. 构建返回字符串
        return f"{city}的天气：{weather}，气温 {temperature}°C，{winddirection}{windpower}级，湿度 {humidity}%，数据更新时间：{reporttime}"

    except Exception as e:
        # 捕获所有异常，确保工具不会因为错误而崩溃
        from core.logger import logger
        logger.error(f"获取天气信息失败: {e}")
        return f"获取天气信息失败，请稍后重试"


@tool
def weather_search_tool(city: str) -> str:
    """查询指定城市的天气信息

    Args:
        city: 城市名称

    Returns:
        天气信息字符串
    """
    return _search_weather_core(city)


get_weather = weather_search_tool
