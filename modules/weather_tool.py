"""
实时天气查询工具
================
基于 langchain_core.tools.BaseTool 封装，供 Agent 自动调用。

数据源：wttr.in（免费，无需 API Key）
特点：
- 按城市名查当前天气 + 未来 1~3 天预报
- 返回自然语言文本，LLM 可直接总结
- 超时/网络异常自动兜底

使用示例：
    from modules.weather_tool import WeatherTool
    tool = WeatherTool()
    print(tool._run(city="杭州", days=2))
"""

import json
from typing import Optional, Type

import requests
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool


# ============================================================
# Pydantic 输入参数 Schema
# ============================================================
class WeatherInput(BaseModel):
    """天气查询入参。"""
    city: str = Field(
        description="城市名称，中文或英文均可。例如 '北京'、'杭州'、'Shanghai'",
    )
    days: int = Field(
        default=1,
        ge=1,
        le=3,
        description="查询天数，1=今天，2=今明两天，3=今明后三天",
    )


# ============================================================
# 天气查询工具
# ============================================================
class WeatherTool(BaseTool):
    """
    实时天气查询工具。

    使用 wttr.in 免费 API，无需注册 Key。
    支持全球城市，返回天气概况、温度、湿度、风速等。
    """

    name: str = "get_weather"
    description: str = (
        "查询指定城市当前天气和未来1~3天预报。"
        "返回天气状况、最高最低温度、湿度、风向风速等信息。"
        "适合在用户询问'某地天气怎么样''什么时候去某地合适'时使用。"
        "参数：city（城市名称，中文或英文），days（天数，默认1）。"
    )
    args_schema: Type[BaseModel] = WeatherInput

    # ---- 天气状况中文映射 ----
    _WEATHER_MAP: dict = {
        "Clear": "晴",
        "Sunny": "晴",
        "Partly cloudy": "多云",
        "Partly Cloudy": "多云",
        "Cloudy": "阴",
        "Overcast": "阴",
        "Mist": "薄雾",
        "Fog": "雾",
        "Freezing fog": "冻雾",
        "Light drizzle": "小雨",
        "Patchy light drizzle": "零星小雨",
        "Light rain": "小雨",
        "Patchy light rain": "零星小雨",
        "Moderate rain": "中雨",
        "Moderate rain at times": "间歇中雨",
        "Heavy rain": "大雨",
        "Heavy rain at times": "间歇大雨",
        "Light snow": "小雪",
        "Patchy light snow": "零星小雪",
        "Moderate snow": "中雪",
        "Heavy snow": "大雪",
        "Thundery outbreaks possible": "可能有雷暴",
        "Patchy rain possible": "可能有雨",
        "Patchy snow possible": "可能有雪",
    }

    def _translate_weather(self, code: str) -> str:
        """将 wttr 英文天气描述转中文。"""
        return self._WEATHER_MAP.get(code, code)

    def _build_fallback(self, city: str) -> str:
        """网络不通时的兜底输出。"""
        return (
            f"天气查询失败：无法连接到天气服务。\n"
            f"请确认城市名称 '{city}' 是否正确，或稍后重试。"
        )

    # ============================================================
    # 核心：调 wttr.in API 并格式化
    # ============================================================
    def _run(self, city: str, days: int = 1) -> str:
        """
        查询天气并返回格式化中文文本。

        参数：
            city: 城市名称
            days: 天数 (1~3)

        返回：
            格式化天气文本
        """
        city_clean = city.strip()
        url = f"https://wttr.in/{city_clean}?format=j1"

        try:
            resp = requests.get(url, timeout=8)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException:
            return self._build_fallback(city_clean)
        except json.JSONDecodeError:
            return self._build_fallback(city_clean)

        # ---- 解析 ----
        try:
            current = data.get("current_condition", [{}])[0]
            forecasts = data.get("weather", [])[:days]

            lines = [f"【{city_clean} 天气】"]

            # 当前天气
            weather_cn = self._translate_weather(
                current.get("weatherDesc", [{}])[0].get("value", "")
            )
            temp_c = current.get("temp_C", "?")
            humidity = current.get("humidity", "?")
            wind_speed = current.get("windspeedKmph", "?")
            wind_dir = current.get("winddir16Point", "?")

            lines.append(
                f"当前：{weather_cn}，气温 {temp_c}°C，"
                f"湿度 {humidity}%，{wind_dir}风 {wind_speed}km/h"
            )

            # 逐日预报
            weekday_map = {
                "Mon": "周一", "Tue": "周二", "Wed": "周三",
                "Thu": "周四", "Fri": "周五", "Sat": "周六", "Sun": "周日",
            }
            for fc in forecasts:
                date = fc.get("date", "?")
                day_of_week = ""
                for item in fc.get("hourly", []):
                    dow = item.get("weatherDesc", [{}])[0].get("value", "")
                    # 从 wttr 返回的 hourly 中取星期几
                    if dow:
                        dow_cn = weekday_map.get(dow, dow)
                        day_of_week = dow_cn
                        break
                max_t = fc.get("maxtempC", "?")
                min_t = fc.get("mintempC", "?")
                hourly = fc.get("hourly", [])
                day_desc_en = hourly[2].get("weatherDesc", [{}])[0].get("value", "") if len(hourly) > 2 else ""
                day_desc = self._translate_weather(day_desc_en)

                lines.append(
                    f"{date} {day_of_week}：{day_desc}，"
                    f"{min_t}°C ~ {max_t}°C"
                )

            return "\n".join(lines)

        except (KeyError, IndexError, TypeError):
            return self._build_fallback(city_clean)
