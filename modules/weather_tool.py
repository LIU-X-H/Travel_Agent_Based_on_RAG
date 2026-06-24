"""
实时天气查询工具
================
数据源: Open-Meteo（免费，无需 API Key，全球覆盖，7天预报）

使用示例:
    from modules.weather_tool import WeatherTool
    tool = WeatherTool()
    print(tool._run(city="杭州", days=5))
"""

from typing import Type

import requests
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool


# ---- 天气代码 → 中文 ----
_WEATHER_CODES = {
    0: "晴", 1: "晴", 2: "多云", 3: "阴",
    45: "雾", 48: "霜雾",
    51: "小雨", 53: "中雨", 55: "大雨",
    61: "小雨", 63: "中雨", 65: "大雨",
    71: "小雪", 73: "中雪", 75: "大雪",
    80: "阵雨", 81: "大阵雨", 82: "暴雨",
    95: "雷暴", 96: "大雷暴", 99: "特大雷暴",
}


class WeatherInput(BaseModel):
    city: str = Field(description="城市名称，中文或英文。例如: '北京' '杭州' 'Shanghai'")
    days: int = Field(default=5, ge=1, le=7,
        description="查询天数。用户问'周末'要传 days=7，问'明天'传 days=2，问'今天'传 days=1")


class WeatherTool(BaseTool):
    name: str = "get_weather"
    description: str = (
        "查询指定城市未来1~7天天气预报（数据源: Open-Meteo，免费，全球覆盖）。"
        "返回每日天气状况、最高最低温度。"
        "适合在用户询问'某地天气''什么时候去某地合适'时使用。"
    )
    args_schema: Type[BaseModel] = WeatherInput

    # ---- 地理编码 ----
    @staticmethod
    def _geocode(city: str) -> tuple:
        """城市名 → (lat, lon, display_name)"""
        url = "https://geocoding-api.open-meteo.com/v1/search"
        resp = requests.get(url, params={
            "name": city, "count": 1, "language": "zh",
            "format": "json"
        }, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        if not results:
            raise ValueError(f"未找到城市: {city}")
        r = results[0]
        return (r["latitude"], r["longitude"], r.get("name", city))

    # ---- 核心 ----
    def _run(self, city: str, days: int = 3) -> str:
        try:
            lat, lon, name = self._geocode(city.strip())
        except ValueError as e:
            return str(e)
        except Exception:
            return f"天气查询失败: 无法解析城市 '{city}'，请检查城市名称。"

        try:
            resp = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat, "longitude": lon,
                    "daily": "weathercode,temperature_2m_max,temperature_2m_min",
                    "forecast_days": days,
                    "timezone": "auto",
                },
                timeout=8,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            return f"天气查询失败: 无法获取 {name} 的预报数据，请稍后重试。"

        daily = data.get("daily", {})
        dates = daily.get("time", [])
        codes = daily.get("weathercode", [])
        max_ts = daily.get("temperature_2m_max", [])
        min_ts = daily.get("temperature_2m_min", [])

        weekday = ["周一","周二","周三","周四","周五","周六","周日"]
        from datetime import date
        today = date.today()

        lines = [f"【{name} 天气】未来 {len(dates)} 天"]
        for i in range(len(dates)):
            wcode = codes[i] if i < len(codes) else 0
            wdesc = _WEATHER_CODES.get(wcode, f"code{wcode}")
            mx = max_ts[i] if i < len(max_ts) else "?"
            mn = min_ts[i] if i < len(min_ts) else "?"
            dt = date.fromisoformat(dates[i])
            wd = weekday[dt.weekday()]
            if dt == today:
                wd = "今天"
            lines.append(f"{dates[i]} {wd}: {wdesc}，{mn}°C ~ {mx}°C")

        return "\n".join(lines)
