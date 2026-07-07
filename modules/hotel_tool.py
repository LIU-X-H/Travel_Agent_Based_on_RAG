"""
酒店实时价格查询工具
====================
生成 Trip.com / 携程 / Booking 直达搜索链接，并提供参考价格区间。

使用：
    from modules.hotel_tool import HotelTool
    print(HotelTool()._run(city="杭州"))
"""

from datetime import date, timedelta
from typing import Type

from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

# 城市参考价格区间（元/晚），基于各城市酒店均价
_PRICE_RANGE = {
    "北京": "400~2000", "上海": "350~1800", "广州": "250~1500",
    "深圳": "300~1600", "杭州": "200~1200", "成都": "150~800",
    "重庆": "150~700", "西安": "150~900", "南京": "200~1000",
    "厦门": "200~1200", "三亚": "300~2000", "丽江": "100~600",
    "大理": "100~500", "桂林": "100~500", "张家界": "100~400",
    "青岛": "200~1000", "大连": "150~800", "哈尔滨": "100~500",
    "昆明": "150~700", "长沙": "150~600", "武汉": "150~700",
    "苏州": "200~1000", "黄山": "150~600", "洛阳": "100~400",
    "敦煌": "150~600", "乌鲁木齐": "150~700", "拉萨": "150~800",
}


class HotelInput(BaseModel):
    city: str = Field(description="城市名称。例如: '杭州' '北京' '成都'")


class HotelTool(BaseTool):
    name: str = "search_hotels"
    description: str = (
        "查询城市酒店参考价格和预订链接。返回该城市酒店价格区间 + Trip.com/携程直达搜索链接。"
        "适合用户问'某地住哪里''酒店多少钱一晚'时使用。"
        "参数：city（城市名）。"
    )
    args_schema: Type[BaseModel] = HotelInput

    def _run(self, city: str) -> str:
        from urllib.parse import quote
        c = city.strip()
        tomorrow = (date.today() + timedelta(days=1)).isoformat()

        pr = _PRICE_RANGE.get(c, "100~800")

        trip_url = (
            f"https://www.trip.com/hotels/list"
            f"?city={quote(c)}&checkin={tomorrow}"
        )
        ctrip_url = f"https://hotels.ctrip.com/hotel/{quote(c)}"

        return (
            f"【{c} 酒店参考】\n\n"
            f"价格区间: {pr} 元/晚\n"
            f"（具体价格因季节、地段、星级浮动）\n\n"
            f"实时价格查询：\n"
            f"1. Trip.com: {trip_url}\n"
            f"2. 携程: {ctrip_url}\n\n"
            f"提示：建议选择评分 4.0 以上、靠近地铁或景区的酒店。"
        )
