"""
行程规划工具
============
基于 langchain_core.tools.BaseTool，供 Agent 自动调用。

功能：接收景点列表 + 天数，按城市和地理位置分组，
      输出合理的每日行程计划。

使用场景：
    用户先调 scenic_spot_search 获景点 → Agent 把结果传给本工具 → 输出日程
"""

import re
from typing import Optional, List, Type

from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool


# ============================================================
# Pydantic 输入参数
# ============================================================
class ItineraryInput(BaseModel):
    """行程规划入参。"""
    spots_text: str = Field(
        description=(
            "从 scenic_spot_search 获取的景点列表原文。"
            "每行一个景点，格式：'景点名 | 城市 | 票价 | 等级 | 标签 | 地址 | 开放时间'。"
            "Agent 应把所有检索到的景点拼接后传入。"
        ),
    )
    days: int = Field(
        default=2, ge=1, le=7,
        description="旅行天数，1~7",
    )
    city: Optional[str] = Field(
        default=None,
        description="主要目的地城市，用于确定行程重心",
    )
    style: Optional[str] = Field(
        default=None,
        description=(
            "旅行偏好风格。例如：'轻松休闲少走路'、'特种兵打卡式'、"
            "'亲子游'、'自然风光优先'、'文化古迹深度'。不传默认均衡安排。"
        ),
    )


# ============================================================
# 行程规划工具
# ============================================================
class ItineraryTool(BaseTool):
    """
    旅游行程规划工具。

    将景点列表按城市+区域分组，按节凑分配天数，输出可执行日程。
    """

    name: str = "plan_itinerary"
    description: str = (
        "根据已检索到的景点列表和旅行天数，规划每日行程安排。"
        "自动按城市和区域分组，避免跨城奔波，合理安排每天2~3个景点。"
        "适合在用户问'帮我规划X天行程'、'怎么安排比较合理'时使用。"
        "参数：spots_text（景点列表原文）、days（天数）、city（可选主城市）、style（可选偏好）。"
    )
    args_schema: Type[BaseModel] = ItineraryInput

    # ---- 城市间参考车程（小时）- 在 _travel_time 中以内联字典使用 ----

    @staticmethod
    def _parse_spots(spots_text: str) -> List[dict]:
        """
        从 LLM 传入的文本中解析景点信息。

        支持的格式：
            景点名 | 城市 | 票价 | 等级 | 标签 | 地址 | 开放时间
            或自由文本中含 'XX位于XX' 的模式
        """
        spots = []
        lines = spots_text.strip().split("\n")

        for line in lines:
            line = line.strip()
            if not line or len(line) < 4:
                continue

            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 3:
                spot = {
                    "name": parts[0],
                    "city": parts[1] if len(parts) > 1 else "",
                    "ticket": parts[2] if len(parts) > 2 else "",
                    "level": parts[3] if len(parts) > 3 else "",
                    "address": parts[5] if len(parts) > 5 else "",
                    "open_time": parts[6] if len(parts) > 6 else "",
                }
                spots.append(spot)
            else:
                # 自由文本提取：'XX位于XX市'
                match = re.match(r"(.+?)位于(.+?)[，。,.\s]", line)
                if match:
                    spots.append({
                        "name": match.group(1).strip(),
                        "city": match.group(2).strip(),
                        "ticket": "", "level": "", "address": "", "open_time": "",
                    })

        return spots

    @staticmethod
    def _travel_time(city_a: str, city_b: str) -> float:
        """估算两城市间车程（小时），无数据默认市区内0.5h / 跨城2h。"""
        if city_a == city_b:
            return 0.5
        key = (city_a, city_b)
        rev_key = (city_b, city_a)
        # 注意：不能使用 ItineraryTool._TRAVEL_TIME 访问类属性，
        # 因为 Pydantic v2 会将带下划线的属性存入 __pydantic_private__，
        # 类级别访问会返回 ModelPrivateAttr 描述符而非实际值。
        # 通过实例方法访问则需要改为非 staticmethod。
        # 作为临时方案，直接使用内联字典
        travel_data = {
            ("北京", "天津"): 0.5, ("上海", "杭州"): 1.0, ("上海", "苏州"): 0.5,
            ("广州", "深圳"): 0.5, ("成都", "重庆"): 1.5, ("成都", "都江堰"): 1.0,
            ("杭州", "苏州"): 1.5, ("南京", "杭州"): 1.5, ("南京", "上海"): 1.0,
            ("西安", "洛阳"): 1.5, ("桂林", "阳朔"): 1.0,
        }
        if key in travel_data:
            return travel_data[key]
        if rev_key in travel_data:
            return travel_data[rev_key]
        return 2.0  # 默认跨城

    @staticmethod
    def _parse_time(open_time: str) -> tuple:
        """从开放时间字符串提取大概开门/关门时间。"""
        # "8:30-17:00" or "旺季8:30-17:00，淡季...""全天开放"
        if "全天" in open_time or not open_time:
            return (8, 18)
        nums = re.findall(r"(\d{1,2}):(\d{2})", open_time)
        if len(nums) >= 2:
            return (int(nums[0][0]), int(nums[-1][0]))
        if nums:
            return (int(nums[0][0]), 18)
        return (8, 18)

    # ============================================================
    # 核心规划逻辑
    # ============================================================
    def _run(
        self,
        spots_text: str,
        days: int = 2,
        city: Optional[str] = None,
        style: Optional[str] = None,
    ) -> str:
        """
        规划行程并返回格式化日程。

        步骤：解析景点 → 按城市分组 → 分配天数 → 输出计划
        """
        spots = self._parse_spots(spots_text)
        if not spots:
            return (
                "[ERROR] 未能从输入中解析出景点信息。请先调用 scenic_spot_search "
                "获取景点列表，再将其结果完整传入本工具。"
            )

        # ---- 1) 按城市分组 ----
        grouped: dict = {}
        for sp in spots:
            c = sp.get("city", "未知")
            grouped.setdefault(c, []).append(sp)

        # ---- 2) 确定主城市 ----
        main_city = city or max(grouped, key=lambda c: len(grouped[c]))
        main_spots = grouped.pop(main_city, [])
        other_spots = []
        for v in grouped.values():
            other_spots.extend(v)

        # ---- 3) 每天节奏 ----
        if style and "特种兵" in style:
            per_day = min(4, max(3, len(spots) // days + 1))
        elif style and "轻松" in style:
            per_day = 2
        else:
            per_day = 3

        # ---- 4) 分配景点到天 ----
        # 规则：主场城市优先、相邻区域合并、考虑开放时间
        schedule = []  # [(day_label, [(spot, note), ...])]

        all_spots = main_spots + other_spots
        spot_idx = 0

        for day in range(1, days + 1):
            day_spots = []
            daily_cities = set()

            for _ in range(per_day):
                if spot_idx >= len(all_spots):
                    break
                sp = all_spots[spot_idx]
                sp_city = sp.get("city", "")

                # 跨城检测：一天最多跨1次城
                if daily_cities and sp_city not in daily_cities:
                    if len(daily_cities) >= 2:
                        break  # 已经跨城了，留到明天

                day_spots.append(sp)
                daily_cities.add(sp_city)
                spot_idx += 1

            if day_spots:
                schedule.append((f"第{day}天", day_spots))

        # 剩余景点追加到最后一天
        while spot_idx < len(all_spots):
            if schedule:
                schedule[-1][1].append(all_spots[spot_idx])
            spot_idx += 1

        # ---- 5) 格式化输出 ----
        if style and "亲子" in style:
            style_note = "（亲子游模式：节奏轻松，每天2~3个景点）"
        elif style and "自然" in style:
            style_note = "（自然风光模式：户外优先，避开正午暴晒）"
        elif style and "古迹" in style:
            style_note = "（文化深度模式：每个景点预留充分时间）"
        else:
            style_note = ""

        lines = [
            f"【{main_city}{days}日游行程计划】{style_note}",
            f"共 {len(spots)} 个景点，覆盖 {len(set(s.get('city','') for s in spots))} 个城市",
            "",
        ]

        for day_label, day_spots in schedule:
            lines.append(f"━━━ {day_label} ━━━")
            prev_city = None

            for i, sp in enumerate(day_spots):
                name = sp.get("name", "?")
                c = sp.get("city", "")
                open_t = sp.get("open_time", "")
                ticket = sp.get("ticket", "")

                # 跨城提示
                if prev_city and c != prev_city:
                    tt = self._travel_time(prev_city, c)
                    lines.append(f"  >> 从{prev_city}前往{c}（约{tt}小时）")

                time_slot = "上午" if i == 0 else ("下午" if i == 1 else "傍晚")
                open_h, close_h = self._parse_time(open_t)
                note = ""
                if open_h > 8:
                    note = f"（注意：{open_h}:00开门，不必太早）"
                elif close_h < 17:
                    note = f"（注意：{close_h}:00关门，建议早点去）"

                ticket_note = f" | 门票{ticket}" if ticket else ""
                lines.append(f"  {time_slot} {name}{ticket_note} {note}")

                prev_city = c

            # 日总结
            day_cities = set(s.get("city", "") for s in day_spots)
            if len(day_cities) > 1:
                lines.append(f"  [小贴士] 本日跨城，建议早点出发")
            lines.append("")

        # 总建议
        lines.append("━━━ 出行小贴士 ━━━")
        lines.append("1. 提前查好各景点是否需要预约购票，热门景点建议提前1~3天预约")
        lines.append("2. 跨城交通首选高铁，市区内打车或地铁均可")
        lines.append("3. 预留弹性时间，行程可根据天气和体力灵活调整")
        if days >= 3:
            lines.append("4. 建议中间安排半天休息，避免连续奔波疲劳")

        return "\n".join(lines)
