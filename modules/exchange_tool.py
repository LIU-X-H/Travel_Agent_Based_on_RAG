"""
汇率换算工具
============
本地 LangChain BaseTool，不依赖 MCP，零事件循环冲突。

使用示例：
    from modules.exchange_tool import ExchangeTool
    tool = ExchangeTool()
    print(tool._run(currency="日元", amount=5000))
"""

from typing import Type

from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool


RATES = {
    "CNY": 1.0, "人民币": 1.0,
    "USD": 0.139, "美元": 0.139, "美金": 0.139,
    "EUR": 0.127, "欧元": 0.127,
    "JPY": 20.82, "日元": 20.82,
    "KRW": 187.3, "韩元": 187.3,
    "GBP": 0.109, "英镑": 0.109,
    "HKD": 1.084, "港币": 1.084,
    "TWD": 4.45, "台币": 4.45, "新台币": 4.45,
    "THB": 4.97, "泰铢": 4.97,
    "SGD": 0.187, "新加坡元": 0.187,
    "MYR": 0.646, "马币": 0.646, "马来西亚林吉特": 0.646,
    "IDR": 2230.0, "印尼盾": 2230.0,
    "VND": 3477.0, "越南盾": 3477.0,
    "INR": 11.56, "印度卢比": 11.56,
    "RUB": 12.52, "卢布": 12.52,
    "AUD": 0.209, "澳元": 0.209,
    "NZD": 0.227, "新西兰元": 0.227,
    "CAD": 0.188, "加元": 0.188,
    "CHF": 0.122, "瑞士法郎": 0.122,
    "AED": 0.510, "阿联酋迪拉姆": 0.510,
    "TRY": 4.52, "土耳其里拉": 4.52,
}

class ExchangeInput(BaseModel):
    currency: str = Field(description="目标货币。如: '日元' '美元' '欧元' '泰铢' '韩元' '港币'")
    amount: float = Field(description="人民币金额")

class ExchangeTool(BaseTool):
    name: str = "exchange_rate"
    description: str = (
        "人民币兑外币汇率换算。输入货币名称和人民币金额，返回对应外币金额。"
        "支持: 美元 日元 欧元 泰铢 韩元 港币 英镑 澳元 等20+货币。"
    )
    args_schema: Type[BaseModel] = ExchangeInput

    def _run(self, currency: str, amount: float) -> str:
        rate = RATES.get(currency.strip())
        if not rate:
            for k in RATES:
                if currency in k or k in currency:
                    rate = RATES[k]; currency = k; break
        if not rate:
            names = [k for k in RATES if len(k) > 1 and '一' <= k[0] <= '鿿']
            return f"不支持的货币 '{currency}'。支持: {', '.join(names[:15])} 等"
        result = amount * rate
        return f"{amount} CNY = {result:,.2f} {currency}  (1 CNY = {rate} {currency})"
