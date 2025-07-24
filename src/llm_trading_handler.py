from openai import OpenAI
from pydantic import BaseModel
from typing import List, Dict
import os


class Trade(BaseModel):
    date: str
    ticker: str
    action: str
    quantity: int

class StockAdvisor:
    def __init__(self, api_key: str, model: str = "gpt-4o-2024-08-06"):
        self.api_key = api_key
        self.model = model
        self.client = OpenAI(api_key=api_key)

    def prepare_prompt(self, stock_stats: List[Dict]) -> str:
        return (
            "Here are some stocks with their stats:\n\n"
            f"{stock_stats}\n\n"
            "Which stocks would you recommend to buy and why? "
            "Give your answer as a list of tickers and a short reasoning."
        )

    def get_recommendation(self, stock_stats: List[Dict]) -> Trade:
        text = self.prepare_prompt(stock_stats)
        response = self.client.responses.parse(
            model=self.model,
            input=[
                {"role": "system", "content": "Extract the trade information."},
                {
                    "role": "user",
                    "content": f"Recomiéndame que stock comprar o vender {text}",
                },
            ],
            text_format=Trade,
        )
        return response.output_parsed

# Ejemplo de uso:
if __name__ == "__main__":
    from data_handler import process_price_data
    
    df = process_price_data(r"C:\Users\aog13\OneDrive\Escritorio\proyectos\trading_platform_uv_industria\data\prices.csv")
    
    stock_stats = []
    for stock in df:
        data = df[stock]
        stock_stats.append({'ticker': stock, 'trend_20d': data["ma_20"].iloc[-1], 'trend_60d': data["ma_60"].iloc[-1], 'volatility': data["volatility_20"].iloc[-1]})

    advisor = StockAdvisor(os.environ["OPENAI_API_KEY"])
    trade_info = advisor.get_recommendation(stock_stats)
    print(trade_info)
