from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union
import pandas as pd
import numpy as np
import os
import httpx

from PortfolioManager import *
from benchmark import *
from stats import *
from data_handler import *
from llm_trading_handler import *


import os
from fastapi import FastAPI, File, UploadFile
from typing import List
from pydantic import BaseModel

from fastapi.middleware.cors import CORSMiddleware

from pathlib import Path
from typing import Union

from fastapi.responses import FileResponse
from starlette.requests import Request
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates









PATH_TO_VUE_APP_BUILD_DIR = f"{os.path.dirname(os.path.realpath(__file__))}/dist"
# For these files on root level, index.html fallback should not happen
STATIC_TOP_LEVEL_FILES = [
    "favicon.ico",
    "manifest.json",
    "logo192.png",
    "logo512.png",
    "robots.txt",
]
app = FastAPI()

# Strategy Registry
AVAILABLE_STRATEGIES = {
    "estrategia_A": "/estrategia_A",
    "estrategia_B": "/estrategia_B"
}
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

 
@app.get("/prices")
async def get_prices():
    """Read prices.csv from the public folder and return the data"""
    try:
        # Get the path to the public folder relative to the API
        csv_path = os.path.join("..", "data", "prices.csv")
        
        # Check if file exists
        if not os.path.exists(csv_path):
            return {"error": "prices.csv file not found"}
        
        # Read CSV file
        df = pd.read_csv(csv_path)
        
        # Convert to list of dictionaries
        prices_data = df.to_dict('records')
        
        return {
            "success": True,
            "data": prices_data,
            "csv_content": df.to_csv(index=False)
        }
    except Exception as e:
        return {"error": f"Failed to read CSV file: {str(e)}"}


@app.get("/strategies")
async def get_available_strategies():
    """Get list of available strategies"""
    return {"strategies": AVAILABLE_STRATEGIES}


async def call_strategy_endpoint(endpoint: str, data: dict):
    """Call a strategy endpoint and return results"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(endpoint, json=data, timeout=30.0)
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "error": f"Strategy endpoint returned {response.status_code}",
                    "strategy_name": f"Strategy at {endpoint}",
                    "success": False
                }
    except Exception as e:
        return {
            "error": f"Failed to call strategy endpoint: {str(e)}",
            "strategy_name": f"Strategy at {endpoint}",
            "success": False
        }
        
        
@app.post("/strategia_A")
async def strategia_A(data: BacktestRequest):

    df = process_price_data_from_json(data)
    
    trades = []
    data = df["AAPL"]
    data2 = df["GOOG"]
    data3 = df["MSFT"]
    for idx, row in data.iterrows():
        stock_stats = []
        try:
            stock_stats.append({'ticker': "AAPL", 'trend_20d': row["ma_20"], 'trend_60d': row["ma_60"], 'volatility': row["volatility_20"]})
            stock_stats.append({'ticker': "GOOG", 'trend_20d': data2.iloc[idx]["ma_20"], 'trend_60d': data2.iloc[idx]["ma_60"], 'volatility': data2.iloc[idx]["volatility_20"]})
            stock_stats.append({'ticker': "MSFT", 'trend_20d': data3.iloc[idx]["ma_20"], 'trend_60d': data3.iloc[idx]["ma_60"], 'volatility': data3.iloc[idx]["volatility_20"]})
        except IndexError:
            pass
        advisor = StockAdvisor("")
        trades.append(dict(advisor.get_recommendation(stock_stats)))
    return trades

# Modelos
class PriceEntry(BaseModel):
    date: str
    ticker: str
    close: float
    # Añade aquí más campos si los necesitas

class BacktestRequest(BaseModel):
    prices: List[PriceEntry]
    strategy_name: str = "estrategia_A"  # Por defecto la A

@app.post("/backtest")
async def backtest(data: BacktestRequest):
    # 1. Buscar el endpoint de la estrategia
    endpoint = AVAILABLE_STRATEGIES.get(data.strategy_name)
    if not endpoint:
        return {"error": f"Estrategia '{data.strategy_name}' no encontrada. Opciones: {list(AVAILABLE_STRATEGIES.keys())}"}

    # 2. Llamar a la estrategia seleccionada como endpoint local
    url = f"http://127.0.0.1:4000{endpoint}"  # asume que FastAPI está localmente en el mismo puerto
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=data.model_dump())
            if resp.status_code != 200:
                return {"error": f"Error llamando a {endpoint}: {resp.status_code} - {resp.text}"}
            signals = resp.json()
    except Exception as e:
        return {"error": f"Error llamando a endpoint de estrategia: {str(e)}"}

    # 3. Continuar con la lógica habitual
    prices_df = pd.DataFrame([p.model_dump() for p in data.prices])
    prices_df['date'] = pd.to_datetime(prices_df['date'])
    prices_df = prices_df.sort_values(['date', 'ticker'])

    stocks = prices_df['ticker'].unique()
    price_pivot = prices_df.pivot(index='date', columns='ticker', values='close').ffill()
    all_dates = price_pivot.index.sort_values()
    signals_map = {(pd.to_datetime(row['date']), row['ticker']): (row['action'], row['quantity']) for row in signals}

    pm = PortfolioManager(initial_cash=10000, max_positions=10, max_per_stock=0.5)
    portfolio_values = []
    cash_values = []
    dates = []

    for dt in all_dates:
        today_prices = price_pivot.loc[dt].to_dict()
        for stock in stocks:
            price = today_prices.get(stock)
            signal = signals_map.get((dt, stock))
            if signal:
                action, quantity = signal
                if action == "BUY":
                    pm.buy(dt.strftime('%Y-%m-%d'), stock, price, quantity)
                elif action == "SELL":
                    pm.sell(dt.strftime('%Y-%m-%d'), stock, price)
        value = pm.current_value(today_prices)
        portfolio_values.append(value)
        cash_values.append(pm.cash)
        dates.append(dt.strftime('%Y-%m-%d'))

    initial_cash = pm.initial_cash
    total_return = 100 * (portfolio_values[-1] - initial_cash) / initial_cash
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    annualized_return = ((portfolio_values[-1] / initial_cash) ** (252/len(portfolio_values)) - 1) * 100 if len(portfolio_values) > 1 else 0
    max_drawdown = 100 * ((np.maximum.accumulate(portfolio_values) - portfolio_values) / np.maximum.accumulate(portfolio_values)).max()
    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0

    benchmark_results = run_benchmark(prices_df, initial_cash)
    strategy_final_return = total_return
    benchmark_final_return = benchmark_results["summary"]["total_return"]
    outperformance = strategy_final_return - benchmark_final_return

    return {
        "strategy": {
            "dates": dates,
            "portfolio_values": portfolio_values,
            "cash_values": cash_values,
            "trade_log": pm.get_trades(),
            "summary": {
                "total_return": round(total_return, 2),
                "annualized_return": round(annualized_return, 2),
                "max_drawdown": round(-max_drawdown, 2),
                "sharpe_ratio": round(sharpe_ratio, 2)
            }
        },
        "benchmark": benchmark_results,
        "comparison": {
            "outperformance": round(outperformance, 2),
            "strategy_beats_benchmark": strategy_final_return > benchmark_final_return
        },
        "dates": dates,
        "portfolio_values": portfolio_values,
        "cash_values": cash_values,
        "trade_log": pm.get_trades(),
        "summary": {
            "total_return": round(total_return, 2),
            "annualized_return": round(annualized_return, 2),
            "max_drawdown": round(-max_drawdown, 2),
            "sharpe_ratio": round(sharpe_ratio, 2)
        }
    }



# https://stackoverflow.com/a/70065066/3652805
def serve_react_app_wrapper(outer_app: FastAPI, build_dir: Union[Path, str]) -> FastAPI:
    """Serves a React application in the root directory `/`

    Args:
        outer_app: FastAPI application instance
        build_dir: React build directory (generated by `yarn build` or
            `npm run build`)

    Returns:
        FastAPI: instance with the react application added
    """
    if isinstance(build_dir, str):
        build_dir = Path(build_dir)

    app.mount(
        "/",
        StaticFiles(directory=PATH_TO_VUE_APP_BUILD_DIR, html=True),
        name="VueApp"
    )

    outer_app.mount(
        "/",
        StaticFiles(directory=PATH_TO_VUE_APP_BUILD_DIR, html=True),
        name="VueApp"
    )
    templates = Jinja2Templates(directory=build_dir.as_posix())

    @outer_app.get("/{full_path:path}", include_in_schema=False)
    async def serve_react_app(
        request: Request, full_path: str
    ):  # pylint: disable=unused-argument
        """Serve the react app
        `full_path` variable is necessary to serve each possible endpoint with
        `index.html` file in order to be compatible with `react-router-dom`
        """
        if full_path in STATIC_TOP_LEVEL_FILES:
            return FileResponse(build_dir / full_path)
        return templates.TemplateResponse("index.html", {"request": request})

    return outer_app


app = serve_react_app_wrapper(app, PATH_TO_VUE_APP_BUILD_DIR)