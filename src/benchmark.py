import pandas as pd
import numpy as np


def run_benchmark(prices_df, initial_cash=10000):
    """
    Benchmark strategy: Equally balanced buy-and-hold
    Buy equal amounts of each stock at the beginning and never sell
    """
    stocks = prices_df['ticker'].unique()
    price_pivot = prices_df.pivot(index='date', columns='ticker', values='close').ffill()
    all_dates = price_pivot.index.sort_values()
    
    # Calculate equal allocation per stock
    cash_per_stock = initial_cash / len(stocks)
    
    # Get first day prices for initial purchase
    first_day = all_dates[0]
    first_day_prices = price_pivot.loc[first_day].to_dict()
    
    # Calculate shares to buy for each stock
    benchmark_positions = {}
    benchmark_trades = []
    remaining_cash = initial_cash
    
    for stock in stocks:
        price = first_day_prices[stock]
        if pd.notnull(price) and price > 0:
            shares = int(cash_per_stock // price)
            if shares > 0:
                cost = shares * price
                benchmark_positions[stock] = shares
                remaining_cash -= cost
                benchmark_trades.append({
                    "date": first_day.strftime('%Y-%m-%d'),
                    "ticker": stock,
                    "action": "BUY (BENCHMARK)",
                    "price": price,
                    "shares": shares,
                    "cash_after": remaining_cash
                })
    
    # Calculate portfolio values over time
    benchmark_values = []
    benchmark_dates = []
    
    for dt in all_dates:
        today_prices = price_pivot.loc[dt].to_dict()
        portfolio_value = remaining_cash
        
        for stock, shares in benchmark_positions.items():
            price = today_prices.get(stock, 0)
            if pd.notnull(price):
                portfolio_value += shares * price
        
        benchmark_values.append(portfolio_value)
        benchmark_dates.append(dt.strftime('%Y-%m-%d'))
    
    # Calculate benchmark metrics
    total_return = 100 * (benchmark_values[-1] - initial_cash) / initial_cash
    returns = np.diff(benchmark_values) / benchmark_values[:-1]
    annualized_return = ((benchmark_values[-1] / initial_cash) ** (252/len(benchmark_values)) - 1) * 100 if len(benchmark_values) > 1 else 0
    max_drawdown = 100 * ((np.maximum.accumulate(benchmark_values) - benchmark_values) / np.maximum.accumulate(benchmark_values)).max()
    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
    
    return {
        "dates": benchmark_dates,
        "portfolio_values": benchmark_values,
        "trade_log": benchmark_trades,
        "summary": {
            "total_return": round(total_return, 2),
            "annualized_return": round(annualized_return, 2),
            "max_drawdown": round(-max_drawdown, 2),
            "sharpe_ratio": round(sharpe_ratio, 2)
        }
    }