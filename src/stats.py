
import pandas as pd
import numpy as np


def compute_stats_from_trade_log(trade_log, initial_cash, prices_df):
    """
    Rebuild the portfolio value time series from the trade log and compute stats.
    """
    # Build a DataFrame from the trade log
    trades_df = pd.DataFrame(trade_log)
    if trades_df.empty:
        return {
            "dates": [],
            "portfolio_values": [],
            "cash_values": [],
            "trade_log": [],
            "summary": {
                "total_return": 0,
                "annualized_return": 0,
                "max_drawdown": 0,
                "sharpe_ratio": 0
            }
        }
    trades_df['date'] = pd.to_datetime(trades_df['date'])
    trades_df = trades_df.sort_values('date')

    # Get all unique dates from prices_df
    all_dates = pd.to_datetime(prices_df['date']).sort_values().unique()
    stocks = prices_df['ticker'].unique()
    price_pivot = prices_df.pivot(index='date', columns='ticker', values='close').ffill()

    # Rebuild portfolio state for each date
    cash = initial_cash
    positions = {stock: 0 for stock in stocks}
    invested_per_stock = {stock: 0.0 for stock in stocks}
    portfolio_values = []
    cash_values = []
    dates = []

    trade_idx = 0
    trades_list = trades_df.to_dict('records')

    for dt in all_dates:
        dt_str = pd.to_datetime(dt).strftime('%Y-%m-%d')
        today_prices = price_pivot.loc[dt].to_dict() if dt in price_pivot.index else {stock: 0 for stock in stocks}
        # Apply all trades for this date
        while trade_idx < len(trades_list) and pd.to_datetime(trades_list[trade_idx]['date']) == dt:
            trade = trades_list[trade_idx]
            stock = trade['ticker']
            action = trade['action']
            shares = trade['shares']
            price = trade['price']
            if action.startswith("BUY") and shares > 0:
                cost = shares * price
                cash -= cost
                positions[stock] += shares
                invested_per_stock[stock] += cost
            elif action.startswith("SELL") and shares > 0:
                proceeds = shares * price
                cash += proceeds
                positions[stock] -= shares
                invested_per_stock[stock] = 0
            trade_idx += 1
        # Compute portfolio value
        value = cash + sum(positions[stock] * today_prices.get(stock, 0) for stock in stocks)
        portfolio_values.append(value)
        cash_values.append(cash)
        dates.append(dt_str)

    # Compute stats
    if len(portfolio_values) > 1:
        total_return = 100 * (portfolio_values[-1] - initial_cash) / initial_cash
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        annualized_return = ((portfolio_values[-1] / initial_cash) ** (252/len(portfolio_values)) - 1) * 100
        max_drawdown = 100 * ((np.maximum.accumulate(portfolio_values) - portfolio_values) / np.maximum.accumulate(portfolio_values)).max()
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
    else:
        total_return = 0
        annualized_return = 0
        max_drawdown = 0
        sharpe_ratio = 0

    return {
        "dates": dates,
        "portfolio_values": portfolio_values,
        "cash_values": cash_values,
        "trade_log": trade_log,
        "summary": {
            "total_return": round(total_return, 2),
            "annualized_return": round(annualized_return, 2),
            "max_drawdown": round(-max_drawdown, 2),
            "sharpe_ratio": round(sharpe_ratio, 2)
        }
    }
