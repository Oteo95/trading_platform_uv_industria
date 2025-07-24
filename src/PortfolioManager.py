from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union
import pandas as pd
import numpy as np
import os
import httpx
import asyncio
import logging

class Event(BaseModel):
    """
    Represents a trading event (buy/sell) for event-driven or batch processing.
    """
    date: str
    ticker: str
    action: str  # "BUY" or "SELL"
    price: float
    quantity: int
    
class Price(BaseModel):
    date: str
    ticker: str
    open: float
    high: float
    low: float
    close: float

class BacktestRequest(BaseModel):
    prices: List[Price]

class MultiStrategyRequest(BaseModel):
    prices: List[Price]
    strategies: List[str] = []  # List of strategy endpoints to run
    initial_cash: Optional[float] = 10000
    parameters: Optional[Dict[str, Any]] = {}

class Event(BaseModel):
    """
    Represents a trading event (buy/sell) for event-driven or batch processing.
    """
    date: str
    ticker: str
    action: str  # "BUY" or "SELL"
    price: float
    quantity: int

class Trade:
    def __init__(self, date, ticker, action, price, shares, cash_after, invested_after, invested_per_stock=None):
        self.date = date
        self.ticker = ticker
        self.action = action
        self.price = price
        self.shares = shares
        self.cash_after = cash_after
        self.invested_after = invested_after
        self.invested_per_stock = invested_per_stock or {}

    def as_dict(self):
        return {
            "date": self.date,
            "ticker": self.ticker,
            "action": self.action,
            "price": self.price,
            "shares": self.shares,
            "cash_after": self.cash_after,
            "invested_after": self.invested_after,
            "invested_per_stock": self.invested_per_stock
        }


class PortfolioManager:
    """
    Professional Portfolio Manager supporting event-driven and batch backtesting.
    Tracks cash, invested capital, and positions per stock.
    """
    def __init__(self, initial_cash: float = 10000, max_positions: int = 10, max_per_stock: float = 0.5):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: Dict[str, int] = {}    # stock: shares
        self.invested_per_stock: Dict[str, float] = {}  # stock: invested capital
        self.trades: List[Trade] = []
        self.max_positions = max_positions    # max number of stocks to hold at once
        self.max_per_stock = max_per_stock    # max % of portfolio per stock
        self.logger = logging.getLogger("PortfolioManager")
        self.logger.setLevel(logging.INFO)

    def buy(self, date: str, stock: str, price: float, desired_money: float):
        """
        Buy as many shares as possible with desired_money, respecting max_per_stock and available cash.
        """
        investable = min(desired_money, self.max_per_stock * self.current_value({stock: price}), self.cash)
        shares = int(investable // price)
        if shares < 1:
            self.logger.info(f"BUY FAILED: {stock} at {price} on {date} (not enough cash or below min size)")
            self.trades.append(Trade(date, stock, "BUY (FAILED)", price, 0, self.cash, self.get_total_invested(), self.invested_per_stock.copy()))
            return
        total_cost = shares * price
        if self.cash >= total_cost:
            self.cash -= total_cost
            self.positions[stock] = self.positions.get(stock, 0) + shares
            self.invested_per_stock[stock] = self.invested_per_stock.get(stock, 0) + total_cost
            self.logger.info(f"BUY: {shares} {stock} at {price} on {date}")
            self.trades.append(Trade(date, stock, "BUY", price, shares, self.cash, self.get_total_invested(), self.invested_per_stock.copy()))
        else:
            self.logger.info(f"BUY FAILED: {stock} at {price} on {date} (not enough cash)")
            self.trades.append(Trade(date, stock, "BUY (FAILED)", price, 0, self.cash, self.get_total_invested(), self.invested_per_stock.copy()))

    def sell(self, date: str, stock: str, price: float):
        """
        Sell all shares of a stock.
        """
        shares = self.positions.get(stock, 0)
        if shares > 0:
            total_return = shares * price
            self.cash += total_return
            invested = self.invested_per_stock.get(stock, 0)
            self.logger.info(f"SELL: {shares} {stock} at {price} on {date}")
            self.trades.append(Trade(date, stock, "SELL", price, shares, self.cash, self.get_total_invested() - invested, self.invested_per_stock.copy()))
            self.positions[stock] = 0
            self.invested_per_stock[stock] = 0
        else:
            self.logger.info(f"SELL FAILED: {stock} at {price} on {date} (no shares)")
            self.trades.append(Trade(date, stock, "SELL (FAILED)", price, 0, self.cash, self.get_total_invested(), self.invested_per_stock.copy()))

    def handle_event(self, event: Union[Event, dict]):
        """
        Process a single event (buy/sell).
        """
        if isinstance(event, dict):
            event = Event(**event)
        if event.action.upper() == "BUY":
            self.buy(event.date, event.ticker, event.price, event.quantity * event.price)
        elif event.action.upper() == "SELL":
            self.sell(event.date, event.ticker, event.price)
        else:
            self.logger.warning(f"Unknown event action: {event.action}")

    def process_events(self, events: List[Union[Event, dict]]):
        """
        Process a batch of events (for backtesting).
        """
        for event in events:
            self.handle_event(event)

    def current_value(self, prices: Dict[str, float]) -> float:
        """
        Calculate total portfolio value given current prices.
        """
        return self.cash + sum(self.positions.get(stock, 0) * price for stock, price in prices.items())

    def get_total_invested(self) -> float:
        """
        Total invested capital (sum of all per-stock investments).
        """
        return sum(self.invested_per_stock.values())

    def get_portfolio_state(self, prices: Optional[Dict[str, float]] = None) -> dict:
        """
        Get current portfolio state, including cash, invested, and per-stock breakdown.
        """
        state = {
            "cash": self.cash,
            "total_invested": self.get_total_invested(),
            "positions": self.positions.copy(),
            "invested_per_stock": self.invested_per_stock.copy(),
        }
        if prices:
            state["current_value"] = self.current_value(prices)
        return state

    def get_trades(self) -> List[dict]:
        """
        Get trade log as list of dicts.
        """
        return [t.as_dict() for t in self.trades]