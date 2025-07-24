# Strategy Development Guide

This guide explains how to create new trading strategies that integrate with the backtesting system.

## Overview

The backtesting system supports multiple trading strategies through a modular endpoint architecture. Each strategy is implemented as a separate endpoint that receives price data and returns standardized results.

## Strategy Endpoint Requirements

### Input Format

All strategy endpoints must accept a POST request with the following JSON structure:

```json
{
  "prices": [
    {
      "date": "2024-07-01",
      "ticker": "AAPL",
      "open": 100.0,
      "high": 105.0,
      "low": 98.0,
      "close": 104.0,
      "volume": 2000.0
    },
    // ... more price records
  ],
  "initial_cash": 10000,
  "parameters": {
    // Strategy-specific parameters
  }
}
```

### Output Format

All strategy endpoints must return the following standardized JSON structure:

```json
{
  "strategy_name": "Your Strategy Name",
  "dates": ["2024-07-01", "2024-07-02", ...],
  "portfolio_values": [10000, 10050, 10100, ...],
  "cash_values": [5000, 4500, 4000, ...],
  "trade_log": [
    {
      "date": "2024-07-01",
      "ticker": "AAPL",
      "action": "BUY",
      "price": 104.0,
      "shares": 10,
      "cash_after": 8960.0
    }
  ],
  "summary": {
    "total_return": 15.5,
    "annualized_return": 12.3,
    "max_drawdown": -5.2,
    "sharpe_ratio": 1.8
  },
  "parameters_used": {
    // Echo back the parameters that were used
  }
}
```

## Creating a New Strategy

### Step 1: Define Your Strategy Logic

Create a new Python file for your strategy (e.g., `strategies/momentum_strategy.py`):

```python
import pandas as pd
import numpy as np
from typing import List, Dict, Any

class MomentumStrategy:
    def __init__(self, initial_cash: float = 10000, lookback_period: int = 20):
        self.initial_cash = initial_cash
        self.lookback_period = lookback_period
        self.cash = initial_cash
        self.positions = {}
        self.trades = []
    
    def generate_signals(self, prices_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate buy/sell signals based on momentum
        """
        signals = []
        
        for ticker in prices_df['ticker'].unique():
            df = prices_df[prices_df['ticker'] == ticker].copy()
            df = df.sort_values('date')
            
            # Calculate momentum indicator
            df['momentum'] = df['close'].pct_change(self.lookback_period)
            
            for i, row in df.iterrows():
                if pd.notnull(row['momentum']):
                    if row['momentum'] > 0.05:  # 5% momentum threshold
                        signals.append({
                            "date": row['date'],
                            "ticker": ticker,
                            "action": "BUY",
                            "quantity": 1000
                        })
                    elif row['momentum'] < -0.05:
                        signals.append({
                            "date": row['date'],
                            "ticker": ticker,
                            "action": "SELL",
                            "quantity": 0
                        })
        
        return pd.DataFrame(signals)
    
    def backtest(self, prices_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run the backtest and return standardized results
        """
        # Implementation similar to existing PortfolioManager
        # ... (implement your backtesting logic here)
        
        return {
            "strategy_name": "Momentum Strategy",
            "dates": dates,
            "portfolio_values": portfolio_values,
            "cash_values": cash_values,
            "trade_log": self.trades,
            "summary": {
                "total_return": total_return,
                "annualized_return": annualized_return,
                "max_drawdown": max_drawdown,
                "sharpe_ratio": sharpe_ratio
            },
            "parameters_used": {
                "lookback_period": self.lookback_period
            }
        }
```

### Step 2: Create Strategy Endpoint

Create a FastAPI endpoint for your strategy:

```python
from fastapi import FastAPI
from strategies.momentum_strategy import MomentumStrategy

app = FastAPI()

@app.post("/momentum-strategy")
async def momentum_strategy_endpoint(data: BacktestRequest):
    """
    Momentum-based trading strategy
    """
    prices_df = pd.DataFrame([p.dict() for p in data.prices])
    prices_df['date'] = pd.to_datetime(prices_df['date'])
    prices_df = prices_df.sort_values(['date', 'ticker'])
    
    # Extract parameters
    initial_cash = data.get('initial_cash', 10000)
    lookback_period = data.get('parameters', {}).get('lookback_period', 20)
    
    # Run strategy
    strategy = MomentumStrategy(initial_cash, lookback_period)
    results = strategy.backtest(prices_df)
    
    return results
```

### Step 3: Register Your Strategy

Add your strategy to the main backtesting system by updating the strategy registry:

```python
AVAILABLE_STRATEGIES = [
    {
        "name": "Moving Average Crossover",
        "endpoint": "/ma-crossover-strategy",
        "description": "Simple moving average crossover strategy",
        "parameters": {
            "short_window": {"type": "int", "default": 2, "min": 1, "max": 50},
            "long_window": {"type": "int", "default": 3, "min": 2, "max": 100}
        }
    },
    {
        "name": "Momentum Strategy",
        "endpoint": "/momentum-strategy", 
        "description": "Momentum-based trading strategy",
        "parameters": {
            "lookback_period": {"type": "int", "default": 20, "min": 5, "max": 100}
        }
    }
]
```

## Strategy Implementation Guidelines

### 1. Portfolio Management

Your strategy should implement proper portfolio management:

```python
class PortfolioManager:
    def __init__(self, initial_cash, max_positions=10, max_per_stock=0.2):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions = {}
        self.trades = []
        self.max_positions = max_positions
        self.max_per_stock = max_per_stock
    
    def buy(self, date, ticker, price, desired_amount):
        # Implement buy logic with position sizing
        pass
    
    def sell(self, date, ticker, price):
        # Implement sell logic
        pass
    
    def current_value(self, prices):
        # Calculate total portfolio value
        return self.cash + sum(self.positions.get(ticker, 0) * price 
                              for ticker, price in prices.items())
```

### 2. Signal Generation

Implement clear signal generation logic:

```python
def generate_signals(self, prices_df):
    """
    Generate trading signals based on your strategy logic
    
    Returns:
        DataFrame with columns: date, ticker, action, quantity
    """
    signals = []
    
    # Your signal generation logic here
    
    return pd.DataFrame(signals)
```

### 3. Performance Metrics

Calculate standard performance metrics:

```python
def calculate_metrics(self, portfolio_values, initial_cash):
    """
    Calculate standard performance metrics
    """
    total_return = 100 * (portfolio_values[-1] - initial_cash) / initial_cash
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    
    annualized_return = ((portfolio_values[-1] / initial_cash) ** 
                        (252/len(portfolio_values)) - 1) * 100
    
    max_drawdown = 100 * ((np.maximum.accumulate(portfolio_values) - 
                          portfolio_values) / 
                         np.maximum.accumulate(portfolio_values)).max()
    
    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
    
    return {
        "total_return": round(total_return, 2),
        "annualized_return": round(annualized_return, 2),
        "max_drawdown": round(-max_drawdown, 2),
        "sharpe_ratio": round(sharpe_ratio, 2)
    }
```

## Testing Your Strategy

### Unit Tests

Create unit tests for your strategy:

```python
import unittest
from strategies.momentum_strategy import MomentumStrategy

class TestMomentumStrategy(unittest.TestCase):
    def setUp(self):
        self.strategy = MomentumStrategy(initial_cash=10000)
    
    def test_signal_generation(self):
        # Test signal generation logic
        pass
    
    def test_portfolio_management(self):
        # Test buy/sell logic
        pass
    
    def test_performance_calculation(self):
        # Test metric calculations
        pass
```

### Integration Tests

Test your strategy endpoint:

```python
def test_strategy_endpoint():
    # Test the FastAPI endpoint
    response = client.post("/momentum-strategy", json=test_data)
    assert response.status_code == 200
    
    result = response.json()
    assert "strategy_name" in result
    assert "portfolio_values" in result
    assert "summary" in result
```

## Best Practices

### 1. Error Handling

Implement robust error handling:

```python
try:
    # Strategy logic
    pass
except Exception as e:
    return {
        "error": f"Strategy execution failed: {str(e)}",
        "strategy_name": "Your Strategy",
        "success": False
    }
```

### 2. Parameter Validation

Validate input parameters:

```python
def validate_parameters(self, parameters):
    """Validate strategy parameters"""
    if parameters.get('lookback_period', 0) < 1:
        raise ValueError("Lookback period must be positive")
```

### 3. Documentation

Document your strategy thoroughly:

```python
class YourStrategy:
    """
    Your Strategy Description
    
    This strategy implements [describe your approach].
    
    Parameters:
        param1 (int): Description of parameter 1
        param2 (float): Description of parameter 2
    
    Returns:
        dict: Standardized backtest results
    """
```

### 4. Performance Optimization

Optimize for performance:

```python
# Use vectorized operations where possible
df['signal'] = np.where(df['indicator'] > threshold, 1, 0)

# Avoid loops when possible
# Use pandas operations instead of iterating
```

## Deployment

### 1. Local Development

Run your strategy locally:

```bash
uvicorn your_strategy:app --reload --port 8001
```

### 2. Production Deployment

Deploy using Docker:

```dockerfile
FROM python:3.9

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "your_strategy:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Example Strategies

### 1. Mean Reversion Strategy

```python
def generate_signals(self, prices_df):
    signals = []
    
    for ticker in prices_df['ticker'].unique():
        df = prices_df[prices_df['ticker'] == ticker].copy()
        df['sma'] = df['close'].rolling(20).mean()
        df['std'] = df['close'].rolling(20).std()
        df['z_score'] = (df['close'] - df['sma']) / df['std']
        
        # Buy when oversold, sell when overbought
        df['signal'] = np.where(df['z_score'] < -2, 'BUY',
                               np.where(df['z_score'] > 2, 'SELL', 'HOLD'))
```

### 2. Breakout Strategy

```python
def generate_signals(self, prices_df):
    signals = []
    
    for ticker in prices_df['ticker'].unique():
        df = prices_df[prices_df['ticker'] == ticker].copy()
        df['high_20'] = df['high'].rolling(20).max()
        df['low_20'] = df['low'].rolling(20).min()
        
        # Buy on breakout above 20-day high
        # Sell on breakdown below 20-day low
        df['signal'] = np.where(df['close'] > df['high_20'].shift(1), 'BUY',
                               np.where(df['close'] < df['low_20'].shift(1), 'SELL', 'HOLD'))
```

## Support and Resources

- Review existing strategies in the `/strategies` directory
- Check the main backtesting system for implementation examples
- Refer to the API documentation for endpoint specifications
- Use the provided testing framework for validation

## Conclusion

Following this guide ensures your strategy integrates seamlessly with the backtesting system and provides consistent, comparable results across all strategies.
