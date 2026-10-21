import pandas as pd
import numpy as np

def calculate_annualized_volatility(daily_returns: pd.DataFrame, trading_days: int = 252) -> pd.Series:
    """Standard deviation of returns, annualized (using 252 trading days)."""
    return daily_returns.std() * np.sqrt(trading_days)

def calculate_max_drawdown(prices: pd.DataFrame) -> pd.Series:
    """
    Maximum Drawdown (MDD): The worst percentage drop from a historical peak.
    Crucial for understanding the maximum pain an investor would have felt.
    """
    rolling_max = prices.cummax()
    drawdown = (prices / rolling_max) - 1.0
    return drawdown.min()

def calculate_sharpe_ratio(daily_returns: pd.DataFrame, risk_free_rate: float = 0.0, trading_days: int = 252) -> pd.Series:
    """Return per unit of total risk."""
    excess_returns = daily_returns - (risk_free_rate / trading_days)
    return (excess_returns.mean() / daily_returns.std()) * np.sqrt(trading_days)

def calculate_sortino_ratio(daily_returns: pd.DataFrame, risk_free_rate: float = 0.0, trading_days: int = 252) -> pd.Series:
    """
    Sortino Ratio: Like Sharpe, but only penalizes DOWNWARD volatility (bad returns).
    Much more realistic for asymmetric assets like Crypto or Emerging Market equities.
    """
    excess_returns = daily_returns - (risk_free_rate / trading_days)
    downside_returns = daily_returns[daily_returns < 0]
    downside_deviation = downside_returns.std() * np.sqrt(trading_days)
    
    return (excess_returns.mean() * trading_days) / downside_deviation

def calculate_value_at_risk(daily_returns: pd.DataFrame, confidence_level: float = 0.95) -> pd.Series:
    """
    Historical VaR: The maximum expected loss over a given time period at a certain confidence level.
    Returns the 5th percentile of returns (if 95% confidence).
    """
    return daily_returns.quantile(1 - confidence_level)