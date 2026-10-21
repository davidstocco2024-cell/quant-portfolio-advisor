import pandas as pd
import numpy as np

def calculate_daily_returns(prices: pd.DataFrame, method: str = 'simple') -> pd.DataFrame:
    """
    Calculates daily returns. 
    'simple': (P_t / P_{t-1}) - 1
    'log': ln(P_t / P_{t-1}) -> Preferred for statistical modeling.
    """
    if method == 'simple':
        returns = prices.pct_change()
    elif method == 'log':
        returns = np.log(prices / prices.shift(1))
    else:
        raise ValueError("Method must be 'simple' or 'log'")
        
    return returns.dropna()

def calculate_cumulative_returns(daily_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the cumulative return of the investment over the period.
    """
    return (1 + daily_returns).cumprod() - 1