import pandas as pd
import numpy as np

def calculate_portfolio_returns(daily_returns: pd.DataFrame, weights: dict) -> pd.Series:
    """
    Calculates the daily returns of a weighted portfolio.
    """
    # Convert weights dict to a pandas Series to ensure alignment
    weights_series = pd.Series(weights)
    
    # Ensure the weights sum to 1.0 (100%)
    if not np.isclose(weights_series.sum(), 1.0):
        raise ValueError("Weights must sum to 1.0")
        
    # Calculate weighted sum of returns
    portfolio_returns = (daily_returns * weights_series).sum(axis=1)
    portfolio_returns.name = "Portfolio"
    
    return portfolio_returns

def calculate_beta(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """
    Beta measures the portfolio's volatility relative to the market (benchmark).
    Beta > 1: More volatile than the market.
    Beta < 1: Less volatile (defensive).
    """
    # Calculate covariance and variance
    covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
    market_variance = np.var(benchmark_returns, ddof=1)
    return covariance / market_variance

def calculate_alpha(portfolio_returns: pd.Series, benchmark_returns: pd.Series, beta: float, risk_free_rate: float = 0.0, trading_days: int = 252) -> float:
    """
    Alpha (Jensen's Alpha) measures the excess return of the portfolio over the expected return
    given its Beta. A positive Alpha means the portfolio outperformed its risk-adjusted expectation.
    """
    # Annualized expected return based on CAPM
    annual_port_return = portfolio_returns.mean() * trading_days
    annual_bench_return = benchmark_returns.mean() * trading_days
    
    expected_return = risk_free_rate + beta * (annual_bench_return - risk_free_rate)
    
    # Alpha is the difference between actual and expected
    return annual_port_return - expected_return

def calculate_correlation_matrix(daily_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Correlation matrix shows how assets move together.
    Values close to 1: Move in the same direction.
    Values close to -1: Move in opposite directions.
    Values close to 0: No linear relationship (Diversification).
    """
    return daily_returns.corr()