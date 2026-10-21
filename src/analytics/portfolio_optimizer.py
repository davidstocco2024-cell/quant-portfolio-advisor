import pandas as pd
import numpy as np
from pypfopt import EfficientFrontier, risk_models, expected_returns

def optimize_portfolio(prices: pd.DataFrame, risk_free_rate: float = 0.0):
    """
    Uses Markowitz Efficient Frontier to find the optimal portfolio weights.
    We calculate two portfolios: Maximum Sharpe Ratio and Minimum Volatility.
    """
    # 1. Calculate expected returns and sample covariance matrix
    mu = expected_returns.mean_historical_return(prices)
    S = risk_models.sample_cov(prices)

    # 2. Optimize for Maximum Sharpe Ratio
    print("Calculating Maximum Sharpe Ratio Portfolio...")
    ef_sharpe = EfficientFrontier(mu, S, weight_bounds=(0, 1)) # Bound: No short selling
    ef_sharpe.max_sharpe(risk_free_rate=risk_free_rate)
    weights_sharpe = ef_sharpe.clean_weights()
    perf_sharpe = ef_sharpe.portfolio_performance(verbose=False)

    # 3. Optimize for Minimum Volatility
    print("Calculating Minimum Volatility Portfolio...")
    ef_min_vol = EfficientFrontier(mu, S, weight_bounds=(0, 1))
    ef_min_vol.min_volatility()
    weights_min_vol = ef_min_vol.clean_weights()
    perf_min_vol = ef_min_vol.portfolio_performance(verbose=False)

    return {
        "sharpe": {"weights": weights_sharpe, "performance": perf_sharpe},
        "min_vol": {"weights": weights_min_vol, "performance": perf_min_vol}
    }
    