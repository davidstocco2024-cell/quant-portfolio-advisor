from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, expected_returns
from pypfopt.hierarchical_portfolio import HRPOpt
from pypfopt.risk_models import CovarianceShrinkage


TRADING_DAYS = 252


def _validate_prices(
    prices: pd.DataFrame,
    max_weight: float | None = None,
) -> None:
    """Validate historical asset prices before optimization."""
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame.")

    if prices.empty:
        raise ValueError("prices cannot be empty.")

    if prices.shape[1] < 2:
        raise ValueError("At least two assets are required.")

    if prices.shape[0] < 3:
        raise ValueError("At least three price observations are required.")

    if not prices.columns.is_unique:
        raise ValueError("Asset columns must be unique.")

    if prices.isna().any().any():
        raise ValueError("prices cannot contain missing values.")

    values = prices.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("prices must contain only finite values.")

    if (values <= 0).any():
        raise ValueError("prices must be strictly positive.")

    if max_weight is not None:
        if not np.isfinite(max_weight) or not 0 < max_weight <= 1:
            raise ValueError("max_weight must be in the interval (0, 1].")

        # The total weight capacity must be sufficient to allocate 100%.
        if prices.shape[1] * max_weight < 1:
            raise ValueError(
                "max_weight is too small: the assets cannot add up to "
                "a fully invested portfolio under this cap."
            )


def _get_inputs(
    prices: pd.DataFrame,
) -> tuple[pd.Series, pd.DataFrame]:
    """Estimate annualized expected returns and covariance matrix."""
    mu = expected_returns.mean_historical_return(
        prices,
        frequency=TRADING_DAYS,
    )

    covariance = CovarianceShrinkage(
        prices,
        frequency=TRADING_DAYS,
    ).ledoit_wolf()

    return mu, covariance


def _maximum_feasible_return(
    expected_returns_series: pd.Series,
    max_weight: float,
) -> float:
    """
    Find the maximum expected return under long-only weights,
    a per-asset cap, and weights summing to one.
    """
    sorted_returns = expected_returns_series.sort_values(ascending=False)

    remaining_weight = 1.0
    portfolio_return = 0.0

    for asset_return in sorted_returns:
        allocated_weight = min(max_weight, remaining_weight)
        portfolio_return += allocated_weight * asset_return
        remaining_weight -= allocated_weight

        if remaining_weight <= 1e-12:
            break

    if remaining_weight > 1e-8:
        raise ValueError("Could not allocate the full portfolio.")

    return float(portfolio_return)


def _markowitz_result(
    mu: pd.Series,
    covariance: pd.DataFrame,
    risk_free_rate: float,
    max_weight: float,
    objective: str,
) -> dict:
    """Run one Markowitz optimization and package its results."""
    ef = EfficientFrontier(
        mu,
        covariance,
        weight_bounds=(0, max_weight),
    )

    if objective == "sharpe":
        ef.max_sharpe(risk_free_rate=risk_free_rate)
    elif objective == "min_vol":
        ef.min_volatility()
    else:
        raise ValueError(f"Unknown Markowitz objective: {objective}")

    weights = ef.clean_weights()
    performance = ef.portfolio_performance(
        verbose=False,
        risk_free_rate=risk_free_rate,
    )

    return {
        "weights": dict(weights),
        "performance": {
            "expected_return": float(performance[0]),
            "volatility": float(performance[1]),
            "sharpe_ratio": float(performance[2]),
        },
    }


def _hrp_result(
    prices: pd.DataFrame,
    risk_free_rate: float,
) -> dict:
    """Optimize using Hierarchical Risk Parity and calculate performance."""
    daily_returns = prices.pct_change().dropna()

    hrp = HRPOpt(daily_returns)
    hrp.optimize()

    weights = hrp.clean_weights()

    # Explicitly align weights to the return columns.
    weight_series = pd.Series(weights, dtype=float).reindex(
        daily_returns.columns
    )

    if weight_series.isna().any():
        raise ValueError("HRP returned weights that do not match asset columns.")

    portfolio_returns = daily_returns.mul(weight_series, axis="columns").sum(
        axis=1
    )

    annualized_return = (
        (1.0 + portfolio_returns).prod()
        ** (TRADING_DAYS / len(portfolio_returns))
        - 1.0
    )

    annualized_volatility = (
        portfolio_returns.std(ddof=1) * np.sqrt(TRADING_DAYS)
    )

    daily_risk_free_rate = risk_free_rate / TRADING_DAYS

    if annualized_volatility > 0:
        sharpe_ratio = (
            (portfolio_returns.mean() - daily_risk_free_rate)
            * TRADING_DAYS
            / annualized_volatility
        )
    else:
        sharpe_ratio = float("nan")

    return {
        "weights": dict(weights),
        "performance": {
            "expected_return": float(annualized_return),
            "volatility": float(annualized_volatility),
            "sharpe_ratio": float(sharpe_ratio),
        },
    }


def optimize_portfolio(
    prices: pd.DataFrame,
    risk_free_rate: float = 0.05,
    max_weight: float = 1.0,
    method: str = "both",
) -> Dict:
    """
    Optimize a portfolio using Markowitz and/or HRP.

    Supported methods:
        "sharpe": Markowitz maximum Sharpe ratio
        "min_vol": Markowitz minimum volatility
        "hrp": Hierarchical Risk Parity
        "both": run all three methods

    max_weight applies to the Markowitz optimizations.
    HRP does not enforce this individual asset weight cap.
    """
    _validate_prices(prices, max_weight=max_weight)

    if not np.isfinite(risk_free_rate):
        raise ValueError("risk_free_rate must be finite.")

    valid_methods = {"sharpe", "min_vol", "hrp", "both"}

    if method not in valid_methods:
        raise ValueError(
            f"method must be one of {sorted(valid_methods)}."
        )

    mu, covariance = _get_inputs(prices)

    results = {}

    if method in ("sharpe", "both"):
        results["sharpe"] = _markowitz_result(
            mu,
            covariance,
            risk_free_rate,
            max_weight,
            objective="sharpe",
        )

    if method in ("min_vol", "both"):
        results["min_vol"] = _markowitz_result(
            mu,
            covariance,
            risk_free_rate,
            max_weight,
            objective="min_vol",
        )

    if method in ("hrp", "both"):
        results["hrp"] = _hrp_result(prices, risk_free_rate)

    return results


def generate_efficient_frontier(
    prices: pd.DataFrame,
    risk_free_rate: float = 0.05,
    max_weight: float = 1.0,
    points: int = 50,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Generate Markowitz efficient-frontier points and optimal portfolios.

    Returns:
        frontier_df: DataFrame with columns "return" and "volatility".
        optimal_points: Results from maximum Sharpe, minimum volatility,
                        and HRP optimizations.
    """
    _validate_prices(prices, max_weight=max_weight)

    if not np.isfinite(risk_free_rate):
        raise ValueError("risk_free_rate must be finite.")

    if not isinstance(points, int) or points < 2:
        raise ValueError("points must be an integer of at least 2.")

    mu, covariance = _get_inputs(prices)

    # Minimum-volatility portfolio gives the lower return endpoint.
    ef_min = EfficientFrontier(
        mu,
        covariance,
        weight_bounds=(0, max_weight),
    )
    ef_min.min_volatility()

    min_return = float(ef_min.portfolio_performance()[0])

    # Calculate the highest feasible expected return under the weight cap.
    max_return = _maximum_feasible_return(mu, max_weight)

    if max_return < min_return:
        raise ValueError(
            "The estimated maximum feasible return is below the "
            "minimum-volatility portfolio return."
        )

    target_returns = np.linspace(min_return, max_return, points)

    frontier_data = []

    for target_return in target_returns:
        try:
            ef = EfficientFrontier(
                mu,
                covariance,
                weight_bounds=(0, max_weight),
            )

            ef.efficient_return(target_return=float(target_return))
            performance = ef.portfolio_performance()

            frontier_data.append(
                {
                    "return": float(performance[0]),
                    "volatility": float(performance[1]),
                }
            )
        except (ValueError, RuntimeError):
            # Some endpoint targets may be numerically infeasible.
            continue

    frontier_df = pd.DataFrame(
        frontier_data,
        columns=["return", "volatility"],
    )

    optimal_points = optimize_portfolio(
        prices=prices,
        risk_free_rate=risk_free_rate,
        max_weight=max_weight,
        method="both",
    )

    return frontier_df, optimal_points