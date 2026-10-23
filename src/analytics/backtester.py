from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


TRADING_DAYS = 252


def _validate_inputs(
    prices: pd.DataFrame,
    optimized_weights: Dict[str, float],
    initial_capital: float,
    rebalance_freq: int,
    transaction_cost: float,
    benchmark_ticker: str,
) -> None:
    """Validate backtest inputs."""
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame.")

    if prices.empty:
        raise ValueError("prices cannot be empty.")

    if prices.shape[0] < 2:
        raise ValueError("At least two price observations are required.")

    if not prices.columns.is_unique:
        raise ValueError("Price columns must be unique.")

    if prices.isna().any().any():
        raise ValueError("prices cannot contain missing values.")

    price_values = prices.to_numpy(dtype=float)

    if not np.isfinite(price_values).all():
        raise ValueError("prices must contain only finite values.")

    if (price_values <= 0).any():
        raise ValueError("prices must be strictly positive.")

    if not prices.index.is_monotonic_increasing:
        raise ValueError("Price dates must be sorted in ascending order.")

    if not prices.index.is_unique:
        raise ValueError("Price dates must be unique.")

    if benchmark_ticker not in prices.columns:
        raise ValueError(
            f"Benchmark '{benchmark_ticker}' not found in price data."
        )

    if not isinstance(optimized_weights, dict) or not optimized_weights:
        raise ValueError("optimized_weights must be a non-empty dictionary.")

    if not np.isfinite(initial_capital) or initial_capital <= 0:
        raise ValueError("initial_capital must be positive and finite.")

    if (
        not isinstance(rebalance_freq, int)
        or isinstance(rebalance_freq, bool)
        or rebalance_freq < 1
    ):
        raise ValueError("rebalance_freq must be a positive integer.")

    if (
        not np.isfinite(transaction_cost)
        or transaction_cost < 0
        or transaction_cost >= 1
    ):
        raise ValueError("transaction_cost must be in the interval [0, 1).")

    for ticker, weight in optimized_weights.items():
        if not np.isfinite(weight):
            raise ValueError(f"Weight for {ticker} must be finite.")

        if weight < 0:
            raise ValueError("This backtester supports long-only weights.")


def _calculate_metrics(
    portfolio_values: pd.Series,
    risk_free_rate: float = 0.05,
) -> Dict:
    """Calculate performance metrics from a portfolio value series."""
    returns = portfolio_values.pct_change().dropna()

    if returns.empty:
        return {
            "Total Return": 0.0,
            "Annualized Return": 0.0,
            "Annualized Volatility": 0.0,
            "Sharpe Ratio": float("nan"),
            "Max Drawdown": 0.0,
            "Underwater": pd.Series(
                0.0,
                index=portfolio_values.index,
                name="Underwater",
            ),
        }

    total_return = portfolio_values.iloc[-1] / portfolio_values.iloc[0] - 1.0

    elapsed_periods = len(returns)
    annualized_return = (
        (portfolio_values.iloc[-1] / portfolio_values.iloc[0])
        ** (TRADING_DAYS / elapsed_periods)
        - 1.0
    )

    annualized_volatility = returns.std(ddof=1) * np.sqrt(TRADING_DAYS)

    daily_risk_free_rate = risk_free_rate / TRADING_DAYS
    annualized_excess_return = (
        (returns.mean() - daily_risk_free_rate) * TRADING_DAYS
    )

    if annualized_volatility > 0:
        sharpe_ratio = (
            annualized_excess_return / annualized_volatility
        )
    else:
        sharpe_ratio = float("nan")

    running_peak = portfolio_values.cummax()
    drawdown = portfolio_values / running_peak - 1.0
    max_drawdown = float(drawdown.min())

    return {
        "Total Return": float(total_return),
        "Annualized Return": float(annualized_return),
        "Annualized Volatility": float(annualized_volatility),
        "Sharpe Ratio": float(sharpe_ratio),
        "Max Drawdown": max_drawdown,
        "Underwater": drawdown.rename("Underwater"),
    }


def run_backtest(
    prices: pd.DataFrame,
    optimized_weights: Dict[str, float],
    initial_capital: float = 10000.0,
    rebalance_freq: int = 21,
    transaction_cost: float = 0.001,
    benchmark_ticker: str = "^GSPC",
) -> Tuple[pd.DataFrame, Dict]:
    """
    Backtest a long-only portfolio with periodic rebalancing and costs.

    The supplied target weights are set at the beginning of the backtest.
    Daily returns are applied using the weights held at the start of each day.
    Rebalancing occurs after every rebalance_freq trading days; the new
    target weights apply to the following day's return.

    transaction_cost is the cost per dollar traded. Gross traded turnover
    is the sum of absolute changes in portfolio weights.
    """
    _validate_inputs(
        prices=prices,
        optimized_weights=optimized_weights,
        initial_capital=initial_capital,
        rebalance_freq=rebalance_freq,
        transaction_cost=transaction_cost,
        benchmark_ticker=benchmark_ticker,
    )

    daily_returns = prices.pct_change().dropna()

    if daily_returns.empty:
        raise ValueError("Not enough data to calculate daily returns.")

    # Portfolio assets must exist in the price data.
    assets = [
        ticker
        for ticker in optimized_weights
        if ticker in daily_returns.columns and ticker != benchmark_ticker
    ]

    if not assets:
        raise ValueError(
            "No valid portfolio assets found in prices. "
            "Check that the weight tickers match the price columns."
        )

    # Ignore a benchmark weight if it was accidentally included.
    raw_weights = pd.Series(
        {ticker: optimized_weights[ticker] for ticker in assets},
        dtype=float,
    )

    if raw_weights.sum() <= 0:
        raise ValueError("Portfolio weights must have a positive sum.")

    target_weights = raw_weights / raw_weights.sum()

    portfolio_asset_returns = daily_returns[assets]
    benchmark_returns = daily_returns[benchmark_ticker]

    # Values include the initial capital point, before the first daily return.
    value_dates = [prices.index[0]]
    portfolio_values = [float(initial_capital)]
    benchmark_values = [float(initial_capital)]

    current_weights = target_weights.copy()
    turnover_history = []
    rebalance_dates = []

    for day_number, date in enumerate(portfolio_asset_returns.index, start=1):
        asset_returns_today = portfolio_asset_returns.loc[date]

        # Compute today's portfolio return using weights held at the start
        # of the day.
        portfolio_return_today = float(
            (asset_returns_today * current_weights).sum()
        )

        new_portfolio_value = (
            portfolio_values[-1] * (1.0 + portfolio_return_today)
        )

        new_benchmark_value = (
            benchmark_values[-1]
            * (1.0 + float(benchmark_returns.loc[date]))
        )

        # After today's return, update weights to reflect market drift.
        drifted_weights = current_weights * (1.0 + asset_returns_today)
        drifted_total = drifted_weights.sum()

        if drifted_total <= 0 or not np.isfinite(drifted_total):
            raise ValueError(
                f"Invalid portfolio weight drift on {date}."
            )

        drifted_weights = drifted_weights / drifted_total

        # Rebalance after every N return observations. The target allocation
        # is used for the next day's return.
        if day_number % rebalance_freq == 0:
            gross_turnover = float(
                (target_weights - drifted_weights).abs().sum()
            )

            cost_fraction = gross_turnover * transaction_cost

            new_portfolio_value *= 1.0 - cost_fraction

            turnover_history.append(gross_turnover)
            rebalance_dates.append(date)

            current_weights = target_weights.copy()
        else:
            current_weights = drifted_weights

        value_dates.append(date)
        portfolio_values.append(new_portfolio_value)
        benchmark_values.append(new_benchmark_value)

    backtest_df = pd.DataFrame(
        {
            "Optimized Portfolio": portfolio_values,
            "Benchmark": benchmark_values,
        },
        index=pd.Index(value_dates, name=prices.index.name),
    )

    portfolio_metrics = _calculate_metrics(
        backtest_df["Optimized Portfolio"]
    )

    benchmark_metrics = _calculate_metrics(
        backtest_df["Benchmark"]
    )

    metrics = {
        "Portfolio": portfolio_metrics,
        "Benchmark": benchmark_metrics,
        "Avg Turnover": (
            float(np.mean(turnover_history))
            if turnover_history
            else 0.0
        ),
        "Num_Rebalances": len(turnover_history),
        "Rebalance Dates": rebalance_dates,
    }

    return backtest_df, metrics