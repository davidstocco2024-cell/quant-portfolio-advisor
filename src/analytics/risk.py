from __future__ import annotations

import numpy as np
import pandas as pd


AssetData = pd.Series | pd.DataFrame


def _validate_data(
    data: AssetData,
    name: str = "data",
    allow_negative: bool = True,
) -> None:
    """Validate that input is a non-empty pandas Series or DataFrame."""
    if not isinstance(data, (pd.Series, pd.DataFrame)):
        raise TypeError(f"{name} must be a pandas Series or DataFrame.")

    if data.empty:
        raise ValueError(f"{name} cannot be empty.")

    if data.isna().any().any():
        raise ValueError(f"{name} cannot contain missing values.")

    if not np.isfinite(data.to_numpy(dtype=float)).all():
        raise ValueError(f"{name} must contain only finite numeric values.")

    if not allow_negative and (data <= 0).any().any():
        raise ValueError(f"{name} must contain only positive values.")


def _validate_rate(rate: float, name: str) -> None:
    """Validate a finite numeric rate."""
    if not np.isscalar(rate) or not np.isfinite(rate):
        raise ValueError(f"{name} must be a finite number.")


def _return_scalar_or_series(
    result: pd.Series | float,
    original: AssetData,
) -> pd.Series | float:
    """Return a scalar for Series input and a Series for DataFrame input."""
    if isinstance(original, pd.Series):
        if isinstance(result, pd.Series):
            return float(result.iloc[0])
        return float(result)

    return result


def calculate_annualized_volatility(
    returns: AssetData,
    trading_days: int = 252,
) -> pd.Series | float:
    """
    Calculate annualized volatility from periodic returns.

    Uses sample standard deviation (ddof=1).
    """
    _validate_data(returns, name="returns")

    if trading_days <= 0:
        raise ValueError("trading_days must be positive.")

    if len(returns) < 2:
        return float("nan") if isinstance(returns, pd.Series) else pd.Series(
            np.nan, index=returns.columns
        )

    volatility = returns.std(ddof=1) * np.sqrt(trading_days)

    if isinstance(returns, pd.Series):
        return float(volatility)

    return volatility


def calculate_max_drawdown(prices: AssetData) -> pd.Series | float:
    """
    Calculate maximum drawdown from a price series or price DataFrame.

    Drawdown is expressed as a negative percentage/decimal.
    """
    _validate_data(prices, name="prices", allow_negative=False)

    running_peak = prices.cummax()
    drawdowns = prices / running_peak - 1.0
    max_drawdown = drawdowns.min()

    if isinstance(prices, pd.Series):
        return float(max_drawdown)

    return max_drawdown


def calculate_sharpe_ratio(
    returns: AssetData,
    risk_free_rate: float = 0.0,
    trading_days: int = 252,
) -> pd.Series | float:
    """
    Calculate annualized Sharpe ratio.

    risk_free_rate is an annualized decimal rate, e.g. 0.05 for 5%.
    """
    _validate_data(returns, name="returns")
    _validate_rate(risk_free_rate, "risk_free_rate")

    if trading_days <= 0:
        raise ValueError("trading_days must be positive.")

    daily_risk_free_rate = risk_free_rate / trading_days
    excess_returns = returns - daily_risk_free_rate

    daily_std = returns.std(ddof=1)
    annualized_excess_return = excess_returns.mean() * trading_days

    sharpe = (
        annualized_excess_return
        / (daily_std * np.sqrt(trading_days))
    )

    sharpe = sharpe.replace([np.inf, -np.inf], np.nan) if isinstance(
        sharpe, pd.Series
    ) else sharpe

    if isinstance(returns, pd.Series):
        if not np.isfinite(sharpe):
            return float("nan")
        return float(sharpe)

    return sharpe


def calculate_sortino_ratio(
    returns: AssetData,
    risk_free_rate: float = 0.0,
    target_return: float = 0.0,
    trading_days: int = 252,
) -> pd.Series | float:
    """
    Calculate annualized Sortino ratio.
    """
    _validate_data(returns, name="returns")
    _validate_rate(risk_free_rate, "risk_free_rate")
    _validate_rate(target_return, "target_return")

    if trading_days <= 0:
        raise ValueError("trading_days must be positive.")

    # Downside deviation respecto al target
    daily_target = target_return / trading_days
    downside_returns = (returns - daily_target).clip(upper=0.0)
    downside_deviation = np.sqrt((downside_returns**2).mean()) * np.sqrt(trading_days)

    # El exceso de retorno debe ser respecto al target (no al risk_free)
    annualized_excess_return = (returns.mean() * trading_days) - target_return

    sortino = annualized_excess_return / downside_deviation

    if isinstance(sortino, pd.Series):
        sortino = sortino.replace([np.inf, -np.inf], np.nan)
        return sortino

    if not np.isfinite(sortino):
        return float("nan")

    return float(sortino)

def calculate_historical_var(
    returns: AssetData,
    confidence_level: float = 0.95,
) -> pd.Series | float:
    """
    Calculate historical Value at Risk (VaR).

    Returns the lower-tail return quantile. VaR is typically negative
    when it represents a loss.
    """
    _validate_data(returns, name="returns")

    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1.")

    quantile = 1.0 - confidence_level
    var = returns.quantile(quantile)

    if isinstance(returns, pd.Series):
        return float(var)

    return var


def calculate_historical_cvar(
    returns: AssetData,
    confidence_level: float = 0.95,
) -> pd.Series | float:
    """
    Calculate historical Conditional Value at Risk (CVaR).

    CVaR is the average return in the lower tail at or below VaR.
    """
    _validate_data(returns, name="returns")

    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1.")

    var = calculate_historical_var(
        returns,
        confidence_level=confidence_level,
    )

    if isinstance(returns, pd.Series):
        tail_returns = returns[returns <= var]

        if tail_returns.empty:
            return float("nan")

        return float(tail_returns.mean())

    tail_means = {}

    for column in returns.columns:
        column_var = var[column]
        tail_returns = returns[column][returns[column] <= column_var]

        tail_means[column] = (
            float(tail_returns.mean())
            if not tail_returns.empty
            else float("nan")
        )

    return pd.Series(tail_means)


def calculate_calmar_ratio(
    prices: AssetData,
    trading_days: int = 252,
) -> pd.Series | float:
    """
    Calculate Calmar ratio: CAGR divided by the absolute maximum drawdown.

    For a DatetimeIndex, CAGR uses elapsed calendar years.
    Otherwise, it estimates years from the number of observations.
    """
    _validate_data(prices, name="prices", allow_negative=False)

    if trading_days <= 0:
        raise ValueError("trading_days must be positive.")

    if len(prices) < 2:
        return float("nan") if isinstance(prices, pd.Series) else pd.Series(
            np.nan, index=prices.columns
        )

    # Sort chronologically when the index is datetime-based.
    if isinstance(prices.index, pd.DatetimeIndex):
        prices = prices.sort_index()

        elapsed_days = (
            prices.index[-1] - prices.index[0]
        ).total_seconds() / (24 * 60 * 60)

        if elapsed_days <= 0:
            return float("nan") if isinstance(prices, pd.Series) else pd.Series(
                np.nan, index=prices.columns
            )

        years = elapsed_days / 365.25
    else:
        years = len(prices) / trading_days

    initial_price = prices.iloc[0]
    final_price = prices.iloc[-1]

    cagr = (final_price / initial_price) ** (1.0 / years) - 1.0

    max_drawdown = calculate_max_drawdown(prices)

    # Handle Series and DataFrame outputs separately.
    if isinstance(max_drawdown, pd.Series):
        absolute_drawdown = max_drawdown.abs().replace(0, np.nan)
        calmar = cagr / absolute_drawdown
        return calmar.replace([np.inf, -np.inf], np.nan)

    absolute_drawdown = abs(float(max_drawdown))

    if not np.isfinite(absolute_drawdown) or absolute_drawdown == 0:
        return float("nan")

    calmar = cagr / absolute_drawdown

    if not np.isfinite(calmar):
        return float("nan")

    return float(calmar)