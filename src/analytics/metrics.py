
import numpy as np
import pandas as pd


def calculate_cumulative_returns(
    daily_returns: pd.DataFrame | pd.Series,
) -> pd.DataFrame | pd.Series:
    """
    Calculates cumulative returns from simple periodic returns.

    Args:
        daily_returns: Simple returns, not logarithmic returns.

    Returns:
        Cumulative returns over time.
    """
    if daily_returns.empty:
        raise ValueError("daily_returns cannot be empty.")

    if daily_returns.isna().any().any():
        raise ValueError("daily_returns contains missing values.")

    if (daily_returns < -1).any().any():
        raise ValueError("Simple returns cannot be below -100%.")

    return (1 + daily_returns).cumprod() - 1




def calculate_annualized_return(
    returns: pd.DataFrame | pd.Series,
    periods_per_year: int = 252,
) -> pd.DataFrame | pd.Series | float:
    """
    Calculates the compound annualized return (CAGR).

    Assumes equally spaced simple returns.
    """
    if returns.empty:
        raise ValueError("returns cannot be empty.")

    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive.")

    if returns.isna().any().any():
        raise ValueError("returns contains missing values.")

    if (returns < -1).any().any():
        raise ValueError("Simple returns cannot be below -100%.")

    number_of_years = len(returns) / periods_per_year

    if number_of_years <= 0:
        raise ValueError("The observation period must be positive.")

    # Convert returns to a NumPy array for type-safe calculations.
    values = returns.to_numpy(dtype=float)

    # Calculate compounded growth along the time axis.
    growth_factor = np.prod(1 + values, axis=0)

    annualized = np.power(
        growth_factor,
        1 / number_of_years,
    ) - 1

    # Return a scalar for a Series and a Series for a DataFrame.
    if isinstance(returns, pd.Series):
        return float(annualized)

    return pd.Series(
        annualized,
        index=returns.columns,
        name="annualized_return",
    )



def _align_returns(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> pd.DataFrame:
    """
    Aligns portfolio and benchmark returns by date.

    Keeps only dates with valid observations for both series.
    """
    if not isinstance(portfolio_returns, pd.Series):
        raise TypeError("portfolio_returns must be a pandas Series.")

    if not isinstance(benchmark_returns, pd.Series):
        raise TypeError("benchmark_returns must be a pandas Series.")

    aligned = pd.concat(
        [
            portfolio_returns.rename("portfolio"),
            benchmark_returns.rename("benchmark"),
        ],
        axis=1,
        join="inner",
    )

    aligned = aligned.dropna()

    if aligned.empty:
        raise ValueError(
            "No overlapping valid observations between portfolio and benchmark."
        )

    return aligned


def calculate_tracking_error(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """
    Calculates annualized tracking error.

    Tracking error is the standard deviation of active returns.
    """
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive.")

    aligned = _align_returns(
        portfolio_returns,
        benchmark_returns,
    )

    active_returns = (
        aligned["portfolio"] - aligned["benchmark"]
    )

    return float(
        active_returns.std(ddof=1) * np.sqrt(periods_per_year)
    )


def calculate_information_ratio(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """
    Calculates the annualized information ratio.

    Uses arithmetic annualized active return divided by
    annualized tracking error.
    """
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive.")

    aligned = _align_returns(
        portfolio_returns,
        benchmark_returns,
    )

    active_returns = (
        aligned["portfolio"] - aligned["benchmark"]
    )

    tracking_error = active_returns.std(ddof=1)

    if tracking_error == 0 or np.isnan(tracking_error):
        return np.nan

    annualized_active_return = (
        active_returns.mean() * periods_per_year
    )

    annualized_tracking_error = (
        tracking_error * np.sqrt(periods_per_year)
    )

    return float(
        annualized_active_return / annualized_tracking_error
    )


def calculate_downside_capture(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """
    Calculates downside capture during periods when
    the benchmark return is negative.

    A value below 1 generally indicates a smaller average
    loss than the benchmark during down-market periods.
    """
    aligned = _align_returns(
        portfolio_returns,
        benchmark_returns,
    )

    down_market = aligned["benchmark"] < 0

    if not down_market.any():
        return np.nan

    portfolio_downside_mean = (
        aligned.loc[down_market, "portfolio"].mean()
    )

    benchmark_downside_mean = (
        aligned.loc[down_market, "benchmark"].mean()
    )

    if benchmark_downside_mean == 0:
        return np.nan

    return float(
        portfolio_downside_mean / benchmark_downside_mean
    )
