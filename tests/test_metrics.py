
import numpy as np
import pandas as pd

from src.analytics.metrics import (
    calculate_cumulative_returns,
    calculate_annualized_return,
    calculate_tracking_error,
    calculate_information_ratio,
    calculate_downside_capture,
)


# Test cumulative returns
simple_returns = pd.Series([0.10, -0.10])

cumulative = calculate_cumulative_returns(simple_returns)

expected_cumulative = (1.10 * 0.90) - 1

assert np.isclose(
    cumulative.iloc[-1],
    expected_cumulative,
)


# Test annualized return
returns = pd.Series([0.01] * 252)

annualized = calculate_annualized_return(returns)

assert np.isclose(annualized, (1.01 ** 252) - 1)


# Test benchmark-relative metrics
dates = pd.date_range(
    "2025-01-01",
    periods=5,
    freq="D",
)

portfolio = pd.Series(
    [0.02, -0.01, 0.03, -0.02, 0.01],
    index=dates,
)

benchmark = pd.Series(
    [0.01, -0.02, 0.02, -0.01, 0.01],
    index=dates,
)

tracking_error = calculate_tracking_error(
    portfolio,
    benchmark,
)

information_ratio = calculate_information_ratio(
    portfolio,
    benchmark,
)

downside_capture = calculate_downside_capture(
    portfolio,
    benchmark,
)

print("Tracking error:", tracking_error)
print("Information ratio:", information_ratio)
print("Downside capture:", downside_capture)

assert tracking_error >= 0
assert np.isfinite(information_ratio)
assert np.isfinite(downside_capture)

print("All metrics tests passed.")
