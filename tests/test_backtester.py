import numpy as np
import pandas as pd
import pytest

from src.analytics.backtester import run_backtest


@pytest.fixture
def sample_prices():
    """Deterministic price history for two assets and a benchmark."""
    dates = pd.date_range("2024-01-01", periods=7, freq="B")

    return pd.DataFrame(
        {
            "AAA": [100, 102, 101, 103, 104, 102, 105],
            "BBB": [100, 99, 101, 100, 102, 104, 103],
            "SPY": [100, 101, 102, 101, 103, 104, 105],
        },
        index=dates,
        dtype=float,
    )


@pytest.fixture
def target_weights():
    return {"AAA": 0.60, "BBB": 0.40}


def test_backtest_returns_expected_columns_and_dates(
    sample_prices,
    target_weights,
):
    backtest, metrics = run_backtest(
        sample_prices,
        target_weights,
        benchmark_ticker="SPY",
    )

    assert list(backtest.columns) == [
        "Optimized Portfolio",
        "Benchmark",
    ]

    # One initial-capital observation plus one for each daily return.
    assert len(backtest) == len(sample_prices)
    assert backtest.index.equals(sample_prices.index)

    assert backtest.iloc[0]["Optimized Portfolio"] == pytest.approx(
        10000.0
    )
    assert backtest.iloc[0]["Benchmark"] == pytest.approx(10000.0)

    assert "Portfolio" in metrics
    assert "Benchmark" in metrics


def test_backtest_values_are_finite_and_positive(
    sample_prices,
    target_weights,
):
    backtest, _ = run_backtest(
        sample_prices,
        target_weights,
        benchmark_ticker="SPY",
    )

    assert np.isfinite(backtest.to_numpy()).all()
    assert (backtest > 0).all().all()


def test_backtest_without_transaction_costs(
    sample_prices,
    target_weights,
):
    backtest, metrics = run_backtest(
        sample_prices,
        target_weights,
        transaction_cost=0.0,
        rebalance_freq=2,
        benchmark_ticker="SPY",
    )

    assert backtest.iloc[-1]["Optimized Portfolio"] > 0
    assert metrics["Num_Rebalances"] > 0


def test_transaction_costs_do_not_increase_portfolio_value(
    sample_prices,
    target_weights,
):
    no_cost_backtest, _ = run_backtest(
        sample_prices,
        target_weights,
        transaction_cost=0.0,
        rebalance_freq=2,
        benchmark_ticker="SPY",
    )

    cost_backtest, _ = run_backtest(
        sample_prices,
        target_weights,
        transaction_cost=0.01,
        rebalance_freq=2,
        benchmark_ticker="SPY",
    )

    assert (
        cost_backtest["Optimized Portfolio"].iloc[-1]
        <= no_cost_backtest["Optimized Portfolio"].iloc[-1]
    )


def test_backtest_reports_rebalances(
    sample_prices,
    target_weights,
):
    _, metrics = run_backtest(
        sample_prices,
        target_weights,
        rebalance_freq=2,
        transaction_cost=0.0,
        benchmark_ticker="SPY",
    )

    # Six daily return observations: rebalance after observations 2, 4, 6.
    assert metrics["Num_Rebalances"] == 3
    assert len(metrics["Rebalance Dates"]) == 3


def test_weights_are_normalized(
    sample_prices,
):
    weights = {"AAA": 3.0, "BBB": 2.0}

    backtest, _ = run_backtest(
        sample_prices,
        weights,
        benchmark_ticker="SPY",
        transaction_cost=0.0,
    )

    # A valid run should produce a finite portfolio value.
    assert np.isfinite(backtest["Optimized Portfolio"].iloc[-1])


def test_backtest_rejects_missing_benchmark(
    sample_prices,
    target_weights,
):
    prices = sample_prices.drop(columns="SPY")

    with pytest.raises(ValueError, match="Benchmark"):
        run_backtest(
            prices,
            target_weights,
            benchmark_ticker="SPY",
        )


def test_backtest_rejects_weights_without_matching_assets(
    sample_prices,
):
    with pytest.raises(ValueError, match="No valid portfolio assets"):
        run_backtest(
            sample_prices,
            {"UNKNOWN": 1.0},
            benchmark_ticker="SPY",
        )


@pytest.mark.parametrize(
    "initial_capital",
    [0, -100, float("nan")],
)
def test_backtest_rejects_invalid_initial_capital(
    sample_prices,
    target_weights,
    initial_capital,
):
    with pytest.raises(ValueError):
        run_backtest(
            sample_prices,
            target_weights,
            initial_capital=initial_capital,
            benchmark_ticker="SPY",
        )


@pytest.mark.parametrize(
    "rebalance_freq",
    [0, -1],
)
def test_backtest_rejects_invalid_rebalance_frequency(
    sample_prices,
    target_weights,
    rebalance_freq,
):
    with pytest.raises(ValueError):
        run_backtest(
            sample_prices,
            target_weights,
            rebalance_freq=rebalance_freq,
            benchmark_ticker="SPY",
        )


def test_backtest_rejects_negative_weights(sample_prices):
    with pytest.raises(ValueError, match="long-only"):
        run_backtest(
            sample_prices,
            {"AAA": 1.2, "BBB": -0.2},
            benchmark_ticker="SPY",
        )


def test_backtest_rejects_nan_prices(
    sample_prices,
    target_weights,
):
    prices = sample_prices.copy()
    prices.iloc[2, 0] = np.nan

    with pytest.raises(ValueError, match="missing values"):
        run_backtest(
            prices,
            target_weights,
            benchmark_ticker="SPY",
        )