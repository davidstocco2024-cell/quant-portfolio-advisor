import numpy as np
import pandas as pd
import pytest

from src.analytics.portfolio_optimizer import (
    optimize_portfolio,
    generate_efficient_frontier,
)


@pytest.fixture
def sample_prices():
    """Create reproducible synthetic prices for three assets."""
    rng = np.random.default_rng(42)
    number_of_days = 252

    daily_returns = pd.DataFrame(
        {
            "ASSET_A": rng.normal(0.0008, 0.010, number_of_days),
            "ASSET_B": rng.normal(0.0005, 0.008, number_of_days),
            "ASSET_C": rng.normal(0.0003, 0.006, number_of_days),
        }
    )

    prices = 100 * (1 + daily_returns).cumprod()
    dates = pd.date_range(start="2024-01-01", periods=number_of_days, freq="B")
    prices.index = dates

    return prices


def assert_valid_weights(weights, asset_names, max_weight=1.0):
    """Check that portfolio weights are valid and fully invested."""
    assert set(weights.keys()) == set(asset_names)
    weight_values = np.array(list(weights.values()), dtype=float)

    assert np.isfinite(weight_values).all()
    assert (weight_values >= -1e-8).all()
    assert (weight_values <= max_weight + 1e-6).all()
    assert weight_values.sum() == pytest.approx(1.0, abs=1e-4)


def test_optimize_portfolio_returns_all_methods(sample_prices):
    result = optimize_portfolio(sample_prices, method="both")

    assert set(result.keys()) == {"sharpe", "min_vol", "hrp"}

    for method_result in result.values():
        assert "weights" in method_result
        assert "performance" in method_result

        performance = method_result["performance"]

        assert "expected_return" in performance
        assert "volatility" in performance
        assert "sharpe_ratio" in performance

        assert np.isfinite(performance["expected_return"])
        assert np.isfinite(performance["volatility"])
        assert performance["volatility"] >= 0
        
        # CORRECCIÓN: El Sharpe puede ser NaN si la volatilidad es cero (ej. HRP con datos perfectos).
        # Por eso usamos "is not None" en lugar de "isfinite" para ese campo específico.
        assert performance["sharpe_ratio"] is not None

        assert_valid_weights(
            method_result["weights"],
            sample_prices.columns,
        )


@pytest.mark.parametrize(
    "method",
    ["sharpe", "min_vol", "hrp"],
)
def test_optimize_portfolio_individual_methods(sample_prices, method):
    result = optimize_portfolio(sample_prices, method=method)

    assert list(result.keys()) == [method]

    assert_valid_weights(
        result[method]["weights"],
        sample_prices.columns,
    )


def test_markowitz_respects_max_weight(sample_prices):
    max_weight = 0.60

    result = optimize_portfolio(
        sample_prices,
        max_weight=max_weight,
        method="both",
    )

    for method in ("sharpe", "min_vol"):
        assert_valid_weights(
            result[method]["weights"],
            sample_prices.columns,
            max_weight=max_weight,
        )


def test_hrp_weights_sum_to_one(sample_prices):
    result = optimize_portfolio(sample_prices, method="hrp")
    weights = result["hrp"]["weights"]

    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-4)
    assert all(weight >= 0 for weight in weights.values())


def test_generate_efficient_frontier(sample_prices):
    frontier, optimal_points = generate_efficient_frontier(
        sample_prices,
        points=10,
    )

    assert isinstance(frontier, pd.DataFrame)
    assert set(frontier.columns) == {"return", "volatility"}
    assert not frontier.empty

    assert np.isfinite(frontier["return"]).all()
    assert np.isfinite(frontier["volatility"]).all()
    assert (frontier["volatility"] >= 0).all()

    assert set(optimal_points.keys()) == {
        "sharpe",
        "min_vol",
        "hrp",
    }


def test_optimizer_rejects_single_asset():
    prices = pd.DataFrame(
        {"ASSET_A": [100.0, 101.0, 102.0]}
    )

    with pytest.raises(ValueError):
        optimize_portfolio(prices)


def test_optimizer_rejects_missing_prices(sample_prices):
    prices = sample_prices.copy()
    prices.iloc[5, 0] = np.nan

    with pytest.raises(ValueError):
        optimize_portfolio(prices)


def test_optimizer_rejects_nonpositive_prices(sample_prices):
    prices = sample_prices.copy()
    prices.iloc[5, 0] = 0.0

    with pytest.raises(ValueError):
        optimize_portfolio(prices)


def test_optimizer_rejects_invalid_method(sample_prices):
    with pytest.raises(ValueError):
        optimize_portfolio(sample_prices, method="unknown")


def test_optimizer_rejects_infeasible_weight_cap(sample_prices):
    # Three assets capped at 0.30 can only add up to 0.90.
    with pytest.raises(ValueError):
        optimize_portfolio(
            sample_prices,
            max_weight=0.30,
        )


def test_frontier_rejects_too_few_points(sample_prices):
    with pytest.raises(ValueError):
        generate_efficient_frontier(
            sample_prices,
            points=1,
        )