import numpy as np
import pandas as pd
import pytest

# Importamos las funciones desde tu módulo src.analytics.risk
from src.analytics.risk import (
    calculate_annualized_volatility,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_historical_var,
    calculate_historical_cvar,
    calculate_calmar_ratio,
)


# ==========================================
# FIXTURES (Datos de prueba)
# ==========================================
@pytest.fixture
def sample_returns():
    """Small sample of daily returns for testing."""
    return pd.Series(
        [0.01, -0.02, 0.03, -0.04, 0.02],
        name="TEST",
    )

@pytest.fixture
def sample_prices():
    """Price series with a known maximum drawdown."""
    return pd.Series(
        [100.0, 110.0, 99.0, 105.0, 95.0],
        name="TEST",
    )

@pytest.fixture
def multi_asset_returns():
    """Daily returns for two assets."""
    return pd.DataFrame(
        {
            "ASSET_A": [0.01, -0.02, 0.03, -0.04, 0.02],
            "ASSET_B": [0.02, -0.01, 0.01, -0.03, 0.04],
        }
    )

@pytest.fixture
def multi_asset_prices():
    """Price histories for two assets."""
    return pd.DataFrame(
        {
            "ASSET_A": [100.0, 110.0, 99.0, 105.0, 95.0],
            "ASSET_B": [100.0, 102.0, 101.0, 98.0, 103.0],
        }
    )


# ==========================================
# TESTS: VOLATILITY
# ==========================================
def test_annualized_volatility_for_series(sample_returns):
    expected = sample_returns.std(ddof=1) * np.sqrt(252)
    result = calculate_annualized_volatility(sample_returns)
    assert isinstance(result, float)
    assert result == pytest.approx(expected)

def test_annualized_volatility_for_dataframe(multi_asset_returns):
    expected = multi_asset_returns.std(ddof=1) * np.sqrt(252)
    result = calculate_annualized_volatility(multi_asset_returns)
    assert isinstance(result, pd.Series)
    pd.testing.assert_series_equal(result, expected)


# ==========================================
# TESTS: MAX DRAWDOWN
# ==========================================
def test_max_drawdown_for_series(sample_prices):
    # El pico es 110, el valle posterior es 95. (95/110 - 1)
    expected = 95.0 / 110.0 - 1.0
    result = calculate_max_drawdown(sample_prices)
    assert isinstance(result, float)
    assert result == pytest.approx(expected)

def test_max_drawdown_for_dataframe(multi_asset_prices):
    result = calculate_max_drawdown(multi_asset_prices)
    assert isinstance(result, pd.Series)
    assert result["ASSET_A"] == pytest.approx(95.0 / 110.0 - 1.0)
    # ASSET_B: pico 102, valle 98 -> (98/102 - 1)
    assert result["ASSET_B"] == pytest.approx(98.0 / 102.0 - 1.0)


# ==========================================
# TESTS: SHARPE RATIO
# ==========================================
def test_sharpe_ratio_for_series(sample_returns):
    expected = (sample_returns.mean() / sample_returns.std(ddof=1)) * np.sqrt(252)
    result = calculate_sharpe_ratio(sample_returns)
    assert isinstance(result, float)
    assert result == pytest.approx(expected)

def test_sharpe_ratio_with_risk_free_rate(sample_returns):
    risk_free_rate = 0.05
    daily_risk_free_rate = risk_free_rate / 252
    expected = (
        (sample_returns.mean() - daily_risk_free_rate)
        / sample_returns.std(ddof=1)
        * np.sqrt(252)
    )
    result = calculate_sharpe_ratio(sample_returns, risk_free_rate=risk_free_rate)
    assert result == pytest.approx(expected)

def test_sharpe_ratio_for_dataframe(multi_asset_returns):
    result = calculate_sharpe_ratio(multi_asset_returns)
    assert isinstance(result, pd.Series)
    for asset in multi_asset_returns.columns:
        expected = (
            multi_asset_returns[asset].mean()
            / multi_asset_returns[asset].std(ddof=1)
            * np.sqrt(252)
        )
        assert result[asset] == pytest.approx(expected)


# ==========================================
# TESTS: SORTINO RATIO (¡Corregido!)
# ==========================================
def test_sortino_ratio_for_series(sample_returns):
    """Test Sortino with a 0% target return."""
    result = calculate_sortino_ratio(sample_returns, target_return=0.0)
    
    # Cálculo manual paso a paso
    downside_returns = sample_returns.clip(upper=0.0)
    downside_deviation = np.sqrt((downside_returns**2).mean()) * np.sqrt(252)
    annualized_excess_return = sample_returns.mean() * 252
    expected = annualized_excess_return / downside_deviation
    
    assert isinstance(result, float)
    assert result == pytest.approx(expected)

def test_sortino_ratio_for_dataframe(multi_asset_returns):
    result = calculate_sortino_ratio(multi_asset_returns, target_return=0.0)
    assert isinstance(result, pd.Series)
    
    for asset in multi_asset_returns.columns:
        asset_returns = multi_asset_returns[asset]
        downside_returns = asset_returns.clip(upper=0.0)
        downside_deviation = np.sqrt((downside_returns**2).mean()) * np.sqrt(252)
        annualized_excess_return = asset_returns.mean() * 252
        expected = annualized_excess_return / downside_deviation
        assert result[asset] == pytest.approx(expected)


# ==========================================
# TESTS: VALUE AT RISK (VaR)
# ==========================================
def test_historical_var(sample_returns):
    # El percentil 5 de los retornos [0.01, -0.02, 0.03, -0.04, 0.02]
    # como son solo 5 datos, el quantile(0.05) va a ser muy cercano al mínimo (-0.04)
    expected = sample_returns.quantile(0.05)
    result = calculate_historical_var(sample_returns, confidence_level=0.95)
    assert result == pytest.approx(expected)

def test_historical_var_multi_asset(multi_asset_returns):
    result = calculate_historical_var(multi_asset_returns, confidence_level=0.95)
    assert isinstance(result, pd.Series)
    for asset in multi_asset_returns.columns:
        expected = multi_asset_returns[asset].quantile(0.05)
        assert result[asset] == pytest.approx(expected)


# ==========================================
# TESTS: CONDITIONAL VALUE AT RISK (CVaR)
# ==========================================
def test_historical_cvar(sample_returns):
    var_threshold = sample_returns.quantile(0.05)
    # CVaR es el promedio de los retornos que son menores o iguales al VaR
    tail_returns = sample_returns[sample_returns <= var_threshold]
    expected = tail_returns.mean()
    result = calculate_historical_cvar(sample_returns, confidence_level=0.95)
    assert result == pytest.approx(expected)

def test_historical_cvar_multi_asset(multi_asset_returns):
    result = calculate_historical_cvar(multi_asset_returns, confidence_level=0.95)
    assert isinstance(result, pd.Series)
    for asset in multi_asset_returns.columns:
        var_threshold = multi_asset_returns[asset].quantile(0.05)
        tail_returns = multi_asset_returns[asset][multi_asset_returns[asset] <= var_threshold]
        expected = tail_returns.mean()
        assert result[asset] == pytest.approx(expected)


# ==========================================
# TESTS: CALMAR RATIO
# ==========================================
def test_calmar_ratio(sample_prices):
    result = calculate_calmar_ratio(sample_prices)
    
    # Como sample_prices no tiene un DatetimeIndex, years = len(prices) / 252
    years = len(sample_prices) / 252
    total_growth = sample_prices.iloc[-1] / sample_prices.iloc[0]
    cagr = total_growth ** (1 / years) - 1
    
    max_dd = calculate_max_drawdown(sample_prices)
    expected = cagr / abs(max_dd)
    
    assert isinstance(result, float)
    assert result == pytest.approx(expected)

def test_calmar_ratio_multi_asset(multi_asset_prices):
    result = calculate_calmar_ratio(multi_asset_prices)
    assert isinstance(result, pd.Series)
    # Verificamos que devuelva un valor por cada activo
    assert len(result) == 2
    assert "ASSET_A" in result.index
    assert "ASSET_B" in result.index


# ==========================================
# TESTS: VALIDACIONES Y ERRORES
# ==========================================
def test_empty_data_raises_error():
    empty_series = pd.Series(dtype=float)
    with pytest.raises(ValueError, match="cannot be empty"):
        calculate_annualized_volatility(empty_series)

def test_nan_data_raises_error():
    nan_series = pd.Series([0.01, np.nan, 0.02])
    with pytest.raises(ValueError, match="missing values"):
        calculate_annualized_volatility(nan_series)