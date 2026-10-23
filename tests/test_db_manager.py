import json
import sqlite3
from contextlib import closing

import pytest

from src.database import db_manager


@pytest.fixture
def test_db(tmp_path, monkeypatch):
    """Use an isolated SQLite database for each test."""
    test_db_path = tmp_path / "test_portfolio.db"
    monkeypatch.setattr(db_manager, "DB_PATH", test_db_path)

    db_manager.init_db()

    return test_db_path


def test_init_db_creates_tables(test_db):
    with closing(sqlite3.connect(test_db)) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }

    assert "optimized_portfolios" in tables
    assert "ai_reports" in tables


def test_save_optimized_portfolio(test_db):
    weights = {"AAPL": 0.6, "MSFT": 0.4}

    db_manager.save_optimized_portfolio(
        portfolio_type="Max Sharpe",
        weights=weights,
        expected_return=0.12,
        expected_vol=0.18,
        sharpe=0.67,
    )

    with closing(sqlite3.connect(test_db)) as conn:
        row = conn.execute(
            """
            SELECT portfolio_type, weights_json,
                   expected_return, expected_volatility, sharpe_ratio
            FROM optimized_portfolios
            """
        ).fetchone()

    assert row[0] == "Max Sharpe"
    assert json.loads(row[1]) == weights
    assert row[2:] == (0.12, 0.18, 0.67)


def test_save_ai_report(test_db):
    db_manager.save_ai_report(
        ticker="aapl",
        recommendation="Hold",
        macro_context="Inflation remains elevated.",
        risk_assessment="Moderate volatility.",
    )

    with closing(sqlite3.connect(test_db)) as conn:
        row = conn.execute(
            """
            SELECT ticker, recommendation, macro_context, risk_assessment
            FROM ai_reports
            """
        ).fetchone()

    assert row == (
        "AAPL",
        "Hold",
        "Inflation remains elevated.",
        "Moderate volatility.",
    )


@pytest.mark.parametrize(
    "weights",
    [
        {},
        {"AAPL": 0.7, "MSFT": 0.2},
        {"AAPL": 1.1, "MSFT": -0.1},
        {"AAPL": float("nan")},
        {"AAPL": float("inf")},
    ],
)
def test_save_portfolio_rejects_invalid_weights(test_db, weights):
    with pytest.raises(ValueError):
        db_manager.save_optimized_portfolio(
            "Max Sharpe",
            weights,
            expected_return=0.1,
            expected_vol=0.2,
            sharpe=0.5,
        )


@pytest.mark.parametrize(
    "expected_vol",
    [-0.1, float("nan"), float("inf")],
)
def test_save_portfolio_rejects_invalid_volatility(test_db, expected_vol):
    with pytest.raises(ValueError):
        db_manager.save_optimized_portfolio(
            "Max Sharpe",
            {"AAPL": 1.0},
            expected_return=0.1,
            expected_vol=expected_vol,
            sharpe=0.5,
        )


def test_save_portfolio_rejects_empty_type(test_db):
    with pytest.raises(ValueError):
        db_manager.save_optimized_portfolio(
            "",
            {"AAPL": 1.0},
            expected_return=0.1,
            expected_vol=0.2,
            sharpe=0.5,
        )


def test_save_ai_report_rejects_empty_required_field(test_db):
    with pytest.raises(ValueError):
        db_manager.save_ai_report(
            ticker="AAPL",
            recommendation="",
            macro_context="Macro context",
            risk_assessment="Risk assessment",
        )