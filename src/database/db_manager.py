import json
import math
import sqlite3
from collections.abc import Mapping
from contextlib import closing
from pathlib import Path
from typing import Any


# Database file in the project root.
DB_PATH = Path(__file__).resolve().parent.parent.parent / "portfolio.db"

# SQL schema located next to this module.
SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


def get_connection() -> sqlite3.Connection:
    """Create and return a connection to the SQLite database."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_db() -> None:
    """Create the database tables if they do not already exist."""
    if not SCHEMA_PATH.is_file():
        raise FileNotFoundError(f"Database schema not found: {SCHEMA_PATH}")

    schema = SCHEMA_PATH.read_text(encoding="utf-8")

    with closing(get_connection()) as conn:
        conn.executescript(schema)


def _validate_text(value: str, field_name: str) -> None:
    """Ensure a required text field is a non-empty string."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")


def _validate_finite_number(value: Any, field_name: str) -> float:
    """Convert a numeric value to float and reject NaN or infinity."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite number.")

    number = float(value)

    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be a finite number.")

    return number


def save_optimized_portfolio(
    portfolio_type: str,
    weights: dict,
    expected_return: float,
    expected_vol: float,
    sharpe: float,
) -> None:
    """Validate and save an optimized portfolio."""
    _validate_text(portfolio_type, "portfolio_type")

    if not isinstance(weights, Mapping) or not weights:
        raise ValueError("weights must be a non-empty mapping.")

    validated_weights = {}

    for ticker, weight in weights.items():
        _validate_text(ticker, "ticker")
        numeric_weight = _validate_finite_number(weight, f"weight for {ticker}")

        if numeric_weight < 0:
            raise ValueError("Portfolio weights must be non-negative.")

        validated_weights[ticker] = numeric_weight

    total_weight = sum(validated_weights.values())

    if not math.isclose(total_weight, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("Portfolio weights must sum to 1.")

    expected_return = _validate_finite_number(
        expected_return, "expected_return"
    )
    expected_vol = _validate_finite_number(expected_vol, "expected_vol")
    sharpe = _validate_finite_number(sharpe, "sharpe")

    if expected_vol < 0:
        raise ValueError("expected_vol must be non-negative.")

    weights_json = json.dumps(validated_weights)

    with closing(get_connection()) as conn:
        with conn:
            conn.execute(
                """
                INSERT INTO optimized_portfolios (
                    portfolio_type,
                    weights_json,
                    expected_return,
                    expected_volatility,
                    sharpe_ratio
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    portfolio_type.strip(),
                    weights_json,
                    expected_return,
                    expected_vol,
                    sharpe,
                ),
            )


def save_ai_report(
    ticker: str,
    recommendation: str,
    macro_context: str,
    risk_assessment: str,
) -> None:
    """Validate and save an AI-generated report."""
    _validate_text(ticker, "ticker")
    _validate_text(recommendation, "recommendation")
    _validate_text(macro_context, "macro_context")
    _validate_text(risk_assessment, "risk_assessment")

    with closing(get_connection()) as conn:
        with conn:
            conn.execute(
                """
                INSERT INTO ai_reports (
                    ticker,
                    recommendation,
                    macro_context,
                    risk_assessment
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    ticker.strip().upper(),
                    recommendation.strip(),
                    macro_context.strip(),
                    risk_assessment.strip(),
                ),
            )