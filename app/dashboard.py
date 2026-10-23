import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Optional, Tuple

# Import our custom modules
from src.data.market_data import fetch_market_data
from src.analytics.metrics import (
    calculate_daily_returns,
    calculate_cumulative_returns,
    calculate_annualized_return,
    calculate_tracking_error,
    calculate_information_ratio,
    calculate_downside_capture,
)
from src.analytics.risk import (
    calculate_annualized_volatility,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_value_at_risk,
    calculate_conditional_value_at_risk,
    calculate_calmar_ratio,
)
from src.analytics.portfolio_optimizer import optimize_portfolio, generate_efficient_frontier
from src.analytics.ai_analyst import generate_market_context
from src.analytics.backtester import run_backtest
from src.database.db_manager import save_ai_report

# =============================================================================
# Page Config
# =============================================================================
st.set_page_config(
    page_title="Quant Portfolio Advisor",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("📊 Quant Portfolio Advisor")
st.markdown("Professional Risk & Portfolio Optimization Dashboard")

# =============================================================================
# Helpers
# =============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def load_prices(tickers: tuple, start: str, end: str) -> pd.DataFrame:
    """Cached market data loader. Tickers must be a tuple for hashing."""
    return fetch_market_data(list(tickers), start, end)


def safe_get_weights(opt_results: Dict[str, Any], key: str) -> Optional[Dict[str, float]]:
    """Return weights dict if the optimization succeeded, else None."""
    section = opt_results.get(key, {})
    if "error" in section or "weights" not in section:
        return None
    return section["weights"]


def portfolio_returns_from_weights(
    daily_returns: pd.DataFrame, weights: Dict[str, float]
) -> pd.Series:
    """Compute portfolio daily returns from a weights dict (robust alignment)."""
    w = pd.Series(weights)
    common = daily_returns.columns.intersection(w.index)
    if len(common) == 0:
        raise ValueError("No overlapping assets between weights and returns.")
    return (daily_returns[common] * w[common]).sum(axis=1)


# =============================================================================
# Sidebar
# =============================================================================
st.sidebar.header("Configuration")

AVAILABLE_ASSETS = [
    "AAPL", "MSFT", "YPF", "BTC-USD", "^GSPC", "MELI",
    "GGAL.BA", "AMD", "PLTR", "GOOGL", "AMZN", "TSLA",
]

tickers = st.sidebar.multiselect(
    "Select Assets",
    AVAILABLE_ASSETS,
    default=["AAPL", "MSFT", "YPF", "BTC-USD", "^GSPC"],
)

benchmark = st.sidebar.selectbox(
    "Benchmark",
    options=AVAILABLE_ASSETS,
    index=AVAILABLE_ASSETS.index("^GSPC") if "^GSPC" in AVAILABLE_ASSETS else 0,
    help="Used for backtest and relative metrics (Tracking Error, IR, etc.)",
)

start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2022-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-01-01"))

risk_free_rate = st.sidebar.number_input(
    "Risk-free rate (annual)",
    min_value=0.0,
    max_value=0.20,
    value=0.05,
    step=0.005,
    format="%.3f",
)

max_weight = st.sidebar.slider(
    "Max weight per asset",
    min_value=0.10,
    max_value=1.0,
    value=0.40,
    step=0.05,
    help="Constraint to avoid extreme concentration in Markowitz portfolios.",
)

# =============================================================================
# Validation & Data Loading
# =============================================================================
if len(tickers) < 2:
    st.warning("Please select at least 2 assets to build a portfolio.")
    st.stop()

if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()

# Always include the benchmark so backtest / relative metrics never fail
tickers_to_fetch = list(dict.fromkeys(tickers + [benchmark]))  # preserve order, unique

try:
    with st.spinner("Fetching market data..."):
        prices = load_prices(
            tuple(tickers_to_fetch),
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
        )
except Exception as e:
    st.error(f"Failed to fetch market data: {e}")
    st.info("Check ticker symbols, date range, or your internet connection.")
    st.stop()

if prices.empty or prices.isna().all().all():
    st.error("No price data returned. Try different tickers or dates.")
    st.stop()

# Keep only the user-selected assets for most calculations
# (benchmark is kept separately when needed)
user_prices = prices[tickers].dropna(how="all")
daily_returns = calculate_daily_returns(user_prices, method="simple")
cumulative_returns = calculate_cumulative_returns(daily_returns)

# Full returns (includes benchmark) for relative metrics / backtest
full_daily_returns = calculate_daily_returns(prices, method="simple")

# =============================================================================
# 1. Cumulative Returns
# =============================================================================
st.subheader("Cumulative Returns")
fig = px.line(
    cumulative_returns,
    x=cumulative_returns.index,
    y=cumulative_returns.columns,
    title="Performance of $1 Invested",
    labels={"value": "Growth of $1", "index": "Date"},
)
fig.update_layout(hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# 2. Correlation Matrix
# =============================================================================
st.subheader("Correlation Matrix")
st.markdown(
    "Values close to **1** mean assets move together (lower diversification benefit). "
    "Values close to **-1** mean they move in opposite directions."
)

corr_matrix = daily_returns.corr()
fig_corr = go.Figure(
    data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale="RdBu",
        zmin=-1,
        zmax=1,
        text=corr_matrix.round(2).values,
        texttemplate="%{text}",
        showscale=True,
    )
)
fig_corr.update_layout(title="Asset Correlation Heatmap", height=500)
st.plotly_chart(fig_corr, use_container_width=True)

# =============================================================================
# 3. Individual Asset Metrics
# =============================================================================
st.subheader("Individual Asset Metrics")
metrics_df = pd.DataFrame(
    {
        "Annualized Volatility": calculate_annualized_volatility(daily_returns),
        "Max Drawdown": calculate_max_drawdown(user_prices),
        "Sharpe Ratio": calculate_sharpe_ratio(daily_returns, risk_free_rate=risk_free_rate),
        "Sortino Ratio": calculate_sortino_ratio(daily_returns, risk_free_rate=risk_free_rate),
        "VaR (95%)": calculate_value_at_risk(daily_returns),
        "CVaR (95%)": calculate_conditional_value_at_risk(daily_returns),
        "Calmar Ratio": calculate_calmar_ratio(user_prices),
    }
).round(4)
st.dataframe(metrics_df, use_container_width=True)

# =============================================================================
# 4. Portfolio Optimization
# =============================================================================
st.markdown("---")
st.subheader("Markowitz Efficient Frontier Optimization")

run_opt = st.button("Run Optimization", type="primary")

if run_opt:
    with st.spinner("Optimizing portfolio (Max Sharpe, Min Vol, HRP)..."):
        try:
            opt_results = optimize_portfolio(
                user_prices,
                risk_free_rate=risk_free_rate,
                max_weight=max_weight,
                method="both",  # assumes your function supports this
            )
            st.session_state["opt_results"] = opt_results
            st.session_state["opt_tickers"] = tickers  # track which assets were used
            st.session_state["opt_benchmark"] = benchmark
        except Exception as e:
            st.error(f"Optimization failed: {e}")
            st.stop()

# Use cached results if available and still relevant
opt_results = st.session_state.get("opt_results")
if opt_results is None:
    st.info("Click **Run Optimization** to compute optimal portfolios.")
else:
    # Warn if tickers changed since last run
    if st.session_state.get("opt_tickers") != tickers:
        st.warning(
            "Asset selection changed since last optimization. "
            "Click **Run Optimization** again to refresh results."
        )

    # ----- Max Sharpe -----
    sharpe_weights = safe_get_weights(opt_results, "sharpe")
    if sharpe_weights:
        st.markdown("### 🏆 Maximum Sharpe Ratio Portfolio")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Optimal Weights:**")
            st.json({k: round(v, 4) for k, v in sharpe_weights.items()})
        with col2:
            st.write("**Expected Performance:**")
            perf = opt_results["sharpe"]["performance"]
            st.metric("Expected Return", f"{perf['expected_return'] * 100:.2f}%")
            st.metric("Expected Volatility", f"{perf['volatility'] * 100:.2f}%")
            st.metric("Sharpe Ratio", f"{perf['sharpe_ratio']:.4f}")

    # ----- Min Volatility -----
    min_vol_weights = safe_get_weights(opt_results, "min_vol")
    if min_vol_weights:
        st.markdown("### 🛡️ Minimum Volatility Portfolio")
        col3, col4 = st.columns(2)
        with col3:
            st.write("**Optimal Weights:**")
            st.json({k: round(v, 4) for k, v in min_vol_weights.items()})
        with col4:
            st.write("**Expected Performance:**")
            perf_min = opt_results["min_vol"]["performance"]
            st.metric("Expected Return", f"{perf_min['expected_return'] * 100:.2f}%")
            st.metric("Expected Volatility", f"{perf_min['volatility'] * 100:.2f}%")
            st.metric("Sharpe Ratio", f"{perf_min['sharpe_ratio']:.4f}")

    # ----- HRP -----
    hrp_weights = safe_get_weights(opt_results, "hrp")
    if hrp_weights:
        st.markdown("### ⚖️ Hierarchical Risk Parity (HRP)")
        st.markdown("*More robust than classic Markowitz during market stress.*")
        col_hrp1, col_hrp2 = st.columns(2)
        with col_hrp1:
            st.write("**HRP Weights:**")
            st.json({k: round(v, 4) for k, v in hrp_weights.items()})
        with col_hrp2:
            st.write("**Expected Performance:**")
            perf_hrp = opt_results["hrp"]["performance"]
            st.metric("Expected Return", f"{perf_hrp['expected_return'] * 100:.2f}%")
            st.metric("Expected Volatility", f"{perf_hrp['volatility'] * 100:.2f}%")
            st.metric("Sharpe Ratio", f"{perf_hrp['sharpe_ratio']:.4f}")

    # =============================================================================
    # 5. Backtest
    # =============================================================================
    st.markdown("---")
    st.markdown("### 📈 Backtest: Optimized Portfolio vs Benchmark")
    st.warning(
        "⚠️ **Disclaimer:** This is an *in-sample* backtest. "
        "Weights were optimized on the same period shown. "
        "Real-world (out-of-sample) performance may differ significantly."
    )
    st.markdown(
        f"Simulating a **$10,000** investment with monthly rebalancing "
        f"(~21 trading days) and **0.1%** transaction costs. Benchmark: **{benchmark}**."
    )

    if sharpe_weights and sum(sharpe_weights.values()) > 0:
        try:
            backtest_df, backtest_metrics = run_backtest(
                prices=prices,  # full prices (includes benchmark)
                optimized_weights=sharpe_weights,
                initial_capital=10_000.0,
                rebalance_freq=21,
                transaction_cost=0.001,
                benchmark_ticker=benchmark,
            )

            st.markdown("#### Backtest Performance Metrics")
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric(
                    "Portfolio Total Return",
                    f"{backtest_metrics['Portfolio']['Total Return'] * 100:.2f}%",
                )
            with col_m2:
                st.metric(
                    "Benchmark Total Return",
                    f"{backtest_metrics['Benchmark']['Total Return'] * 100:.2f}%",
                )
            with col_m3:
                st.metric(
                    "Avg Turnover",
                    f"{backtest_metrics['Avg Turnover'] * 100:.2f}%",
                )

            fig_bt = go.Figure()
            fig_bt.add_trace(
                go.Scatter(
                    x=backtest_df.index,
                    y=backtest_df["Optimized Portfolio"],
                    mode="lines",
                    name="Optimized Portfolio (Max Sharpe)",
                    line=dict(color="#00CC96", width=2),
                )
            )
            fig_bt.add_trace(
                go.Scatter(
                    x=backtest_df.index,
                    y=backtest_df["Benchmark"],
                    mode="lines",
                    name=f"Benchmark ({benchmark})",
                    line=dict(color="#EF553B", width=2),
                )
            )
            fig_bt.update_layout(
                title="Portfolio Value Over Time ($10,000 Initial)",
                xaxis_title="Date",
                yaxis_title="Portfolio Value ($)",
                hovermode="x unified",
            )
            st.plotly_chart(fig_bt, use_container_width=True)

        except Exception as e:
            st.error(f"Backtest failed: {e}")

    # =============================================================================
    # 6. Portfolio vs Benchmark Analytics
    # =============================================================================
    st.markdown("---")
    st.markdown("### 🎯 Portfolio vs Benchmark Analysis")

    if sharpe_weights:
        try:
            portfolio_daily_returns = portfolio_returns_from_weights(
                full_daily_returns, sharpe_weights
            )
            if benchmark not in full_daily_returns.columns:
                st.warning(f"Benchmark {benchmark} not found in returns data.")
            else:
                benchmark_daily_returns = full_daily_returns[benchmark]

                col_p1, col_p2, col_p3, col_p4 = st.columns(4)
                with col_p1:
                    ann_ret = calculate_annualized_return(portfolio_daily_returns)
                    st.metric("Portfolio Ann. Return", f"{ann_ret * 100:.2f}%")
                with col_p2:
                    te = calculate_tracking_error(
                        portfolio_daily_returns, benchmark_daily_returns
                    )
                    st.metric("Tracking Error", f"{te * 100:.2f}%")
                with col_p3:
                    ir = calculate_information_ratio(
                        portfolio_daily_returns, benchmark_daily_returns
                    )
                    st.metric("Information Ratio", f"{ir:.2f}")
                with col_p4:
                    dc = calculate_downside_capture(
                        portfolio_daily_returns, benchmark_daily_returns
                    )
                    st.metric("Downside Capture", f"{dc * 100:.2f}%")
        except Exception as e:
            st.error(f"Could not compute relative metrics: {e}")

    # =============================================================================
    # 7. Efficient Frontier
    # =============================================================================
    st.markdown("---")
    st.markdown("### 📉 Efficient Frontier")
    st.markdown(
        "The curve shows the set of optimal portfolios that offer the highest "
        "expected return for each level of risk."
    )

    with st.spinner("Calculating Efficient Frontier..."):
        try:
            frontier_df, _ = generate_efficient_frontier(
                user_prices,
                risk_free_rate=risk_free_rate,
                max_weight=max_weight,
                points=40,
            )

            fig_ef = go.Figure()
            fig_ef.add_trace(
                go.Scatter(
                    x=frontier_df["volatility"],
                    y=frontier_df["return"],
                    mode="lines+markers",
                    name="Efficient Frontier",
                    line=dict(color="#636EFA", width=2),
                    marker=dict(size=5),
                )
            )

            # Highlight key portfolios if available
            def add_portfolio_point(weights_key: str, name: str, color: str, symbol: str):
                section = opt_results.get(weights_key, {})
                if "performance" in section and "error" not in section:
                    p = section["performance"]
                    fig_ef.add_trace(
                        go.Scatter(
                            x=[p["volatility"]],
                            y=[p["expected_return"]],
                            mode="markers",
                            name=name,
                            marker=dict(size=14, color=color, symbol=symbol),
                        )
                    )

            add_portfolio_point("sharpe", "Max Sharpe", "#00CC96", "star")
            add_portfolio_point("min_vol", "Min Volatility", "#AB63FA", "diamond")
            add_portfolio_point("hrp", "HRP", "#FFA15A", "circle")

            fig_ef.update_layout(
                title="Efficient Frontier (Risk vs Return)",
                xaxis_title="Annualized Volatility (Risk)",
                yaxis_title="Expected Annual Return",
                hovermode="closest",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_ef, use_container_width=True)
        except Exception as e:
            st.error(f"Efficient Frontier calculation failed: {e}")

    # =============================================================================
    # 8. Download
    # =============================================================================
    st.markdown("### 📥 Download Data")
    if sharpe_weights:
        weights_df = pd.DataFrame(
            list(sharpe_weights.items()), columns=["Asset", "Weight"]
        )
        weights_df["Weight (%)"] = (weights_df["Weight"] * 100).round(2)
        weights_df = weights_df.drop(columns=["Weight"])

        csv = weights_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Optimal Weights (CSV)",
            data=csv,
            file_name="optimal_portfolio_weights.csv",
            mime="text/csv",
        )

# =============================================================================
# 9. AI Fundamental Analysis
# =============================================================================
st.markdown("---")
st.subheader("🤖 AI Fundamental Analysis")
st.markdown("Select an asset to generate an AI-powered financial context report.")

selected_asset = st.selectbox("Choose an asset for AI analysis:", tickers)

if st.button(f"Generate AI Report for {selected_asset}"):
    with st.spinner("Analyzing market context..."):
        try:
            asset_metrics = metrics_df.loc[selected_asset].to_dict()
            ai_report = generate_market_context(selected_asset, asset_metrics)

            if "error" in ai_report:
                st.error(ai_report["error"])
            else:
                st.info(f"**Recommendation:** {ai_report['recommendation']}")

                col_ai1, col_ai2 = st.columns(2)
                with col_ai1:
                    st.markdown("**🌍 Macro Context**")
                    st.write(ai_report["macro_context"])
                with col_ai2:
                    st.markdown("**⚠️ Risk Assessment**")
                    st.write(ai_report["risk_assessment"])

                # Optional: ask before saving
                if st.checkbox("Save this report to the database", value=True):
                    save_ai_report(
                        ticker=selected_asset,
                        recommendation=ai_report["recommendation"],
                        macro_context=ai_report["macro_context"],
                        risk_assessment=ai_report["risk_assessment"],
                    )
                    st.success("✅ Report saved to database.")
        except Exception as e:
            st.error(f"AI analysis failed: {e}")