import pandas as pd

# Import our custom modules
from src.data.market_data import fetch_market_data
from src.analytics.metrics import calculate_daily_returns, calculate_cumulative_returns
from src.analytics.risk import (
    calculate_annualized_volatility, 
    calculate_max_drawdown, 
    calculate_sharpe_ratio, 
    calculate_sortino_ratio,
    calculate_value_at_risk
)
from src.analytics.portfolio import (
    calculate_portfolio_returns,
    calculate_beta,
    calculate_alpha,
    calculate_correlation_matrix
)
from src.analytics.portfolio_optimizer import optimize_portfolio

if __name__ == "__main__":
    # ==========================================
    # 1. CONFIGURATION
    # ==========================================
    tickers = ["^GSPC", "MSFT", "AAPL", "YPF", "BTC-USD"]
    benchmark = "^GSPC"
    start_date = "2022-01-01"
    end_date = "2024-01-01"
    
    # Tactical portfolio weights
    weights = {
        "^GSPC": 0.40,
        "MSFT": 0.20,
        "AAPL": 0.10,
        "YPF": 0.10,
        "BTC-USD": 0.20
    }
    
    # ==========================================
    # 2. DATA INGESTION
    # ==========================================
    prices = fetch_market_data(tickers, start_date, end_date)
    
    # ==========================================
    # 3. RETURNS CALCULATION
    # ==========================================
    daily_returns = calculate_daily_returns(prices, method='simple')
    port_returns = calculate_portfolio_returns(daily_returns, weights)
    port_returns_df = port_returns.to_frame()
    
    # ==========================================
    # 4. INDIVIDUAL RISK METRICS
    # ==========================================
    print("\n" + "="*60)
    print("INDIVIDUAL ASSET RISK & PERFORMANCE METRICS")
    print("="*60)
    
    metrics_df = pd.DataFrame({
        'Annualized Vol': calculate_annualized_volatility(daily_returns),
        'Max Drawdown': calculate_max_drawdown(prices),
        'Sharpe Ratio': calculate_sharpe_ratio(daily_returns),
        'Sortino Ratio': calculate_sortino_ratio(daily_returns),
        'VaR (95%)': calculate_value_at_risk(daily_returns, 0.95)
    })
    print(metrics_df.round(4).to_string())
    
    # ==========================================
    # 5. PORTFOLIO RISK METRICS (TACTICAL)
    # ==========================================
    print("\n" + "="*60)
    print("TACTICAL PORTFOLIO METRICS")
    print("="*60)
    
    port_vol = calculate_annualized_volatility(port_returns_df)['Portfolio']
    port_sharpe = calculate_sharpe_ratio(port_returns_df)['Portfolio']
    port_sortino = calculate_sortino_ratio(port_returns_df)['Portfolio']
    
    bench_returns = daily_returns[benchmark]
    port_beta = calculate_beta(port_returns, bench_returns)
    port_alpha = calculate_alpha(port_returns, bench_returns, port_beta)
    
    print(f"Annualized Volatility: {port_vol:.4f}")
    print(f"Sharpe Ratio:          {port_sharpe:.4f}")
    print(f"Sortino Ratio:         {port_sortino:.4f}")
    print(f"Beta (vs S&P 500):     {port_beta:.4f}")
    print(f"Alpha (Jensen's):      {port_alpha:.4f}")
    
    port_cum_ret = calculate_cumulative_returns(port_returns_df)
    bench_cum_ret = calculate_cumulative_returns(daily_returns[[benchmark]])
    print(f"Portfolio Return:      {(port_cum_ret['Portfolio'].iloc[-1] * 100):.2f}%")
    print(f"S&P 500 Return:        {(bench_cum_ret[benchmark].iloc[-1] * 100):.2f}%")
    
    # ==========================================
    # 6. CORRELATION MATRIX
    # ==========================================
    print("\n" + "="*60)
    print("CORRELATION MATRIX (Diversification)")
    print("="*60)
    corr_matrix = calculate_correlation_matrix(daily_returns)
    print(corr_matrix.round(2).to_string())
    
    # ==========================================
    # 7. OPTIMIZATION (MARKOWITZ)
    # ==========================================
    print("\n" + "="*60)
    print("MARKOWITZ EFFICIENT FRONTIER OPTIMIZATION")
    print("="*60)
    
    # Assuming a 5% risk-free rate (e.g., US Treasury Bills)
    optimized_results = optimize_portfolio(prices, risk_free_rate=0.05) 
    
    print("\n--- Maximum Sharpe Ratio Portfolio ---")
    print("Weights:", dict(optimized_results["sharpe"]["weights"]))
    print(f"Expected Annual Return: {optimized_results['sharpe']['performance'][0]*100:.2f}%")
    print(f"Expected Annual Volatility: {optimized_results['sharpe']['performance'][1]*100:.2f}%")
    print(f"Sharpe Ratio: {optimized_results['sharpe']['performance'][2]:.4f}")
    
    print("\n--- Minimum Volatility Portfolio ---")
    print("Weights:", dict(optimized_results["min_vol"]["weights"]))
    print(f"Expected Annual Return: {optimized_results['min_vol']['performance'][0]*100:.2f}%")
    print(f"Expected Annual Volatility: {optimized_results['min_vol']['performance'][1]*100:.2f}%")
    print(f"Sharpe Ratio: {optimized_results['min_vol']['performance'][2]:.4f}")