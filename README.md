# 📈 Quant Portfolio Advisor & Risk Analytics

A quantitative financial analysis, risk management, and portfolio optimization system built in Python. This project is designed to assist investment advisors and financial analysts in data-driven decision-making, combining Modern Portfolio Theory (MPT) with advanced risk metrics.

---

## 🎯 Project Objectives

Provide a comprehensive analytical engine to:
- Evaluate historical performance and risk-adjusted volatility of financial assets (Equities, ADRs/CEDEARs, Crypto, and Market Indices).
- Identify and quantify tail risk using advanced quantitative metrics.
- Optimize asset allocation to build efficient portfolios tailored to investor risk profiles.

---

## 🛠️ Tech Stack

- **Language:** Python 3.10+
- **Data Analysis:** `pandas`, `numpy`
- **Market Data Retrieval:** `yfinance`
- **Optimization & Risk Modeling:** `scipy`, `PyPortfolioOpt`
- **Visualization & Dashboard:** `streamlit`, `plotly` *(in active development)*

---

## 🧠 Financial Metrics & Models

The engine automatically computes:
- **Returns:** Simple daily returns, logarithmic returns, cumulative returns.
- **Risk & Volatility:** Annualized volatility, Maximum Drawdown (MDD).
- **Performance Ratios:** Sharpe Ratio, Sortino Ratio (isolating downside volatility).
- **Distribution Analytics:** Skewness and Kurtosis to detect fat tails and black swan risks.

---

## 📂 Project Structure

```text
quant-portfolio-advisor/
│
├── app/
│   └── dashboard.py          # Interactive Streamlit interface
│
├── src/
│   ├── data/                 # Market data extraction and cleaning
│   │   └── market_data.py
│   │
│   ├── analytics/            # Quantitative analytics and risk engine
│   │   ├── returns.py
│   │   ├── risk_metrics.py
│   │   └── portfolio_optimizer.py
│   │
│   ├── alerts/               # Rule engine and AI sentiment analysis
│   │   ├── rules_engine.py
│   │   └── ai_news.py
│   │
│   └── database/             # Portfolio and position persistence
│       ├── db_manager.py
│       └── schema.sql
│
├── main.py                   # Main orchestrator script
├── requirements.txt          # Project dependencies
└── README.md
