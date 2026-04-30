# Algorithmic GARP Engine V2 📈🤖

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-green)
![Smoke Test](https://github.com/johannesbambang/algorithmic-garp-engine/actions/workflows/main.yml/badge.svg)

**An automated quantitative pipeline combining “Growth at a Reasonable Price” (GARP) fundamentals with Hidden Markov Models (HMM) and Mean-Variance Optimization (MVO).**

---

## 📊 Performance at a Glance

*Backtest vs. Jakarta Composite Index (^JKSE)*

<img width="1501" height="817" alt="backtest_30042026_223645" src="https://github.com/user-attachments/assets/c30f2056-19dd-48e1-9195-0e061c69d528" />


*Asset Clusters: Volatility vs. Momentum*

<img width="1211" height="817" alt="clusters_30042026_223631" src="https://github.com/user-attachments/assets/15abd758-7cac-45bc-aa5e-6d32bcdccca1" />


---

## 🧠 How It Works

### Market Regime Detection (HMM)
A 2‑state Gaussian HMM separates the market into **Low Volatility / Steady** and **High Volatility** regimes. This dynamic approach replaces static clustering, adapting to structural shifts in market volatility.

### Portfolio Optimization (MVO)
We maximize the **Sharpe Ratio**, the industry‑standard measure of risk‑adjusted return:

**Sharpe Ratio = (Rₚ – R_f) / σₚ**

where  
- Rₚ = expected portfolio return  
- R_f = risk‑free rate  
- σₚ = portfolio volatility (standard deviation)

This ensures volatile assets receive lower weights, strictly enforcing capital preservation.

### Supervised Learning (XGBoost)
Enriched features – **Parkinson Volatility**, **Average True Range (ATR)**, **Dividend Yield**, and **P/E Ratio** – are fed into an XGBoost classifier to predict >5% forward returns over the next 20 trading days.

---

## 🏗️ Architecture
algorithmic-garp-engine/
├── src/
│ ├── data_engine.py # yfinance extraction & K‑Means clustering
│ ├── models.py # HMM regime detection + XGBoost training
│ ├── optimizer.py # MVO / Inverse Volatility portfolio
│ └── visualizer.py # Backtest chart & cluster plot
├── .github/workflows/ # CI smoke test
├── config.yaml # Tickers, benchmark, and test settings
├── main.py # CLI orchestrator (supports --test-mode)
├── requirements.txt
├── BEGINNER_GUIDE.md # Companion investing guide
├── LICENSE
└── README.md

---

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/johannesbambang/algorithmic-garp-engine.git
cd algorithmic-garp-engine
python -m venv .venv
# Activate virtual environment:
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
