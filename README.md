# Algorithmic GARP Engine V2 📈🤖

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-green)
![Smoke Test](https://github.com/johannesbambang/algorithmic-garp-engine/actions/workflows/main.yml/badge.svg)

**An automated quantitative pipeline combining “Growth at a Reasonable Price” (GARP) fundamentals with Hidden Markov Models (HMM) and Mean-Variance Optimization (MVO).**

---

## 📊 Performance at a Glance

*Backtest vs. Jakarta Composite Index (^JKSE)*

![Backtest](./img/backtest_results.png)

*Asset Clusters: Volatility vs. Momentum*

![Clusters](./img/clusters.png)

---

## 🧠 How It Works

### Market Regime Detection (HMM)
A 2‑state Gaussian HMM separates the market into **Low Volatility / Steady** and **High Volatility** regimes.

### Portfolio Optimization (MVO)
We maximize the **Sharpe Ratio**:
\[
S_p = \frac{R_p - R_f}{\sigma_p}
\]

### Supervised Learning (XGBoost)
Enriched features (Parkinson Volatility, ATR, Dividend Yield, PE) predict >5% forward returns.

---

## 🏗️ Architecture
