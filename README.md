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
