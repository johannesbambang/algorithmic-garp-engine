# Algorithmic GARP Engine V2 📈🤖

An automated quantitative analysis pipeline that combines traditional **"Growth at a Reasonable Price" (GARP)** fundamental analysis with **Probabilistic Market Regime Detection** and **Mean-Variance Optimization**.

## 🎯 Executive Summary
Traditional value investing often misses high-growth opportunities, while pure growth investing exposes portfolios to severe drawdowns. This engine solves that by combining **Hidden Markov Models (HMM)** for dynamic regime detection with **Mean-Variance Optimization (MVO)** to maximize the risk-adjusted return (Sharpe Ratio).

## 🏗️ System Architecture
This project is built in Python and executes a 4-phase pipeline:
1. **Data Engineering:** Extracts live fundamental data (PEG, ROE) and calculates technical indicators (RSI, Rolling Volatility) using yfinance and pandas-ta.
2. **Unsupervised Learning (Market Regimes):** Replaces static K-Means with **Hidden Markov Models (HMM)**. The system now dynamically detects transition probabilities between hidden market states (e.g., Low, Mid, and High Volatility regimes).
3. **Supervised Learning (Prediction):** Trains an **XGBoost Classifier** on historical data to predict the probability of positive future returns based on momentum and volatility features.
4. **Portfolio Optimization (Financial Engineering):** Rejects naive $1/N$ diversification in favor of an **Inverse Volatility (Risk Parity)** weighting system. This ensures highly volatile assets receive lower capital allocation, strictly enforcing capital preservation.

## 📊 Sample Output
* **HMM Regime Detection:** Automatically identifies the current structural state of the market.
* **Optimization Goal:** Maximize Sharpe Ratio (balancing returns vs. risk).
* **Result:** A highly adaptive portfolio that adjusts its core protection based on the detected market regime.

## 🛠️ Technology Stack
* **Language:** Python
* **Data & Math:** `pandas`, `numpy`, `yfinance`, `pandas-ta`
* **Machine Learning:** `scikit-learn`, `xgboost`, `hmmlearn (New in V2)`
* **Backtesting:** backtrader (Current implementation phase)
* **Visualization:** `matplotlib`
