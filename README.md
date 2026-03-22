# Algorithmic GARP Engine 📈🤖

An automated quantitative analysis pipeline that combines traditional "Growth at a Reasonable Price" (GARP) fundamental analysis with modern Machine Learning and Risk Parity optimization.

## 🎯 Executive Summary
Traditional value investing often misses high-growth opportunities, while pure growth investing exposes portfolios to severe drawdowns. This engine solves that by combining AI-driven market regime clustering with a strict mathematical focus on capital preservation. 

## 🏗️ System Architecture
This project is built in Python and executes a 4-phase pipeline:
1. **Data Engineering:** Extracts live fundamental data (PEG, ROE) and calculates technical indicators (RSI, Rolling Volatility).
2. **Unsupervised Learning (Market Regimes):** Utilizes **K-Means Clustering** to segment assets into distinct risk/reward profiles, allowing the system to recognize structural market differences.
3. **Supervised Learning (Prediction):** Trains an **XGBoost Classifier** on historical data to predict the probability of positive future returns based on momentum and volatility features.
4. **Portfolio Optimization (Financial Engineering):** Rejects naive $1/N$ diversification in favor of an **Inverse Volatility (Risk Parity)** weighting system. This ensures highly volatile assets receive lower capital allocation, strictly enforcing capital preservation.

## 📊 Sample Output
* **Highest Weighted Asset:** BBCA.JK (Low Volatility, Stable ROE)
* **Lowest Weighted Asset:** NVDA (High Volatility, High Momentum)
* *The algorithm successfully protects the portfolio's core while maintaining exposure to high-growth assets.*

## 🛠️ Technology Stack
* **Language:** Python
* **Data & Math:** `pandas`, `numpy`, `yfinance`, `pandas-ta`
* **Machine Learning:** `scikit-learn`, `xgboost`
* **Visualization:** `matplotlib`