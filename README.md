# Algorithmic GARP Engine V2 📈🤖

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-green)
![Smoke Test](https://github.com/johannesbambang/algorithmic-garp-engine/actions/workflows/main.yml/badge.svg)

**An automated quantitative pipeline combining “Growth at a Reasonable Price” (GARP) fundamentals with Hidden Markov Models (HMM) and Mean-Variance Optimization (MVO).**

---

## 📊 Performance at a Glance

*Backtest vs. Jakarta Composite Index (^JKSE)*

<img width="1501" height="817" alt="backtest_30042026_224331" src="https://github.com/user-attachments/assets/f94712b9-f80d-484a-9f4a-34224dd03a07" />


*Asset Clusters: Volatility vs. Momentum*

<img width="1211" height="817" alt="clusters_30042026_224308" src="https://github.com/user-attachments/assets/11a71761-4861-4e26-8796-31ac6463ae6b" />


---

## 🧠 How It Works

### Market Regime Detection (HMM)
A 2‑state Gaussian HMM separates the market into **Low Volatility / Steady** and **High Volatility** regimes. This dynamic approach replaces static clustering, adapting to structural shifts in market volatility.

### Portfolio Optimization (MVO)
We maximize the **Sharpe Ratio**, the industry‑standard measure of risk‑adjusted return:

**Sharpe Ratio = (Rₚ – R_f) / σₚ**

where:
- Rₚ = expected portfolio return
- R_f = risk‑free rate
- σₚ = portfolio volatility (standard deviation)

This ensures volatile assets receive lower weights, strictly enforcing capital preservation.

### Supervised Learning (XGBoost)
Enriched features – **Parkinson Volatility**, **Average True Range (ATR)**, **Dividend Yield**, and **P/E Ratio** – are fed into an XGBoost classifier to predict >5% forward returns over the next 20 trading days.

---

## 🏗️ Architecture

```
algorithmic-garp-engine/
├── src/
│   ├── data_engine.py
│   ├── models.py
│   ├── optimizer.py
│   └── visualizer.py
├── .github/workflows/
├── config.yaml
├── main.py
├── requirements.txt
├── BEGINNER_GUIDE.md
├── LICENSE.md
└── README.md
```

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
```

### 2. Configure Your Universe
Edit `config.yaml`:

```yaml
market_settings:
  benchmark: "^JKSE"
  ...
```

### 3. Run the Full Pipeline
```bash
python main.py
```
This will:
- Detect the current macro regime (HMM)
- Cluster your assets by volatility & momentum
- Train an XGBoost predictor for the first ticker
- Optimise portfolio weights (MVO)
- Show cumulative return backtest vs. benchmark
- Show asset cluster chart

### 4. Smoke Test (for CI/CD)
```bash
python main.py --test-mode
```
Uses a minimal watchlist and short lookback to quickly verify the entire pipeline.

---

## 📘 Beginner Investor Guide

If you’re new to long‑term investing, we’ve also prepared a **companion guide** that covers:

- USA vs. Indonesia investment vehicles
- A 20‑year Moderate‑Aggressive portfolio allocation
- Dividend compounding & tax‑free reinvestment strategies
- Cost analysis (why low‑cost index funds win)

Read it here: **[BEGINNER_GUIDE.md](BEGINNER_GUIDE.md)**

---

## 📄 License

This project is licensed under the MIT License – see **[LICENSE.md](LICENSE.md)** for details.

---

<p align="center">
  Made with 🔬 by <a href="https://github.com/johannesbambang">johannesbambang</a>
</p>
