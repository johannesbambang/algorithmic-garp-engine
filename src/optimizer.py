import yfinance as yf
import pandas as pd
import numpy as np

try:
    from pypfopt import EfficientFrontier, risk_models, expected_returns
    HAS_PYPFOPT = True
except ImportError:
    HAS_PYPFOPT = False

def optimize_portfolio(tickers, lookback="2y"):
    """
    Phase 4: Portfolio optimisation.
    Uses PyPortfolioOpt if available, otherwise Inverse Volatility (Risk Parity).
    """
    print("\n--- Phase 4: Portfolio Optimization (MVO) ---")
    data = yf.download(tickers, period=lookback, progress=False)['Close']
    data = data.ffill().dropna()

    if data.empty:
        print("⚠️  No price data; returning equal weights.")
        return pd.Series(1/len(tickers), index=tickers)

    if HAS_PYPFOPT:
        print("Using Mean-Variance Optimization (PyPortfolioOpt)")
        mu = expected_returns.mean_historical_return(data)
        S = risk_models.sample_cov(data)
        ef = EfficientFrontier(mu, S)
        weights = ef.max_sharpe()
        cleaned = ef.clean_weights()
        return pd.Series(cleaned).reindex(tickers, fill_value=0)
    else:
        print("Using Inverse Volatility (Risk Parity)")
        returns = data.pct_change().dropna()
        vol = returns.std()
        inv_vol = 1 / vol
        return (inv_vol / inv_vol.sum()).reindex(tickers, fill_value=0)