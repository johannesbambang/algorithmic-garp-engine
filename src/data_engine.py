import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from typing import List

def fetch_and_cluster(
    tickers: List[str],
    cluster_count: int = 3,
    lookback_period: str = "1y"
) -> pd.DataFrame:
    """
    Downloads fundamental + technical data, scales, and runs K‑Means clustering.

    Parameters
    ----------
    tickers : list of str
    cluster_count : int
    lookback_period : str

    Returns
    -------
    pd.DataFrame with columns Ticker, PEG, ROE, RSI, Vol, Cluster.
    """
    raw_data = []

    for t in tickers:
        stock = yf.Ticker(t)
        info = stock.info
        hist = stock.history(period=lookback_period)

        if hist.empty:
            print(f"⚠️  No data for {t}, skipping.")
            continue

        # Fundamentals
        pe = info.get('forwardPE', 0)
        growth = info.get('earningsQuarterlyGrowth', 0) * 100
        peg = pe / growth if (growth and pe) else 2.0
        roe = info.get('returnOnEquity', 0)

        # Technicals
        close = hist['Close']
        rsi_series = ta.rsi(close, length=14)
        rsi_val = rsi_series.iloc[-1] if not rsi_series.empty else 50
        vol = close.pct_change().rolling(20).std().iloc[-1]

        raw_data.append({
            'Ticker': t,
            'PEG': peg,
            'ROE': roe,
            'RSI': rsi_val,
            'Vol': vol
        })
        print(f"✓ Data extracted: {t}")

    if not raw_data:
        raise ValueError("No valid data retrieved for any ticker.")

    df = pd.DataFrame(raw_data).fillna(0)

    # Feature scaling
    features = ['PEG', 'ROE', 'RSI', 'Vol']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])

    n_samples = len(df)
    effective_clusters = min(n_samples, cluster_count)
    if effective_clusters < 2:
        print("⚠️  Not enough assets for clustering. Assigning cluster 0 to all.")
        df['Cluster'] = 0
        return df

    kmeans = KMeans(n_clusters=effective_clusters, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    return df