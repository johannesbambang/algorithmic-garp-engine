import os
import matplotlib
if os.environ.get('CI'):
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

# Ensure the image folder exists
os.makedirs('img', exist_ok=True)

def timestamp():
    return datetime.now().strftime("%d%m%Y_%H%M%S")

def plot_clusters(df):
    print("\n--- Asset Cluster Chart ---")
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(df['Vol'], df['RSI'], c=df['Cluster'], cmap='viridis', s=100)
    for i, txt in enumerate(df['Ticker']):
        plt.annotate(txt, (df['Vol'].iloc[i], df['RSI'].iloc[i]),
                     xytext=(5,5), textcoords='offset points')
    plt.colorbar(scatter, label='Cluster')
    plt.title('Algorithmic GARP Engine V2: Asset Clusters')
    plt.xlabel('Volatility (Risk)')
    plt.ylabel('RSI (Momentum)')
    plt.grid(True, alpha=0.3)

    ts_file = f'img/clusters_{timestamp()}.png'
    plt.savefig(ts_file, dpi=150, bbox_inches='tight')
    plt.savefig('img/clusters.png', dpi=150, bbox_inches='tight')   # for README
    print(f"✅ Cluster chart saved: {ts_file} and img/clusters.png")

    if not os.environ.get('CI'):
        plt.show()
    plt.close()

def run_backtest_report(tickers, weights, benchmark):
    print(f"\n--- Phase 5: Backtest vs {benchmark} ---")
    # Fetch data
    data = yf.download(tickers, period="2y", progress=False)['Close'].ffill().dropna()
    bench_data = yf.Ticker(benchmark).history(period="2y")['Close']

    # Timezone fix
    data.index = pd.to_datetime(data.index).tz_localize(None).normalize()
    bench_data.index = pd.to_datetime(bench_data.index).tz_localize(None).normalize()

    returns = data.pct_change().dropna()
    bench_returns = bench_data.pct_change().dropna()

    # Align dates
    aligned_returns, aligned_bench = returns.align(bench_returns, join='inner', axis=0)
    aligned_weights = weights.reindex(aligned_returns.columns).fillna(0)

    port_returns = aligned_returns.dot(aligned_weights)
    port_cum = (1 + port_returns).cumprod()
    bench_cum = (1 + aligned_bench).cumprod()

    # Metrics
    port_total = port_cum.iloc[-1] - 1
    bench_total = bench_cum.iloc[-1] - 1
    port_sharpe = (port_returns.mean() / port_returns.std()) * np.sqrt(252)
    bench_sharpe = (aligned_bench.mean() / aligned_bench.std()) * np.sqrt(252)

    print(f"Benchmark -> Return: {bench_total:.2%}, Sharpe: {bench_sharpe:.2f}")
    print(f"GARP Engine -> Return: {port_total:.2%}, Sharpe: {port_sharpe:.2f}")

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(port_cum.index, port_cum, label='GARP Engine V2', color='blue', linewidth=2)
    plt.plot(bench_cum.index, bench_cum, label=f'Benchmark ({benchmark})', color='gray', linestyle='--')
    plt.title('Phase 5: Backtest Performance (2 Years)')
    plt.ylabel('Cumulative Return')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True, alpha=0.3)

    ts_file = f'img/backtest_{timestamp()}.png'
    plt.savefig(ts_file, dpi=150, bbox_inches='tight')
    plt.savefig('img/backtest_results.png', dpi=150, bbox_inches='tight')  # for README
    print(f"✅ Backtest chart saved: {ts_file} and img/backtest_results.png")

    if not os.environ.get('CI'):
        plt.show()
    plt.close()