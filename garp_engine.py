import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM 
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings

# Suppress yfinance warnings for cleaner output
warnings.filterwarnings('ignore')

# Optional: Portfolio Optimization
try:
    from pypfopt import EfficientFrontier, risk_models, expected_returns
    HAS_PYPFOPT = True
except ImportError:
    HAS_PYPFOPT = False

# --- CONFIGURATION ---
WATCHLIST = ['BBCA.JK', 'ANTM.JK', 'NVDA', 'AAPL', 'PNLF.JK', 'TLKM.JK', 'ASII.JK']
BENCHMARK = '^JKSE' # Jakarta Composite Index (IHSG)

def detect_macro_regime(benchmark=BENCHMARK):
    """Phase 1: HMM on Historical Time-Series Data"""
    print(f"--- Phase 1: HMM Macro Regime Detection ({benchmark}) ---")
    data = yf.Ticker(benchmark).history(period="2y")
    
    # Calculate daily returns and drop NaNs
    returns = data['Close'].pct_change().dropna().values.reshape(-1, 1)
    
    # HMM properly applied to a time-series sequence
    model = GaussianHMM(n_components=2, covariance_type="diag", n_iter=1000, random_state=42)
    model.fit(returns)
    current_state = model.predict(returns)[-1]
    
    regime = "High Volatility" if current_state == 1 else "Low Volatility / Steady"
    print(f"Current Macro Market Regime detected as: {regime}")
    return current_state

def fetch_and_cluster(tickers):
    """Phase 2: Data Engineering & Cross-Sectional Clustering"""
    print("\n--- Phase 2: Cross-Sectional Asset Clustering ---")
    raw_data = []
    
    for t in tickers:
        stock = yf.Ticker(t)
        info = stock.info
        hist = stock.history(period="1y")
        
        if not hist.empty:
            pe = info.get('forwardPE', 0)
            growth = info.get('earningsQuarterlyGrowth', 0) * 100
            peg = pe / growth if (growth > 0 and pe) else 2.0 
            roe = info.get('returnOnEquity', 0)
            
            close = hist['Close']
            rsi = ta.rsi(close, length=14).iloc[-1] if 'ta' in globals() else 50
            vol = close.pct_change().rolling(20).std().iloc[-1]
            
            raw_data.append({'Ticker': t, 'PEG': peg, 'ROE': roe, 'RSI': rsi, 'Vol': vol})
            print(f"Data Extracted: {t}")

    df = pd.DataFrame(raw_data).fillna(0)
    features = ['PEG', 'ROE', 'RSI', 'Vol']
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(df[features])
    
    # Use K-Means for grouping different assets on the same timeline
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(x_scaled)
    
    return df

def train_prediction_model(ticker):
    """Phase 3: Supervised Learning (XGBoost)"""
    print(f"\n--- Phase 3: Training XGBoost for {ticker} ---")
    stock = yf.Ticker(ticker)
    df = stock.history(period="2y")
    
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['Vol'] = df['Close'].pct_change().rolling(20).std()
    df['MA_Dist'] = (df['Close'] - df['Close'].rolling(50).mean()) / df['Close'].rolling(50).mean()
    
    df['Future_Return'] = df['Close'].shift(-20) / df['Close'] - 1
    df['Target'] = (df['Future_Return'] > 0.05).astype(int)
    
    df = df.dropna()
    X = df[['RSI', 'Vol', 'MA_Dist']]
    y = df['Target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5)
    model.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"XGBoost Prediction Accuracy: {acc:.2%}")
    return model

def optimize_portfolio_v2(tickers):
    """Phase 4: Advanced Portfolio Optimization"""
    print("\n--- Phase 4: Portfolio Optimization (V2) ---")
    data = yf.download(tickers, period="2y")['Close']
    
    # FIX: Forward-fill NaNs caused by mismatched international market holidays
    data = data.ffill().dropna() 
    returns = data.pct_change().dropna()
    
    if HAS_PYPFOPT:
        print("Using Mean-Variance Optimization (PyPortfolioOpt)")
        mu = expected_returns.mean_historical_return(data)
        S = risk_models.sample_cov(data)
        ef = EfficientFrontier(mu, S)
        weights = ef.max_sharpe() 
        cleaned_weights = ef.clean_weights()
        return pd.Series(cleaned_weights)
    else:
        print("Using Fallback: Inverse Volatility (Risk Parity)")
        volatility = returns.std()
        inv_vol = 1 / volatility
        return inv_vol / inv_vol.sum()

def run_backtest(tickers, weights, benchmark=BENCHMARK):
    """Phase 5: Historical Backtesting against Benchmark"""
    print(f"\n--- Phase 5: Historical Backtesting ({benchmark}) ---")
    
    # 1. Fetch historical data
    # Suppress output during download for a cleaner console
    data = yf.download(tickers, period="2y", progress=False)['Close'].ffill().dropna()
    bench_data = yf.Ticker(benchmark).history(period="2y")['Close']
    
    # --- THE TIMEZONE FIX ---
    # Strip timezone metadata and keep only the calendar date
    data.index = pd.to_datetime(data.index).tz_localize(None).normalize()
    bench_data.index = pd.to_datetime(bench_data.index).tz_localize(None).normalize()
    # ------------------------
    
    # 2. Calculate daily returns
    returns = data.pct_change().dropna()
    bench_returns = bench_data.pct_change().dropna()
    
    # Align dates (handles mismatched trading days globally)
    aligned_returns, aligned_bench = returns.align(bench_returns, join='inner', axis=0)
    
    # Ensure weights align with the columns of the return dataframe
    aligned_weights = weights.reindex(aligned_returns.columns).fillna(0)
    
    # 3. Calculate Portfolio Returns via Matrix Multiplication
    portfolio_returns = aligned_returns.dot(aligned_weights)
    
    # 4. Calculate Cumulative Returns
    port_cum_returns = (1 + portfolio_returns).cumprod()
    bench_cum_returns = (1 + aligned_bench).cumprod()
    
    # 5. Calculate Metrics
    port_total_return = port_cum_returns.iloc[-1] - 1
    bench_total_return = bench_cum_returns.iloc[-1] - 1
    
    port_max_dd = ((port_cum_returns / port_cum_returns.cummax()) - 1).min()
    bench_max_dd = ((bench_cum_returns / bench_cum_returns.cummax()) - 1).min()
    
    port_sharpe = (portfolio_returns.mean() / portfolio_returns.std()) * np.sqrt(252)
    bench_sharpe = (aligned_bench.mean() / aligned_bench.std()) * np.sqrt(252)
    
    # Print Results
    print(f"Benchmark ({benchmark}) -> Return: {bench_total_return:.2%}, Max DD: {bench_max_dd:.2%}, Sharpe: {bench_sharpe:.2f}")
    print(f"GARP Engine MVO   -> Return: {port_total_return:.2%}, Max DD: {port_max_dd:.2%}, Sharpe: {port_sharpe:.2f}")
    
    # 6. Visualize the Backtest
    plt.figure(figsize=(12, 6))
    plt.plot(port_cum_returns.index, port_cum_returns, label='GARP Engine V2.1 (MVO)', color='blue', linewidth=2)
    plt.plot(bench_cum_returns.index, bench_cum_returns, label=f'Benchmark ({benchmark})', color='gray', linestyle='--')
    plt.title('Phase 5: Backtest Performance (2 Years)')
    plt.ylabel('Cumulative Return')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def visualize_regimes(df):
    """Visualizes Asset Clusters"""
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(df['Vol'], df['RSI'], c=df['Cluster'], cmap='viridis', s=100)
    for i, txt in enumerate(df['Ticker']):
        plt.annotate(txt, (df['Vol'].iloc[i], df['RSI'].iloc[i]), xytext=(5,5), textcoords='offset points')
    plt.colorbar(scatter, label='K-Means Cluster ID')
    plt.title('Algorithmic GARP Engine V2: Asset Clusters')
    plt.xlabel('Volatility (Risk)')
    plt.ylabel('RSI (Momentum)')
    plt.grid(True, alpha=0.3)
    plt.show()

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Macro Regime Detection
    macro_state = detect_macro_regime()
    
    # 2. Cross-Sectional Asset Clustering
    df_results = fetch_and_cluster(WATCHLIST)
    print("\n--- Asset Cluster Summary ---")
    print(df_results[['Ticker', 'Cluster', 'Vol', 'RSI']].sort_values(by='Cluster'))
    
    # 3. Train Prediction Benchmark
    train_prediction_model('NVDA')
    
    # 4. Enhanced Portfolio Optimization
    final_weights = optimize_portfolio_v2(WATCHLIST)
    print("\n--- Final Recommended Weights ---")
    for ticker, weight in final_weights.items():
        print(f"{ticker}: {weight*100:.2f}%")
        
    # 5. Run the Backtest Simulation
    run_backtest(WATCHLIST, final_weights, BENCHMARK)
        
    # 6. Show the Asset Cluster Chart
    # Note: This will appear AFTER you close the Backtest chart window
    visualize_regimes(df_results)
