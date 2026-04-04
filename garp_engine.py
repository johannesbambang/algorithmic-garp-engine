import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM # Added for HMM Regime Detection
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Optional: Portfolio Optimization
try:
    from pypfopt import EfficientFrontier, risk_models, expected_returns
    HAS_PYPFOPT = True
except ImportError:
    HAS_PYPFOPT = False

# --- CONFIGURATION ---
WATCHLIST = ['BBCA.JK', 'ANTM.JK', 'NVDA', 'AAPL', 'PNLF.JK', 'TLKM.JK', 'ASII.JK']

def fetch_and_detect_regimes(tickers):
    """
    Phase 1 & 2: Data Engineering & HMM Regime Detection.
    Transitions from K-Means to Hidden Markov Models.
    """
    print("--- Phase 1 & 2: Data Engineering & HMM Regime Detection ---")
    raw_data = []
    
    for t in tickers:
        stock = yf.Ticker(t)
        info = stock.info
        hist = stock.history(period="1y")
        
        if not hist.empty:
            # Fundamentals (GARP Logic)
            pe = info.get('forwardPE', 0)
            growth = info.get('earningsQuarterlyGrowth', 0) * 100
            # Handle growth for PEG calculation
            peg = pe / growth if (growth > 0 and pe) else 2.0 
            roe = info.get('returnOnEquity', 0)
            
            # Technicals
            close = hist['Close']
            rsi = ta.rsi(close, length=14).iloc[-1] if 'ta' in globals() else 50
            vol = close.pct_change().rolling(20).std().iloc[-1]
            
            raw_data.append({
                'Ticker': t, 'PEG': peg, 'ROE': roe, 'RSI': rsi, 'Vol': vol
            })
            print(f"Data Extracted: {t}")

    df = pd.DataFrame(raw_data).fillna(0)

    # Scaling Features
    features = ['PEG', 'ROE', 'RSI', 'Vol']
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(df[features])
    
    # HMM Regime Detection
    # GaussianHMM treats RSI and Volatility as a sequence of market states
    model = GaussianHMM(n_components=3, covariance_type="diag", n_iter=1000, random_state=42)
    model.fit(x_scaled)
    df['Cluster'] = model.predict(x_scaled)
    
    return df

def train_prediction_model(ticker):
    """Phase 3: Supervised Learning (XGBoost)"""
    print(f"\n--- Phase 3: Training XGBoost for {ticker} ---")
    stock = yf.Ticker(ticker)
    df = stock.history(period="2y")
    
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['Vol'] = df['Close'].pct_change().rolling(20).std()
    df['MA_Dist'] = (df['Close'] - df['Close'].rolling(50).mean()) / df['Close'].rolling(50).mean()
    
    # Labeling: 1 if return > 5% in next 20 days
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
    """
    Phase 4: Advanced Portfolio Optimization.
    Uses Mean-Variance Optimization if PyPortfolioOpt is available.
    """
    print("\n--- Phase 4: Portfolio Optimization (V2) ---")
    data = yf.download(tickers, period="2y")['Close']
    returns = data.pct_change().dropna()
    
    if HAS_PYPFOPT:
        print("Using Mean-Variance Optimization (PyPortfolioOpt)")
        mu = expected_returns.mean_historical_return(data)
        S = risk_models.sample_cov(data)
        ef = EfficientFrontier(mu, S)
        weights = ef.max_sharpe() # Optimize for the best Sharpe Ratio
        cleaned_weights = ef.clean_weights()
        return pd.Series(cleaned_weights)
    else:
        print("Using Fallback: Inverse Volatility (Risk Parity)")
        volatility = returns.std()
        inv_vol = 1 / volatility
        return inv_vol / inv_vol.sum()

def visualize_regimes(df):
    """Visualizes HMM states"""
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(df['Vol'], df['RSI'], c=df['Cluster'], cmap='viridis', s=100)
    for i, txt in enumerate(df['Ticker']):
        plt.annotate(txt, (df['Vol'].iloc[i], df['RSI'].iloc[i]), xytext=(5,5), textcoords='offset points')
    plt.colorbar(scatter, label='HMM Regime ID')
    plt.title('Algorithmic GARP Engine V2: HMM Market Regimes')
    plt.xlabel('Volatility (Risk)')
    plt.ylabel('RSI (Momentum)')
    plt.grid(True, alpha=0.3)
    plt.show()

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. HMM Regime Detection
    df_results = fetch_and_detect_regimes(WATCHLIST)
    print("\n--- HMM Regime Summary ---")
    print(df_results[['Ticker', 'Cluster', 'Vol', 'RSI']].sort_values(by='Cluster'))
    
    # 2. Train Prediction Benchmark
    train_prediction_model('NVDA')
    
    # 3. Enhanced Portfolio Optimization
    final_weights = optimize_portfolio_v2(WATCHLIST)
    print("\n--- Final Recommended Weights ---")
    for ticker, weight in final_weights.items():
        print(f"{ticker}: {weight*100:.2f}%")
            
    # 4. Show the Updated Chart
    visualize_regimes(df_results)
