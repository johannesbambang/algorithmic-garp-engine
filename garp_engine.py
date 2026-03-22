import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- CONFIGURATION ---
# Your preferred blend of IHSG and US Stocks
WATCHLIST = ['BBCA.JK', 'ANTM.JK', 'NVDA', 'AAPL', 'PNLF.JK', 'TLKM.JK', 'ASII.JK']

def fetch_and_cluster(tickers):
    print("--- Phase 1 & 2: Data Engineering & Clustering ---")
    raw_data = []
    
    for t in tickers:
        stock = yf.Ticker(t)
        info = stock.info
        hist = stock.history(period="1y")
        
        if not hist.empty:
            # Fundamentals (GARP Logic)
            pe = info.get('forwardPE', 0)
            growth = info.get('earningsQuarterlyGrowth', 0) * 100
            peg = pe / growth if (growth > 0 and pe) else 2.0 
            roe = info.get('returnOnEquity', 0)
            
            # Technicals (Momentum & Risk)
            # Use simple pandas if pandas_ta isn't working
            close = hist['Close']
            rsi = ta.rsi(close, length=14).iloc[-1] if 'ta' in globals() else 50
            vol = close.pct_change().rolling(20).std().iloc[-1]
            
            raw_data.append({
                'Ticker': t, 'PEG': peg, 'ROE': roe, 'RSI': rsi, 'Vol': vol
            })
            print(f"Data Extracted: {t}")

    df = pd.DataFrame(raw_data).fillna(0)

    # Scaling & Clustering
    features = ['PEG', 'ROE', 'RSI', 'Vol']
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(df[features])
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(x_scaled)
    
    return df

def train_prediction_model(ticker):
    print(f"\n--- Phase 3: Training XGBoost for {ticker} ---")
    stock = yf.Ticker(ticker)
    df = stock.history(period="2y")
    
    # Simple Feature Engineering
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['Vol'] = df['Close'].pct_change().rolling(20).std()
    df['MA_Dist'] = (df['Close'] - df['Close'].rolling(50).mean()) / df['Close'].rolling(50).mean()
    
    # Labeling: 1 if price increases > 5% in next 20 days
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

def optimize_portfolio_lite(tickers):
    print("\n--- Phase 4: Risk Parity Optimization (Lite) ---")
    # Download 2 years of price data
    data = yf.download(tickers, period="2y")['Close']
    returns = data.pct_change().dropna()
    
    # Calculate Volatility (Standard Deviation of returns)
    volatility = returns.std()
    
    # Inverse Volatility Weighting
    # The math: Weight = (1 / Vol) / Sum(1 / Vol)
    inv_vol = 1 / volatility
    weights = inv_vol / inv_vol.sum()
    
    return weights

def visualize_results(df):
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(df['Vol'], df['RSI'], c=df['Cluster'], cmap='viridis', s=100)
    for i, txt in enumerate(df['Ticker']):
        plt.annotate(txt, (df['Vol'].iloc[i], df['RSI'].iloc[i]), xytext=(5,5), textcoords='offset points')
    plt.colorbar(scatter, label='Cluster ID')
    plt.title('Johannes Style: Market Regimes (Vol vs RSI)')
    plt.xlabel('Volatility (Risk)')
    plt.ylabel('RSI (Momentum)')
    plt.grid(True, alpha=0.3)
    plt.show()

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Run Data Engineering & Clustering
    df_results = fetch_and_cluster(WATCHLIST)
    print("\n--- AI Clustering Summary ---")
    print(df_results.sort_values(by='Cluster'))
    
    # 2. Train Prediction Model (on NVDA as a benchmark)
    train_prediction_model('NVDA')
    
    # 3. Portfolio Optimization (Lite Version)
    final_weights = optimize_portfolio_lite(WATCHLIST)
    print("\n--- Final Recommended Weights (Risk-Parity) ---")
    for ticker, weight in final_weights.items():
        print(f"{ticker}: {weight*100:.2f}%")
            
    # 4. Show the Chart
    visualize_results(df_results)