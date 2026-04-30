import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
from hmmlearn.hmm import GaussianHMM
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def detect_macro_regime(benchmark: str, n_components: int = 2) -> int:
    """Phase 1: HMM for market regime detection."""
    data = yf.Ticker(benchmark).history(period="2y")
    returns = data['Close'].pct_change().dropna().values.reshape(-1, 1)

    model = GaussianHMM(n_components=n_components,
                        covariance_type="diag",
                        n_iter=1000,
                        tol=1e-3,
                        random_state=42)
    model.fit(returns)

    current_state = model.predict(returns)[-1]
    regime = "High Volatility" if current_state == 1 else "Low Volatility / Steady"
    print(f"✓ Current Macro Market Regime: {regime}")
    return current_state

def train_prediction_model(ticker: str):
    """Phase 3: XGBoost with enriched features."""
    stock = yf.Ticker(ticker)
    df = stock.history(period="2y")
    info = stock.info

    # --- Feature Engineering ---
    df['RSI'] = ta.rsi(df['Close'], length=14)

    # Parkinson Volatility
    df['Parkinson_Vol'] = np.sqrt(
        (1 / (4 * np.log(2))) *
        (np.log(df['High'] / df['Low'])**2).rolling(20).mean()
    )

    # Average True Range
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

    # Dividend yield (fundamental)
    df['Div_Yield'] = info.get('dividendYield', 0)

    # P/E ratio (fundamental)
    df['PE_Ratio'] = info.get('forwardPE', 0)

    # Target: 5% return in next 20 days
    df['Future_Return'] = df['Close'].shift(-20) / df['Close'] - 1
    df['Target'] = (df['Future_Return'] > 0.05).astype(int)

    df = df.dropna()
    features = ['RSI', 'Parkinson_Vol', 'ATR', 'Div_Yield', 'PE_Ratio']
    X = df[features]
    y = df['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"✓ XGBoost Prediction Accuracy for {ticker}: {acc:.2%}")
    return model