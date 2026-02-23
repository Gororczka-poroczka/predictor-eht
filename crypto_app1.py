import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_datareader.data as web
import datetime
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# Page configuration
st.set_page_config(page_title="Crypto AI Fund", layout="wide")
st.title(" Crypto AI Hedge Fund: RSI Edition")

@st.cache_data
def load_data():
    # 1. Data Acquisition (Last 5 Years)
    start = datetime.datetime.now() - datetime.timedelta(days=365*5)
    end = datetime.datetime.now()
    
    # Fetch Macro Data from FRED (Fed Balance Sheet, Treasury Account, Reverse Repo)
    fred_data = web.DataReader(['WALCL', 'WTREGEN', 'RRPONTSYD'], 'fred', start, end)
    fred_data['RRPONTSYD'] = fred_data['RRPONTSYD'] * 1000  # Convert Billions to Millions
    fred_data = fred_data.ffill().dropna()
    fred_data['Net_Liquidity'] = fred_data['WALCL'] - fred_data['WTREGEN'] - fred_data['RRPONTSYD']
    
    # Fetch Market Data (Ethereum, S&P 500, 10Y Treasury Yield)
    tickers = ['ETH-USD', '^GSPC', '^TNX']
    market_data = yf.download(tickers, period='5y')['Close']
    market_data.columns = ['ETH_Price', 'SP500', 'US10Y']
    
    # Merge datasets
    df = market_data.join(fred_data['Net_Liquidity']).ffill().dropna()
    
    # 2. Feature Engineering (Percentage Returns and RSI)
    df['Return_ETH'] = df['ETH_Price'].pct_change()
    df['Return_SP500'] = df['SP500'].pct_change()
    df['Return_Liq'] = df['Net_Liquidity'].pct_change()
    
    # Calculate Relative Strength Index (RSI - 14 Days)
    delta = df['ETH_Price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # Deviations from Simple Moving Averages
    df['Dist_SMA7'] = df['ETH_Price'] / df['ETH_Price'].rolling(window=7).mean() - 1
    df['Dist_SMA30'] = df['ETH_Price'] / df['ETH_Price'].rolling(window=30).mean() - 1
    
    df = df.dropna()
    
    # Target: Will the price increase tomorrow? (1 = Yes, 0 = No)
    df['Target'] = (df['ETH_Price'].shift(-1) > df['ETH_Price']).astype(int)
    
    # Extract the most recent data point for tomorrow's prediction
    today_data = df.iloc[-1:] 
    df = df.dropna() 
    
    return df, today_data

st.write("📥 Fetching market data and calculating indicators...")
df, today_data = load_data()

# --- MODEL TRAINING ---
features = ['Return_ETH', 'Return_SP500', 'Return_Liq', 'RSI_14', 'Dist_SMA7', 'Dist_SMA30']
X = df[features]
y = df['Target']

# Define test set (last 365 days)
test_size = 365
X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]

# Initialize XGBoost with optimized hyper-parameters
clf = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
clf.fit(X_train, y_train)

# Generate predictions for the test period
signals = clf.predict(X_test)
acc = accuracy_score(y_test, signals)

# --- BACKTESTING ---
# Shift returns to align tomorrow's price change with today's prediction
test_returns = df['Return_ETH'].shift(-1).iloc[-test_size:]
strategy_returns = signals * test_returns

# Calculate equity curves
cumulative_market = (1 + test_returns).cumprod()
cumulative_strategy = (1 + strategy_returns).cumprod()

# --- USER INTERFACE ---
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.metric(label="Directional Accuracy (Backtest)", value=f"{acc*100:.2f}%")
    st.write(f"Current RSI: **{today_data['RSI_14'].values[0]:.2f}**")

with col2:
    # Predict tomorrow's signal
    tomorrow_prediction = clf.predict(today_data[features])[0]
    st.subheader("Recommendation for Tomorrow:")
    if tomorrow_prediction == 1:
        st.success("🟢 BUY / HOLD")
    else:
        st.error("🔴 EXIT TO CASH")

st.markdown("---")
st.subheader("Performance Comparison (AI + RSI) vs. Market")

# Plotting the results
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(cumulative_market.index, cumulative_market, label='Market (ETH Buy & Hold)', color='gray', alpha=0.5)
ax.plot(cumulative_strategy.index, cumulative_strategy, label='AI + RSI Strategy', color='blue', linewidth=2)
ax.set_ylabel('Equity Growth (Initial = 1.0)')
ax.legend()
ax.grid(True, alpha=0.3)

st.pyplot(fig)