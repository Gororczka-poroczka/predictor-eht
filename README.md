# predictor-eht
# ETH Price Prediction Bot 

High-performance machine learning pipeline designed to predict Ethereum (ETH) price direction by combining **US Macroeconomics** with **Technical Analysis**.

---

### Key Features

* **Macro-Driven Insights**: Analyzes **US Net Dollar Liquidity** (Fed Balance Sheet - Treasury Account - Reverse Repo) to gauge market "free money".
* **Market Correlations**: Monitors **S&P 500** and **US 10Y Treasury Yields** for global risk-on/risk-off sentiment.
* **Technical Intelligence**: Real-time calculation of **RSI**, **SMA (7/30)**, and **Rolling Volatility**.

---

### Performance Results

* **Directional Accuracy**: Achieved a steady **~55% accuracy** on a 1-year backtest.
* **Strategy vs Market**:
* **AI Strategy**: **1.15x profit** (+15%)
* **Market (Buy & Hold)**: **0.7x** (-30%)
* The bot successfully avoided major drawdowns by exiting to cash during high-risk periods.



---

###  Tech Stack

* **Language**: Python
* **AI Model**: XGBoost Classifier
* **Dashboard**: Streamlit
* **Data Sources**: Yahoo Finance & FRED (Federal Reserve Economic Data)

---

### Project Structure

* `crypto_app.py`: Main web dashboard for real-time signals.
* `crypto_predictor.ipynb`: Research and development notebook.
* `requirements.txt`: Project dependencies.

---

###  Quick Start

1. **Install Dependencies**:
```bash
pip install -r requirements.txt

```


2. **Launch Dashboard**:
```bash
streamlit run crypto_app.py
