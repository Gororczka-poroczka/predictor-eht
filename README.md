# predictor-eht
ETH Price Prediction Bot
This project uses Machine Learning (XGBoost) to predict if Ethereum (ETH) price will go up or down tomorrow.

How it works
The bot analyzes the following data points to make a prediction:

Market Data: ETH price, S&P 500 index, and US 10Y Treasury Yield.

Fed Liquidity: US Net Dollar Liquidity calculated from Fed balance sheet data.

Indicators: RSI (Relative Strength Index), SMA (7 and 30 days), and price volatility.

Results
Accuracy: ~52-55% during a 1-year backtest.

Performance: The strategy finished at 1.15x profit while the market dropped to 0.7x (avoiding major losses).

Project Files
crypto_app.py: The web dashboard (Streamlit).

crypto_predictor.ipynb: Development and testing notebook.

requirements.txt: List of necessary Python libraries.
Setup
1 pip install -r requirements.txt
2 streamlit run crypto_app1.py
