# 📈 Stock Price Visualizer with LSTM Prediction

A Streamlit web app that allows users to visualize and compare historical stock prices, analyze trends, and forecast future prices using an LSTM (Long Short-Term Memory) deep learning model.

---

## 🚀 Features

- 📊 Visualize daily stock closing prices
- 📉 Compare two stock tickers side by side
- 📦 Display volume and moving averages (SMA, EMA)
- 📆 Select date ranges dynamically
- 🤖 Predict future stock prices using LSTM
- 📈 Plot predicted vs. actual price curves

---

## 🧠 LSTM Model

The app integrates a Long Short-Term Memory (LSTM) neural network to predict stock prices based on historical data.

### Model Highlights:
- Scales the closing price data using `MinMaxScaler`
- Trains an LSTM model on a sequence of past `n` days
- Forecasts future prices and visualizes the results

### Libraries Used:
- TensorFlow / Keras
- NumPy
- pandas
- matplotlib / plotly
- scikit-learn

---

## 🛠 Tech Stack

- Python
- Streamlit
- yfinance (stock data)
- pandas, NumPy
- matplotlib / plotly
- scikit-learn
- TensorFlow / Keras

---
