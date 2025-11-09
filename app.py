import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------
st.set_page_config(page_title="Stock Price Predictor", layout="wide")
st.title("📈 Stock Price Prediction App")

st.sidebar.header("Select Parameters")
ticker = st.sidebar.text_input("Enter Stock Symbol (e.g., AAPL, GOOGL, MSFT):", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2025-01-01"))

# --------------------------------------------------
# Fetch Data
# --------------------------------------------------
st.subheader(f"Showing data for {ticker}")

data = yf.download(ticker, start=start_date, end=end_date)
if data.empty:
    st.error("⚠️ No data found. Please check the ticker symbol or date range.")
    st.stop()

st.write(data.tail())

# --------------------------------------------------
# Plot Stock Prices
# --------------------------------------------------
st.subheader("📊 Stock Closing Price")
st.line_chart(data['Close'])

# --------------------------------------------------
# Data Preprocessing
# --------------------------------------------------
scaler = MinMaxScaler(feature_range=(0, 1))
close_data = data['Close'].values.reshape(-1, 1)
scaled_data = scaler.fit_transform(close_data)

train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size - 60:]

def create_dataset(dataset, time_step=60):
    X, y = [], []
    for i in range(time_step, len(dataset)):
        X.append(dataset[i - time_step:i, 0])
        y.append(dataset[i, 0])
    return np.array(X), np.array(y)

X_train, y_train = create_dataset(train_data)
X_test, y_test = create_dataset(test_data)

# Reshape for LSTM input
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# --------------------------------------------------
# LSTM Model
# --------------------------------------------------
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(60, 1)),
    LSTM(50),
    Dense(25),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# --------------------------------------------------
# Train the Model
# --------------------------------------------------
with st.spinner("Training the LSTM model... ⏳"):
    model.fit(X_train, y_train, batch_size=32, epochs=3, verbose=0)

st.success("✅ Model trained successfully!")

# --------------------------------------------------
# Predictions
# --------------------------------------------------
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

real_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

# --------------------------------------------------
# Display Results
# --------------------------------------------------
st.subheader("📉 Actual vs Predicted Prices")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(real_prices, label="Actual Price", color="blue")
ax.plot(predictions, label="Predicted Price", color="red")
ax.set_xlabel("Days")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)

# --------------------------------------------------
# Comparison: AAPL vs GOOGL
# --------------------------------------------------
st.subheader("📊 Comparison: AAPL vs GOOGL")

try:
    comp_data = yf.download(['AAPL', 'GOOGL'], start='2023-01-01', end='2025-01-01')['Close']
    comp_data = comp_data.dropna()
    st.line_chart(comp_data)
except Exception as e:
    st.error(f"Error fetching comparison data: {e}")

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit, TensorFlow, and yfinance")
