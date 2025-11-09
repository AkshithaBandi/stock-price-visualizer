import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import datetime

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(page_title="📈 Advanced Stock Price Visualizer", layout="wide")

st.title("📊 Advanced Stock Price Visualizer")
st.markdown("Analyze, compare, and predict stock prices using real-time data and machine learning.")

# --------------------------------------------------
# Sidebar Inputs
# --------------------------------------------------
st.sidebar.header("Settings")

tickers = st.sidebar.text_input(
    "Enter stock tickers separated by commas (e.g. AAPL, GOOGL, MSFT)",
    "AAPL, GOOGL"
).upper().split(",")

start_date = st.sidebar.date_input("Start Date", datetime.date(2022, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())

show_ma = st.sidebar.checkbox("Show Moving Averages (SMA & EMA)", value=True)
sma_period = st.sidebar.slider("SMA Period", 5, 100, 20)
ema_period = st.sidebar.slider("EMA Period", 5, 100, 20)

st.sidebar.markdown("---")
st.sidebar.subheader("Prediction Settings")
predict_ticker = st.sidebar.text_input("Ticker for Prediction", "AAPL").upper()
future_days = st.sidebar.slider("Days to Predict", 1, 30, 7)

# --------------------------------------------------
# Fetch Data
# --------------------------------------------------
@st.cache_data
def fetch_data(ticker):
    try:
        df = yf.download(ticker, start=start_date, end=end_date)
        if df.empty:
            return pd.DataFrame()
        if show_ma:
            df[f"SMA_{sma_period}"] = df['Close'].rolling(window=sma_period).mean()
            df[f"EMA_{ema_period}"] = df['Close'].ewm(span=ema_period, adjust=False).mean()
        return df
    except Exception:
        return pd.DataFrame()

# --------------------------------------------------
# Visualization Section
# --------------------------------------------------
st.header("📈 Stock Data Visualization")

data_dict = {t.strip(): fetch_data(t.strip()) for t in tickers}

cols = st.columns(len(tickers))
for i, t in enumerate(tickers):
    df = data_dict.get(t)
    try:
        if df is None or df.empty or 'Close' not in df.columns:
            cols[i].metric(label=t, value="No data", delta="—")
            continue
        last_price = float(df['Close'].iloc[-1])
        prev = float(df['Close'].iloc[-2]) if len(df) > 1 else last_price
        delta = last_price - prev
        cols[i].metric(label=t, value=f"${last_price:,.2f}", delta=f"{delta:+.2f}")
    except Exception as e:
        cols[i].metric(label=t, value="Error", delta="—")
        st.warning(f"⚠️ Error processing {t}: {e}")

for t in tickers:
    df = data_dict.get(t)
    if df is None or df.empty:
        st.warning(f"No data available for {t}")
        continue

    st.subheader(f"📊 {t} Price Chart")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close', line=dict(color='cyan')))
    if show_ma:
        fig.add_trace(go.Scatter(x=df.index, y=df[f"SMA_{sma_period}"], name=f"SMA {sma_period}", line=dict(color='orange')))
        fig.add_trace(go.Scatter(x=df.index, y=df[f"EMA_{ema_period}"], name=f"EMA {ema_period}", line=dict(color='magenta')))
    fig.update_layout(title=f"{t} Closing Price", xaxis_title="Date", yaxis_title="Price (USD)")
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# Comparison Section
# --------------------------------------------------
if len(tickers) >= 2:
    st.header("🔁 Stock Comparison")
    try:
        comp_data = yf.download([t.strip() for t in tickers], start=start_date, end=end_date)['Close'].dropna()
        if not comp_data.empty:
            st.line_chart(comp_data)
        else:
            st.warning("No comparison data found.")
    except Exception as e:
        st.warning(f"Comparison error: {e}")

# --------------------------------------------------
# LSTM Prediction Section
# --------------------------------------------------
st.header("🤖 LSTM Stock Price Prediction")

if st.button("Predict Future Prices"):
    try:
        df_pred = yf.download(predict_ticker, start="2015-01-01", end=datetime.date.today())

        if df_pred.empty or 'Close' not in df_pred.columns:
            st.warning("No data found or missing 'Close' column for the selected ticker.")
        else:
            # Filter only Close column
            data = df_pred.filter(['Close']).dropna()

            # Check again to ensure valid data
            if data.empty:
                st.warning("No valid 'Close' price data available for this stock.")
            else:
                dataset = data.values

                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_data = scaler.fit_transform(dataset)

                training_data_len = int(np.ceil(len(dataset) * 0.8))
                train_data = scaled_data[0:training_data_len, :]

                # Build training sequences
                x_train, y_train = [], []
                for i in range(60, len(train_data)):
                    x_train.append(train_data[i - 60:i, 0])
                    y_train.append(train_data[i, 0])

                if len(x_train) == 0 or len(y_train) == 0:
                    st.warning("Insufficient data to train the LSTM model. Try a stock with longer history.")
                else:
                    x_train, y_train = np.array(x_train), np.array(y_train)
                    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

                    model = Sequential([
                        LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
                        LSTM(50, return_sequences=False),
                        Dense(25),
                        Dense(1)
                    ])
                    model.compile(optimizer='adam', loss='mean_squared_error')

                    with st.spinner("Training LSTM model... ⏳"):
                        model.fit(x_train, y_train, batch_size=32, epochs=3, verbose=0)

                    # Prepare for future prediction
                    last_60_days = scaled_data[-60:]
                    input_seq = last_60_days
                    predictions = []
                    for _ in range(future_days):
                        X_test = np.array([input_seq[-60:, 0]])
                        X_test = np.reshape(X_test, (1, 60, 1))
                        pred_price = model.predict(X_test, verbose=0)
                        predictions.append(pred_price[0][0])
                        input_seq = np.append(input_seq, pred_price)[-60:]

                    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
                    future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=future_days)
                    pred_df = pd.DataFrame({'Date': future_dates, 'Predicted Close': predictions.flatten()})

                    st.subheader(f"📅 {predict_ticker} {future_days}-Day Price Prediction")
                    st.dataframe(pred_df.set_index('Date'))

                    fig_pred = go.Figure()
                    fig_pred.add_trace(go.Scatter(x=data.index, y=data['Close'], name="Historical", line=dict(color='blue')))
                    fig_pred.add_trace(go.Scatter(x=pred_df['Date'], y=pred_df['Predicted Close'], name="Prediction", line=dict(color='red')))
                    fig_pred.update_layout(title=f"{predict_ticker} Price Prediction", xaxis_title="Date", yaxis_title="Price (USD)")
                    st.plotly_chart(fig_pred, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction error: {e}")


# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit, TensorFlow, and yfinance")
