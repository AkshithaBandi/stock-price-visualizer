# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# TensorFlow / Keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.set_page_config(page_title="Enhanced Stock Price Visualizer", layout="wide")
st.title("📈 Enhanced Stock Price Visualizer")

# === Main Inputs ===
ticker1 = st.text_input("Enter first stock ticker:", "AAPL").upper()
ticker2 = st.text_input("Enter second stock ticker (optional):", "").upper()
start_date = st.date_input("Start Date", pd.to_datetime("2022-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("today"))
show_ma = st.checkbox("Show Moving Averages (SMA & EMA)")
sma_period = st.slider("SMA Period", 5, 100, 20)
ema_period = st.slider("EMA Period", 5, 100, 20)

@st.cache_data(ttl=60*60)  # cache for 1 hour
def fetch_data(ticker, start_date, end_date, show_ma, sma_period, ema_period):
    """Download data and compute moving averages (if requested)."""
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    except Exception:
        df = pd.DataFrame()
    if not df.empty and show_ma:
        df[f"SMA_{sma_period}"] = df['Close'].rolling(window=sma_period).mean()
        df[f"EMA_{ema_period}"] = df['Close'].ewm(span=ema_period, adjust=False).mean()
    return df

# === Visualization Section ===
if st.button("Visualize"):
    try:
        df1 = fetch_data(ticker1, start_date, end_date, show_ma, sma_period, ema_period)
        if df1.empty:
            st.warning(f"No data found for {ticker1}")
        else:
            st.subheader(f"📊 Stock Chart for {ticker1}")
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=df1.index, y=df1['Close'], name="Close Price"))
            if show_ma:
                sma_col = f"SMA_{sma_period}"
                ema_col = f"EMA_{ema_period}"
                if sma_col in df1.columns:
                    fig1.add_trace(go.Scatter(x=df1.index, y=df1[sma_col], name=f"SMA {sma_period}"))
                if ema_col in df1.columns:
                    fig1.add_trace(go.Scatter(x=df1.index, y=df1[ema_col], name=f"EMA {ema_period}"))
            fig1.update_layout(title=f"{ticker1} Closing Price with MA", xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(fig1, use_container_width=True)

            # Volume Chart
            if 'Volume' in df1.columns:
                fig_vol = go.Figure()
                fig_vol.add_trace(go.Bar(x=df1.index, y=df1['Volume'], name="Volume"))
                fig_vol.update_layout(title=f"{ticker1} Trading Volume", xaxis_title="Date", yaxis_title="Volume")
                st.plotly_chart(fig_vol, use_container_width=True)

        if ticker2:
            df2 = fetch_data(ticker2, start_date, end_date, show_ma, sma_period, ema_period)
            if df2.empty:
                st.warning(f"No data found for {ticker2}")
            else:
                st.subheader(f"🔁 Comparison: {ticker1} vs {ticker2}")

                # Align dates for comparison (merge on index)
                merged = pd.DataFrame({ticker1: df1['Close']}).join(pd.DataFrame({ticker2: df2['Close']}), how='outer').dropna()
                if merged.empty:
                    st.warning("No overlapping dates to compare the two tickers.")
                else:
                    fig_compare = go.Figure()
                    fig_compare.add_trace(go.Scatter(x=merged.index, y=merged[ticker1], name=ticker1))
                    fig_compare.add_trace(go.Scatter(x=merged.index, y=merged[ticker2], name=ticker2))
                    fig_compare.update_layout(title="Stock Price Comparison", xaxis_title="Date", yaxis_title="Price")
                    st.plotly_chart(fig_compare, use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")

# === LSTM Prediction Section ===
st.header("🤖 LSTM Stock Price Prediction")

predict_ticker = st.text_input("Enter a stock ticker for prediction", "AAPL").upper()
future_days = st.slider("Days to predict into the future", 1, 30, 7)

if st.button("Predict Future Price"):
    try:
        # Download historical data
        df = yf.download(predict_ticker, start="2015-01-01", end=pd.to_datetime("today"), progress=False)

        if df.empty:
            st.warning(f"No data found for {predict_ticker}. Try another ticker.")
        else:
            # Ensure Close exists
            if 'Close' not in df.columns:
                st.error("No 'Close' price found in the downloaded data.")
            else:
                data = df.filter(['Close'])

                if data.empty:
                    st.error("Close column is empty; cannot proceed.")
                else:
                    dataset = data.values  # shape (n_rows, 1)

                    # Scale data
                    scaler = MinMaxScaler(feature_range=(0, 1))
                    scaled_data = scaler.fit_transform(dataset)

                    # Split training data (80%)
                    training_data_len = int(np.ceil(len(dataset) * 0.8))
                    train_data = scaled_data[0:training_data_len, :]

                    # Prepare training sequences
                    x_train, y_train = [], []
                    seq_len = 60
                    for i in range(seq_len, len(train_data)):
                        x_train.append(train_data[i-seq_len:i, 0])
                        y_train.append(train_data[i, 0])

                    if len(x_train) == 0:
                        st.error("Not enough data to create training sequences. Need at least 60+ samples.")
                    else:
                        x_train = np.array(x_train)
                        y_train = np.array(y_train)
                        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

                        # Build LSTM model
                        model = Sequential()
                        model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
                        model.add(LSTM(50, return_sequences=False))
                        model.add(Dense(25))
                        model.add(Dense(1))
                        model.compile(optimizer='adam', loss='mean_squared_error')

                        with st.spinner("Training the LSTM model (this may take a while)..."):
                            # Keep epochs small in Streamlit Cloud if you care about speed/limits
                            model.fit(x_train, y_train, batch_size=1, epochs=5, verbose=0)

                        # Prepare input sequence (last 60 days from full scaled_data)
                        last_60_days = scaled_data[-seq_len:, :]  # shape (60, 1)
                        input_seq = last_60_days.copy()

                        predictions = []
                        for _ in range(future_days):
                            X_test = input_seq[-seq_len:, :]  # ensures shape (60,1)
                            X_test = X_test.reshape(1, seq_len, 1)
                            pred_price = model.predict(X_test, verbose=0)[0][0]  # scalar
                            predictions.append(pred_price)

                            # append predicted value as new row (keep 2D shape)
                            new_row = np.array([[pred_price]])
                            input_seq = np.vstack((input_seq, new_row))
                            # keep only last seq_len rows for next iteration
                            if input_seq.shape[0] > seq_len:
                                input_seq = input_seq[-seq_len:, :]

                        # inverse scale predictions
                        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

                        future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=future_days)
                        prediction_df = pd.DataFrame({'Date': future_dates, 'Predicted Close': predictions.flatten()})

                        st.subheader("📅 Future Price Prediction")
                        st.dataframe(prediction_df.set_index('Date'))

                        fig_pred = go.Figure()
                        fig_pred.add_trace(go.Scatter(x=data.index, y=data['Close'], name="Historical"))
                        fig_pred.add_trace(go.Scatter(x=prediction_df['Date'], y=prediction_df['Predicted Close'], name="Prediction"))
                        fig_pred.update_layout(title=f"{predict_ticker} Price Prediction", xaxis_title="Date", yaxis_title="Price")
                        st.plotly_chart(fig_pred, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction error: {e}")
