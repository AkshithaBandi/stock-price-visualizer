import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import numpy as np

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

def fetch_data(ticker):
    df = yf.download(ticker, start=start_date, end=end_date)
    if not df.empty and show_ma:
        df[f"SMA_{sma_period}"] = df['Close'].rolling(window=sma_period).mean()
        df[f"EMA_{ema_period}"] = df['Close'].ewm(span=ema_period, adjust=False).mean()
    return df

# === Visualization Section ===
if st.button("Visualize"):
    try:
        df1 = fetch_data(ticker1)
        if df1.empty:
            st.warning(f"No data found for {ticker1}")
        else:
            st.subheader(f"📊 Stock Chart for {ticker1}")
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=df1.index, y=df1['Close'], name="Close Price"))
            if show_ma:
                fig1.add_trace(go.Scatter(x=df1.index, y=df1[f"SMA_{sma_period}"], name=f"SMA {sma_period}"))
                fig1.add_trace(go.Scatter(x=df1.index, y=df1[f"EMA_{ema_period}"], name=f"EMA {ema_period}"))
            fig1.update_layout(title=f"{ticker1} Closing Price with MA", xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(fig1, use_container_width=True)

            # Volume Chart
            fig_vol = go.Figure()
            fig_vol.add_trace(go.Bar(x=df1.index, y=df1['Volume'], name="Volume"))
            fig_vol.update_layout(title=f"{ticker1} Trading Volume", xaxis_title="Date", yaxis_title="Volume")
            st.plotly_chart(fig_vol, use_container_width=True)

        if ticker2:
            df2 = fetch_data(ticker2)
            if df2.empty:
                st.warning(f"No data found for {ticker2}")
            else:
                st.subheader(f"🔁 Comparison: {ticker1} vs {ticker2}")
                fig_compare = go.Figure()
                fig_compare.add_trace(go.Scatter(x=df1.index, y=df1['Close'], name=ticker1))
                fig_compare.add_trace(go.Scatter(x=df2.index, y=df2['Close'], name=ticker2))
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
        df = yf.download(predict_ticker, start="2015-01-01", end=pd.to_datetime("today"))
        data = df.filter(['Close'])
        dataset = data.values

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)

        training_data_len = int(np.ceil(len(dataset) * 0.8))
        train_data = scaled_data[0:training_data_len, :]

        x_train = []
        y_train = []
        for i in range(60, len(train_data)):
            x_train.append(train_data[i-60:i, 0])
            y_train.append(train_data[i, 0])

        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')

        with st.spinner("Training the LSTM model..."):
            model.fit(x_train, y_train, batch_size=1, epochs=5, verbose=0)

        last_60_days = scaled_data[-60:]
        input_seq = last_60_days
        predictions = []

        for _ in range(future_days):
            X_test = []
            X_test.append(input_seq[-60:, 0])
            X_test = np.array(X_test)
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
            pred_price = model.predict(X_test, verbose=0)
            predictions.append(pred_price[0][0])
            input_seq = np.append(input_seq, pred_price)[-60:]

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
        st.error(f"Prediction error: {e}") # THIS IS MY APP.PY COD

