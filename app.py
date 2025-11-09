# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import io

st.set_page_config(page_title="Advanced Stock Price Visualizer", layout="wide")
st.title("🚀 Advanced Stock Price Visualizer — All in One")

# -------------------------
# Helper indicator functions
# -------------------------
def sma(series, period):
    return series.rolling(window=period).mean()

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(window=period).mean()
    ma_down = down.rolling(window=period).mean()
    rs = ma_up / (ma_down + 1e-9)
    return 100 - (100 / (1 + rs))

def macd(series, n_fast=12, n_slow=26, n_signal=9):
    ema_fast = ema(series, n_fast)
    ema_slow = ema(series, n_slow)
    macd_line = ema_fast - ema_slow
    signal = macd_line.ewm(span=n_signal, adjust=False).mean()
    hist = macd_line - signal
    return macd_line, signal, hist

def bollinger_bands(series, period=20, n_std=2):
    ma = sma(series, period)
    std = series.rolling(window=period).std()
    upper = ma + (std * n_std)
    lower = ma - (std * n_std)
    return upper, lower

# -------------------------
# Cached data fetcher
# -------------------------
@st.cache_data(ttl=60*30)
def download_data(ticker, start_date, end_date):
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if df.empty:
            return pd.DataFrame()
        df.index = pd.to_datetime(df.index)
        return df
    except Exception:
        return pd.DataFrame()

# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.header("Controls")
tickers_input = st.sidebar.text_input("Enter up to 3 tickers (comma-separated)", "AAPL, GOOGL")
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()][:3]
col1, col2 = st.sidebar.columns(2)
start_date = col1.date_input("Start date", pd.to_datetime("2021-01-01"))
end_date = col2.date_input("End date", pd.to_datetime("today"))

# indicators
st.sidebar.markdown("**Indicators**")
show_sma = st.sidebar.checkbox("SMA", True)
sma_period = st.sidebar.slider("SMA period", 5, 200, 20)
show_ema = st.sidebar.checkbox("EMA", False)
ema_period = st.sidebar.slider("EMA period", 5, 200, 20)
show_rsi = st.sidebar.checkbox("RSI", False)
rsi_period = st.sidebar.slider("RSI period", 5, 50, 14)
show_macd = st.sidebar.checkbox("MACD", False)
show_bbands = st.sidebar.checkbox("Bollinger Bands", False)
bb_period = st.sidebar.slider("BBands period", 10, 50, 20)

# prediction settings
st.sidebar.markdown("---")
st.sidebar.subheader("Prediction (LSTM)")
predict_ticker = st.sidebar.selectbox("Ticker to predict", options=tickers if tickers else ["AAPL"])
future_days = st.sidebar.slider("Days to predict into future", 1, 30, 7)
train_epochs = st.sidebar.slider("LSTM epochs (small recommended)", 1, 10, 3)
train_batch = st.sidebar.selectbox("Batch size", [1, 8, 16, 32], index=2)

# -------------------------
# Fetch all data
# -------------------------
if not tickers:
    st.error("Enter at least one ticker symbol in the sidebar.")
    st.stop()

data_dict = {}
for t in tickers:
    df = download_data(t, start_date, end_date)
    if df.empty:
        st.warning(f"No data for {t}. It may be an invalid ticker or no data in date range.")
    data_dict[t] = df

# ----- Tabs layout -----
tabs = st.tabs(["Dashboard", "Indicators", "Comparison", "Correlation", "Prediction", "Export"])

# -------------------------
# Dashboard Tab
# -------------------------
with tabs[0]:
    st.header("Overview Dashboard")
    # show latest price cards
    cols = st.columns(len(tickers))
    # --- Display Latest Stock Prices Safely ---
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


    # interactive chart for selected ticker (first by default)
    show_t = st.selectbox("Show ticker chart", tickers, index=0)
    df_show = data_dict.get(show_t)
    if df_show is None or df_show.empty:
        st.warning("No data to plot.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df_show.index, open=df_show['Open'], high=df_show['High'],
                                     low=df_show['Low'], close=df_show['Close'], name=show_t))
        if show_sma:
            fig.add_trace(go.Scatter(x=df_show.index, y=sma(df_show['Close'], sma_period), name=f"SMA {sma_period}"))
        if show_ema:
            fig.add_trace(go.Scatter(x=df_show.index, y=ema(df_show['Close'], ema_period), name=f"EMA {ema_period}"))
        if show_bbands:
            upper, lower = bollinger_bands(df_show['Close'], bb_period)
            fig.add_trace(go.Scatter(x=df_show.index, y=upper, name="BB Upper", line=dict(dash='dash')))
            fig.add_trace(go.Scatter(x=df_show.index, y=lower, name="BB Lower", line=dict(dash='dash')))

        fig.update_layout(title=f"{show_t} Price Chart", xaxis_title="Date", yaxis_title="Price (USD)",
                          height=600, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Indicators Tab
# -------------------------
with tabs[1]:
    st.header("Technical Indicators")
    chosen = st.selectbox("Select ticker for indicators", tickers, index=0)
    df_ind = data_dict.get(chosen)
    if df_ind is None or df_ind.empty:
        st.warning("No data.")
    else:
        ind_df = df_ind[['Close']].copy()
        if show_sma:
            ind_df[f"SMA_{sma_period}"] = sma(ind_df['Close'], sma_period)
        if show_ema:
            ind_df[f"EMA_{ema_period}"] = ema(ind_df['Close'], ema_period)
        if show_rsi:
            ind_df[f"RSI_{rsi_period}"] = rsi(ind_df['Close'], rsi_period)
        if show_macd:
            macd_line, macd_signal, macd_hist = macd(ind_df['Close'])
            ind_df['MACD'] = macd_line
            ind_df['MACD_Signal'] = macd_signal
            ind_df['MACD_Hist'] = macd_hist
        if show_bbands:
            upper, lower = bollinger_bands(ind_df['Close'], bb_period)
            ind_df['BB_upper'] = upper
            ind_df['BB_lower'] = lower

        st.dataframe(ind_df.tail(100), use_container_width=True)

        # Plot RSI and MACD below price
        fig_price = px.line(ind_df, x=ind_df.index, y='Close', title=f"{chosen} Close Price")
        if show_sma:
            fig_price.add_scatter(x=ind_df.index, y=ind_df[f"SMA_{sma_period}"], name=f"SMA {sma_period}")
        st.plotly_chart(fig_price, use_container_width=True)

        if show_rsi:
            fig_rsi = px.line(ind_df, x=ind_df.index, y=f"RSI_{rsi_period}", title="RSI")
            fig_rsi.update_yaxes(range=[0, 100])
            st.plotly_chart(fig_rsi, use_container_width=True)

        if show_macd:
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Bar(x=ind_df.index, y=ind_df['MACD_Hist'], name='MACD Hist'))
            fig_macd.add_trace(go.Scatter(x=ind_df.index, y=ind_df['MACD'], name='MACD Line'))
            fig_macd.add_trace(go.Scatter(x=ind_df.index, y=ind_df['MACD_Signal'], name='Signal'))
            st.plotly_chart(fig_macd, use_container_width=True)

# -------------------------
# Comparison Tab
# -------------------------
with tabs[2]:
    st.header("Multi-stock Comparison")
    # build close price DF for all tickers
    close_frames = []
    for t in tickers:
        df = data_dict.get(t)
        if df is not None and not df.empty:
            close_frames.append(df['Close'].rename(t))
    if not close_frames:
        st.warning("No data for comparison.")
    else:
        combined = pd.concat(close_frames, axis=1).dropna()
        st.line_chart(combined)

        # normalize (index) comparison option
        if st.checkbox("Normalize to 100 at start", value=False):
            norm = combined / combined.iloc[0] * 100
            st.line_chart(norm)

# -------------------------
# Correlation Tab
# -------------------------
with tabs[3]:
    st.header("Correlation Matrix")
    close_frames = []
    for t in tickers:
        df = data_dict.get(t)
        if df is not None and not df.empty:
            close_frames.append(df['Close'].rename(t))
    if len(close_frames) < 2:
        st.info("At least 2 tickers with data are required for correlation.")
    else:
        combined = pd.concat(close_frames, axis=1).dropna()
        corr = combined.corr()
        fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', title="Price Correlation")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(corr, use_container_width=True)

# -------------------------
# Prediction Tab (LSTM)
# -------------------------
with tabs[4]:
    st.header("LSTM Forecasting")
    if predict_ticker not in tickers or not tickers:
        st.warning("Prediction ticker must be one of the tickers in the sidebar. Adjust sidebar selection.")
    else:
        df_pred = data_dict.get(predict_ticker)
        if df_pred is None or df_pred.empty:
            st.warning("No data to predict for selected ticker.")
        else:
            st.write(f"Predicting for {predict_ticker} using historical Close prices.")
            # prepare data
            close_vals = df_pred['Close'].values.reshape(-1, 1)
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled = scaler.fit_transform(close_vals)

            seq_len = 60
            if len(scaled) < seq_len + 1:
                st.error("Not enough historical data for LSTM (need > 60 days).")
            else:
                # build sequences using last 80% as train
                train_size = int(len(scaled) * 0.8)
                train = scaled[:train_size]
                X_train, y_train = [], []
                for i in range(seq_len, len(train)):
                    X_train.append(train[i-seq_len:i, 0])
                    y_train.append(train[i, 0])

                X_train = np.array(X_train)
                y_train = np.array(y_train)
                X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

                # build model
                model = Sequential()
                model.add(LSTM(50, return_sequences=True, input_shape=(seq_len, 1)))
                model.add(LSTM(50, return_sequences=False))
                model.add(Dense(25))
                model.add(Dense(1))
                model.compile(optimizer='adam', loss='mean_squared_error')

                with st.spinner("Training LSTM (this runs in app; keep epochs small)..."):
                    model.fit(X_train, y_train, epochs=train_epochs, batch_size=train_batch, verbose=0)

                # prepare input sequence (last seq_len from full scaled)
                input_seq = scaled[-seq_len:].reshape(seq_len, 1).copy()
                preds_scaled = []
                for _ in range(future_days):
                    X_test = input_seq[-seq_len:].reshape(1, seq_len, 1)
                    pred = model.predict(X_test, verbose=0)[0][0]
                    preds_scaled.append(pred)
                    # append and keep rolling window
                    input_seq = np.vstack([input_seq, [[pred]]])
                    if input_seq.shape[0] > seq_len:
                        input_seq = input_seq[-seq_len:, :]

                preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
                future_index = pd.date_range(start=df_pred.index[-1] + pd.Timedelta(days=1), periods=future_days)
                pred_df = pd.DataFrame({'Date': future_index, 'Predicted_Close': preds}).set_index('Date')
                st.subheader("Future Predictions")
                st.dataframe(pred_df)

                # show on chart: historical last N + predictions
                hist_to_show = df_pred['Close'].tail(200).copy()
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=hist_to_show.index, y=hist_to_show.values, name='Historical'))
                fig.add_trace(go.Scatter(x=pred_df.index, y=pred_df['Predicted_Close'], name='Predicted', line=dict(dash='dash')))
                fig.update_layout(title=f"{predict_ticker} Historical + Predicted", xaxis_title="Date", yaxis_title="Price (USD)")
                st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Export Tab
# -------------------------
with tabs[5]:
    st.header("Export Data")
    # combine and allow downloading combined close prices and indicators for chosen ticker
    export_ticker = st.selectbox("Choose ticker to export data", tickers, index=0)
    df_export = data_dict.get(export_ticker)
    if df_export is None or df_export.empty:
        st.warning("No data to export.")
    else:
        export_df = df_export.copy()
        if show_sma:
            export_df[f"SMA_{sma_period}"] = sma(export_df['Close'], sma_period)
        if show_ema:
            export_df[f"EMA_{ema_period}"] = ema(export_df['Close'], ema_period)
        if show_rsi:
            export_df[f"RSI_{rsi_period}"] = rsi(export_df['Close'], rsi_period)
        if show_macd:
            macd_line, macd_signal, macd_hist = macd(export_df['Close'])
            export_df['MACD'] = macd_line
            export_df['MACD_Signal'] = macd_signal
            export_df['MACD_Hist'] = macd_hist
        if show_bbands:
            upper, lower = bollinger_bands(export_df['Close'], bb_period)
            export_df['BB_upper'] = upper
            export_df['BB_lower'] = lower

        st.dataframe(export_df.tail(50), use_container_width=True)

        csv = export_df.to_csv().encode('utf-8')
        st.download_button("Download CSV", data=csv, file_name=f"{export_ticker}_data.csv", mime='text/csv')

# Footer
st.markdown("---")
st.caption("Advanced Stock Price Visualizer • Built with Streamlit, yfinance, Plotly, TensorFlow (LSTM).")
