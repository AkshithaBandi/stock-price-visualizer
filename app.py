# app.py
import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import pandas as pd
from utils import get_stock_data, moving_averages, lstm_predict_close
import alerts
from db import init_db, get_conn_cursor
from auth import login_user, register_user, logout

# AI keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

groq_client = None
openai_client = None

if GROQ_API_KEY:
    try:
        from groq import Groq
        groq_client = Groq(api_key=GROQ_API_KEY)
    except:
        pass

if OPENAI_API_KEY:
    try:
        import openai
        openai.api_key = OPENAI_API_KEY
        openai_client = openai
    except:
        pass

st.set_page_config(page_title="Stock Price Visualizer", layout="wide")

# Initialize DB
init_db()

# Session defaults
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

# ----------------------------
# LOGIN / SIGNUP PAGES
# ----------------------------
def login_page():
    st.title("🔐 Login")
    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")

    if st.button("Login"):
        user = login_user(username, password)
        if user:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success(f"Logged in as {username}")
            st.rerun()
        else:
            st.error("Invalid username or password")

def signup_page():
    st.title("📝 Signup")
    username = st.text_input("Username", key="signup_username")
    password = st.text_input("Password", type="password", key="signup_password")

    if st.button("Create Account"):
        if register_user(username, password):
            st.success("Account created! You can now login.")
        else:
            st.error("Username already exists.")

# ----------------------------
# SETTINGS PAGE
# ----------------------------
def settings_page():
    st.title("⚙️ Settings")
    st.subheader("Change Username")
    new_username = st.text_input("New Username", key="settings_username")
    if st.button("Update Username"):
        conn, cursor = get_conn_cursor()
        try:
            cursor.execute("UPDATE users SET username=? WHERE username=?",
                           (new_username, st.session_state.username))
            conn.commit()
            st.session_state.username = new_username
            st.success("Username updated successfully!")
        except Exception as e:
            st.error(f"Error: {e}")
        finally:
            conn.close()

# ----------------------------
# APP FLOW
# ----------------------------
if not st.session_state.logged_in:
    tab1, tab2 = st.tabs(["Login", "Signup"])
    with tab1:
        login_page()
    with tab2:
        signup_page()
    st.stop()

# Sidebar navigation
st.sidebar.title(f"Welcome, {st.session_state.username}")
page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "LSTM Predictor", "AI Recommendation", "Alerts", "Settings"]
)
st.sidebar.button("Logout", on_click=logout)

# ----------------------------
# DASHBOARD
# ----------------------------
if page == "Dashboard":
    st.title("📊 Stock Price Visualizer")
    ticker = st.text_input("Ticker:", "AAPL", key="dash_ticker")
    period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y"], key="dash_period")

    if st.button("Fetch Data", key="fetch_dashboard"):
        df = get_stock_data(ticker, period)
        if df is None or df.empty:
            st.error("No data found.")
        else:
            df = moving_averages(df)
            st.line_chart(df["Close"])

# ----------------------------
# LSTM PREDICTOR
# ----------------------------
elif page == "LSTM Predictor":
    st.title("🤖 LSTM Predictor")
    ticker = st.text_input("Ticker:", "AAPL", key="lstm_ticker")
    if st.button("Predict", key="predict_lstm"):
        preds = lstm_predict_close(ticker)
        st.write(preds)

# ----------------------------
# AI RECOMMENDATION
# ----------------------------
elif page == "AI Recommendation":
    st.title("🧠 AI Recommendation")
    ticker = st.text_input("Ticker:", "AAPL", key="ai_ticker")
    if st.button("Get Recommendation", key="ai_button"):
        df = get_stock_data(ticker, "1mo")
        if df is not None and not df.empty:
            last = df["Close"].iloc[-1]
            prev = df["Close"].iloc[-2]
            pct = (last - prev) / prev * 100
            prompt = f"Give Buy/Hold/Sell recommendation for {ticker}. Recent change: {pct:.2f}%."
            if groq_client:
                out = groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": prompt}]
                )
                st.write(out.choices[0].message.content)
            elif openai_client:
                response = openai_client.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}]
                )
                st.write(response.choices[0].message["content"])
            else:
                st.error("No AI API key found.")

# ----------------------------
# ALERTS
# ----------------------------
elif page == "Alerts":
    st.title("🔔 Alerts")
    t = st.text_input("Ticker:", "AAPL", key="alert_ticker")
    th = st.number_input("Alert Price:", value=200.0, key="alert_threshold")
    if st.button("Check Alert", key="check_alert"):
        st.write(alerts.get_price_alert(t, th))

# ----------------------------
# SETTINGS
# ----------------------------
elif page == "Settings":
    settings_page()
