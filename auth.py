# auth.py
import hashlib
from db import get_conn_cursor
import streamlit as st

# -----------------
# Helper Functions
# -----------------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password):
    conn, cursor = get_conn_cursor()
    try:
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)",
                       (username, hash_password(password)))
        conn.commit()
        return True
    except:
        return False
    finally:
        conn.close()

def login_user(username, password):
    conn, cursor = get_conn_cursor()
    try:
        cursor.execute("SELECT * FROM users WHERE username=? AND password=?",
                       (username, hash_password(password)))
        return cursor.fetchone()
    finally:
        conn.close()

def logout():
    st.session_state.logged_in = False
    st.session_state.username = ""
