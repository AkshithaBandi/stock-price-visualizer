# settings.py
import streamlit as st
from db import get_conn_cursor

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
