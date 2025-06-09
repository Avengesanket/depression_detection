import streamlit as st
from db_utils import get_db_connection
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def authenticate_user(username, password):
    conn = get_db_connection()
    if not conn or conn.closed:
        st.error("Database connection unavailable.")
        return False, None
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT id, password FROM users WHERE username = %s", (username,))
            result = cursor.fetchone()
        if result:
            user_id, hashed_password = result
            if pwd_context.verify(password, hashed_password):
                return True, user_id
    except Exception as e:
        st.error(f"Database error: {e}")
    return False, None

def show_login_page():
    if st.session_state.get("registration_complete"):
        st.success("Registration successful! You may now log in.")
        del st.session_state["registration_complete"]
    st.title("Login")
    st.markdown('<div class="form-container">', unsafe_allow_html=True)

    with st.form("login_form"):
        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", placeholder="Enter your password")
        login_btn = st.form_submit_button("Login")
    st.markdown('</div>', unsafe_allow_html=True)

    if login_btn:
        if username and password:
            valid, user_id = authenticate_user(username, password)
            if valid:
                st.session_state.update({
                    "authenticated": True,
                    "username": username,
                    "user_id": user_id
                })
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid username or password.")
        else:
            st.warning("Please enter both username and password.")
