import streamlit as st
from db_utils import get_db_connection
import psycopg2
import re
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def validate_password(password):
    """
    Validates the password against a set of security rules.
    Returns (True, None) if valid, or (False, list_of_errors) if invalid.
    """
    errors = []
    if not (8 <= len(password) <= 16):
        errors.append("• Must be between 8 and 16 characters long.")

    if not re.search(r"[A-Z]", password):
        errors.append("• Must contain at least one uppercase letter (A-Z).")
        
    if not re.search(r"[a-z]", password):
        errors.append("• Must contain at least one lowercase letter (a-z).")
        
    if not re.search(r"\d", password):
        errors.append("• Must contain at least one number (0-9).")

    if not re.search(r"[!@#$%^&*()_+\-=\[\]{};':\"\\|,.<>\/?~`]", password):
        errors.append("• Must contain at least one special character (e.g., !@#$).")
        
    return (False, errors) if errors else (True, None)

def show_register_page():
    """Displays the user registration form and delegates submission handling."""
    st.title("Create Your Account")
    
    # UX IMPROVEMENT: Help text to guide the user on password requirements.
    password_help_text = """
    Your password must contain:
    - 8-16 characters
    - At least one uppercase letter (A-Z)
    - At least one lowercase letter (a-z)
    - At least one number (0-9)
    - At least one special character (e.g., !@#$%)
    """

    with st.form("registration_form"):
        st.info("All fields are required to create an account.")
        full_name = st.text_input("Full Name", placeholder="Enter your full name")

        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=13, max_value=110, step=1, value=22)
        with col2:
            gender = st.selectbox("Gender", ["Male", "Female", "Prefer not to say"])

        username = st.text_input("Username", placeholder="Choose a unique username")
        
        col3, col4 = st.columns(2)
        with col3:
            password = st.text_input("Password", type="password", placeholder="Create a strong password", help=password_help_text)
        with col4:
            confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm your password")

        submit_button = st.form_submit_button("Register")

    if submit_button:
        handle_registration(full_name, age, gender, username, password, confirm_password)

def handle_registration(full_name, age, gender, username, password, confirm_password):
    """Validates form data and calls the user creation function."""
    if not all([full_name, age, gender, username, password, confirm_password]):
        st.error("Please fill in all fields.")
        return

    if password != confirm_password:
        st.error("Passwords do not match. Please try again.")
        return

    is_valid, error_messages = validate_password(password)
    if not is_valid:
        st.error("Your password is not strong enough. Please fix the following issues:\n\n" + "\n".join(error_messages))
        return

    if register_user(full_name, age, gender, username, password):
        st.session_state["registration_complete"] = True
        st.session_state["page"] = "Login"
        st.balloons()
        st.rerun()

def register_user(full_name, age, gender, username, password):
    """Hashes the password and inserts the new user into the database."""
    try:
        conn = get_db_connection()
        if not conn:
            st.error("Database connection is not available. Please try again later.")
            return False
            
        hashed_password = pwd_context.hash(password)
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1 FROM users WHERE username = %s", (username,))
            if cursor.fetchone():
                st.error("Username already exists. Please choose a different one.")
                return False
            
            cursor.execute(
                """
                INSERT INTO users (full_name, age, gender, username, password)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (full_name, age, gender, username, hashed_password)
            )
            conn.commit()
            return True
    except psycopg2.IntegrityError:
        if conn: conn.rollback()
        st.error("Username already exists. Please choose a different one.")
    except Exception as e:
        if conn: conn.rollback()
        st.error(f"An unexpected error occurred during registration: {e}")
    return False