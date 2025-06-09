import streamlit as st
from pathlib import Path
from streamlit_option_menu import option_menu
from dotenv import load_dotenv
import nltk

from login import show_login_page
from register import show_register_page
from home import show_home_page
from phq9 import show_phq9_page
from extra_questions import show_extra_questions
from suggestions import show_suggestions
from db_utils import init_db

st.set_page_config(page_title="Depression Detection System", layout="wide")

# --- Centralized Styling ---
def load_css():
    """Loads all CSS from the styles.css file."""
    css_file = Path(__file__).parent / "styles.css"
    if css_file.exists():
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- Session State Initialization ---
def init_session_state():
    """Initializes session state with default values."""
    defaults = {
        "authenticated": False,
        "username": None,
        "user_id": None,
        "page": "Home",
        # Flags to control assessment flow and data loading
        "phq9_progress_loaded": False,
        "extra_progress_loaded": False
    }
    for key, val in defaults.items():
        st.session_state.setdefault(key, val)

# --- Main Application ---
def main():
    """Main function to run the Streamlit application."""
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
    try:
        nltk.data.find('sentiment/vader_lexicon')
    except LookupError:
        nltk.download('vader_lexicon')
    load_dotenv()
    load_css()
    init_db()
    init_session_state()

    if not st.session_state.authenticated:
        # Display Login/Register pages
        with st.sidebar:
            selected = option_menu(
                "Welcome", ["Login", "Register"],
                icons=["box-arrow-in-right", "person-plus"], menu_icon="cast", default_index=0
            )
        if selected == "Login":
            show_login_page()
        else:
            show_register_page()
    else:
        # Display main application for authenticated users
        render_main_app()

def render_main_app():
    """Renders the main application interface for a logged-in user."""
    with st.sidebar:
        st.subheader(f"Welcome, {st.session_state.username}!")
        
        # Dynamically build navigation based on user progress
        page_options = ["Home", "PHQ-9"]
        page_icons = ["house", "clipboard-check"]
        if st.session_state.get("phq9_completed"):
            page_options.append("Extended Assessment")
            page_icons.append("file-earmark-text")
        if st.session_state.get("extra_completed"):
            page_options.append("Report & Suggestions")
            page_icons.append("graph-up-arrow")
            
        default_page_key = st.session_state.get("page", "Home")
        default_index = page_options.index(default_page_key) if default_page_key in page_options else 0

        selected = option_menu(
            "Navigation", page_options,
            icons=page_icons, menu_icon="compass", default_index=default_index
        )
        
        st.session_state.page = selected # Update page state

        st.markdown("---")
        if st.button("Logout"):
            # Clear all session state keys on logout
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # Routing to the selected page
    page_map = {
        "Home": show_home_page,
        "PHQ-9": show_phq9_page,
        "Extended Assessment": show_extra_questions,
        "Report & Suggestions": show_suggestions
    }
    page_function = page_map.get(st.session_state.page, show_home_page)
    page_function()

if __name__ == "__main__":
    main()
