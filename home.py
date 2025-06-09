import streamlit as st

def show_home_page():
    try:
        st.title("Welcome to the Depression Detection System")

        with st.container():
            st.markdown("""
                ### About This Tool
                This tool helps assess your mental health by analyzing your responses to standardized questions.
            """)

            st.subheader("How It Works")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("""
                    #### Step 1: Mood Assessment
                    - Answer 9 PHQ-9 questions
                    - Choose how often you experience each symptom
                    - Be honest with your responses
                """)

            with col2:
                st.markdown("""
                    #### Step 2: Open-Ended Questions
                    - Share thoughts in your own words
                    - Describe how you’ve been feeling
                    - Mention any specific concerns
                """)

            with col3:
                st.markdown("""
                    #### Step 3: Results & Support
                    - View a summary of your responses
                    - Receive helpful suggestions
                    - Optionally share with a healthcare provider
                """)

            st.markdown("---")
            st.markdown("""
                > **Note:** This tool does not replace a professional diagnosis.  
                > If you're experiencing severe symptoms, please seek help from a licensed mental health provider.
            """)

            if st.button("Start PHQ-9 Assessment", type="primary"):
                st.session_state["page"] = "PHQ-9"
                st.rerun()

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.button("Retry", on_click=lambda: st.rerun())

if __name__ == "__main__":
    show_home_page()
