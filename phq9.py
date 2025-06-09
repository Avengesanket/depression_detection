import streamlit as st
from db_utils import (
    save_phq9_response,
    get_saved_phq9_responses,
    delete_phq9_responses,
    delete_extra_assessment_data,
    delete_user_suggestions
)

def show_phq9_page():
    """
    Displays the PHQ-9 assessment, handles progress saving, and shows results.
    """
    st.title("🧠 PHQ-9 Depression Assessment")

    user_id = st.session_state.get("user_id")
    if not user_id:
        st.warning("⚠️ You must be logged in to take the assessment.")
        return

    # --- Initialize session state variables ---
    st.session_state.setdefault("phq9_completed", False)
    st.session_state.setdefault("scores", [0] * 9)
    st.session_state.setdefault("current_question", 0)
    st.session_state.setdefault("phq9_progress_loaded", False)

    questions = [
        "1. Little interest or pleasure in doing things",
        "2. Feeling down, depressed, or hopeless",
        "3. Trouble falling or staying asleep, or sleeping too much",
        "4. Feeling tired or having little energy",
        "5. Poor appetite or overeating",
        "6. Feeling bad about yourself — or that you are a failure or have let yourself or your family down",
        "7. Trouble concentrating on things, such as reading or watching television",
        "8. Moving or speaking so slowly that other people could have noticed, or being so fidgety or restless that you've been moving a lot more than usual",
        "9. Thoughts that you would be better off dead, or thoughts of hurting yourself"
    ]
    options = [
        "Not at all",
        "Several days",
        "More than half the days",
        "Nearly every day"
    ]

    # --- Load user's progress from DB (only once per session) ---
    if not st.session_state.phq9_progress_loaded:
        saved_responses = get_saved_phq9_responses(user_id)
        if saved_responses:
            for q_idx, value in saved_responses.items():
                if 0 <= q_idx < 9:
                    st.session_state.scores[q_idx] = value
            
            if len(saved_responses) == 9:
                st.session_state.phq9_completed = True
                st.session_state.phq9_score = sum(st.session_state.scores)
            else:
                st.session_state.current_question = len(saved_responses)
        
        st.session_state.phq9_progress_loaded = True
        st.rerun()

    # --- Display Results or Questions based on completion status ---
    if st.session_state.phq9_completed:
        display_phq9_results()
    else:
        display_phq9_questions(questions, options)

def get_severity(score):
    """Returns the depression severity category based on the PHQ-9 score."""
    if score <= 4:
        return "Minimal or No Depression"
    elif score <= 9:
        return "Mild Depression"
    elif score <= 14:
        return "Moderate Depression"
    elif score <= 19:
        return "Moderately Severe Depression"
    else:
        return "Severe Depression"

def display_phq9_results():
    """Shows the final score, interpretation, and next steps."""
    total_score = sum(st.session_state.scores)
    severity = get_severity(total_score)
    st.session_state["phq9_score"] = total_score

    st.success("✅ PHQ-9 Assessment Completed!")
    st.metric(label="🧾 Your PHQ-9 Score", value=total_score)
    st.info(f"**Interpretation**: This score suggests **{severity}**.")

    st.markdown("---")
    st.info("Your responses have been automatically saved.")

    if st.button("➡ Continue to Additional Questions", type="primary"):
        st.session_state["page"] = "Extended Assessment"
        st.rerun()

    with st.expander("⚠️ Manage Your Data"):
        st.warning("This will permanently delete ALL your assessment data (PHQ-9, Extended Questions, and Suggestions). This action cannot be undone.")
        if st.button("Delete All My Assessment Data", type="primary"):
            user_id = st.session_state.get("user_id")
            
            delete_phq9_responses(user_id)
            delete_extra_assessment_data(user_id)
            delete_user_suggestions(user_id)

            st.success("Your assessment data has been successfully deleted.")
            
            keys_to_clear = [
                "phq9_completed", "phq9_score", "scores", "current_question",
                "phq9_progress_loaded", "extra_completed", "extra_assessment_submitted", "page"
            ]
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            
            st.balloons()
            st.rerun()

def display_phq9_questions(questions, options):
    """Displays the current PHQ-9 question and navigation buttons."""
    user_id = st.session_state.user_id
    idx = st.session_state.current_question

    st.markdown("##### Over the last **2 weeks**, how often have you been bothered by the following problem?")
    st.progress((idx + 1) / len(questions), text=f"Question {idx + 1} of {len(questions)}")
    st.subheader(questions[idx])

    current_score = st.session_state.scores[idx]
    
    selected_option = st.radio(
        "Select your answer:",
        options=list(range(len(options))),
        format_func=lambda x: f"{options[x]} ({x})",
        index=current_score,
        key=f"radio_q{idx}",
        horizontal=True
    )
    
    st.session_state.scores[idx] = selected_option

    # --- Navigation Buttons ---
    col1, col2, col3 = st.columns([1, 4, 1])
    with col1:
        if idx > 0:
            if st.button("⬅ Previous", use_container_width=True):
                st.session_state.current_question -= 1
                st.rerun()

    with col3:
        if idx == len(questions) - 1:
            if st.button("✅ Submit", type="primary", use_container_width=True):
                save_phq9_response(user_id, idx, selected_option)
                st.session_state.phq9_completed = True
                st.rerun()
        else:
            if st.button("Next ➡", type="primary", use_container_width=True):
                save_phq9_response(user_id, idx, selected_option)
                st.session_state.current_question += 1
                st.rerun()