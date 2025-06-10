import streamlit as st
from db_utils import (
    save_extra_assessment_data,
    get_extra_assessment_data,
    get_user_profile,
    delete_extra_assessment_data,
    delete_user_suggestions
)

# A constant list of all keys ensures secure and explicit data handling
ASSESSMENT_KEYS = [
    "employment", "living_situation", "mood_patterns_input", "quality_of_life_slider",
    "life_satisfaction_input", "sleep_quality_radio", "sleep_disturbances_multi",
    "regular_bedtime_radio", "bedtime_routine_input", "energy_level_radio",
    "energy_fluctuation_input", "eating_patterns_radio", "eating_details_input",
    "weight_changes_radio", "weight_details_input", "concentration_radio",
    "concentration_details_input", "memory_issues_radio", "suicidal_plan_radio",
    "suicidal_intent_radio", "protective_factors_input"
]

def show_extra_questions():
    """
    Manages the extended assessment page, including data loading, form display, and submission.
    """
    st.title("Extended Mental Health Assessment")

    user_id = st.session_state.get("user_id")
    username = st.session_state.get("username", "User")

    if not user_id:
        st.error("User not identified. Please log in again.")
        return

    # Load data once per session to populate state for widgets
    if not st.session_state.get("extra_progress_loaded", False):
        load_and_set_progress(user_id)
        st.session_state["extra_progress_loaded"] = True
        
    phq9_scores = st.session_state.get("scores", [])
    if not phq9_scores or len(phq9_scores) != 9:
        st.warning("PHQ-9 scores not found. Please complete the PHQ-9 assessment first.")
        if st.button("Go to PHQ-9 Assessment"):
            st.session_state.page = "PHQ-9"
            st.rerun()
        return
    
    load_user_profile(username)

    if st.session_state.get("extra_completed"):
        display_completed_view()
    
    display_assessment_form(phq9_scores)

def load_and_set_progress(user_id):
    """Loads saved assessment data from the DB and populates session state."""
    record = get_extra_assessment_data(user_id)
    if record and record.get("assessment_data"):
        data = record["assessment_data"]
        for key, value in data.items():
            if key in ASSESSMENT_KEYS: 
                st.session_state[key] = value
        st.session_state["extra_completed"] = True

def load_user_profile(username):
    """Fetches user profile if not already in session state."""
    if "age" not in st.session_state or "gender" not in st.session_state:
        profile = get_user_profile(username)
        if profile:
            st.session_state["age"] = profile.get("age")
            st.session_state["gender"] = profile.get("gender")

def display_completed_view():
    """Shows a message for a completed assessment and data management options."""
    st.success("You have already completed the extended assessment.")
    st.info("You can review or update your answers below and resubmit if needed.")
    if st.button("Continue to Report & Suggestions"):
        st.session_state.page = "Report & Suggestions"
        st.rerun()
    
    with st.expander("⚠️ Manage Your Data"):
        st.info("This will delete your answers to these extended questions and the final suggestions. Your PHQ-9 score will be kept.")
        if st.button("Delete Extended Assessment Responses", type="primary"):
            user_id = st.session_state.get("user_id")
            extra_deleted = delete_extra_assessment_data(user_id)
            delete_user_suggestions(user_id) 

            if extra_deleted:
                st.success("Your extended assessment data has been deleted.")
                keys_to_clear = ["extra_completed", "extra_progress_loaded"] + ASSESSMENT_KEYS
                for key in keys_to_clear:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
            else:
                st.error("There was an issue deleting your data.")

def display_assessment_form(phq9_scores):
    """Builds and displays the dynamic questionnaire form."""
    # Determine which sections to show based on PHQ-9 scores
    show_sleep = phq9_scores[2] >= 1
    show_energy = phq9_scores[3] >= 1
    show_appetite = phq9_scores[4] >= 1
    show_concentration = phq9_scores[6] >= 1
    show_suicidal = phq9_scores[8] > 0
    
    with st.form("extra_questions_form"):
        with st.expander("Confirm Profile & Additional Details", expanded=True):
            st.write(f"**Age:** {st.session_state.get('age', 'N/A')}")
            st.write(f"**Gender:** {st.session_state.get('gender', 'N/A')}")
            
            emp_opts = ["Student", "Working Full-Time", "Working Part-Time", "Unemployed", "Retired", "Homemaker", "Other", "Prefer not to say"]
            liv_opts = ["Alone", "With partner/spouse", "With family", "With children", "With roommates", "Other", "Prefer not to say"]
            
            st.selectbox("Current Employment Status:", emp_opts, index=emp_opts.index(st.session_state.get("employment", "Student")), key="employment")
            st.selectbox("Current Living Situation:", liv_opts, index=liv_opts.index(st.session_state.get("living_situation", "Alone")), key="living_situation")

        st.markdown("---")
        st.info("Please answer the following to help us understand your situation better.")
        
        st.header("Emotional Patterns & Well-being")
        st.text_area("1. Describe your typical mood fluctuations over the past 2 weeks.", key="mood_patterns_input", value=st.session_state.get("mood_patterns_input", ""))
        st.slider("2. On a scale of 1 to 10, rate your overall quality of life recently.", 1, 10, key="quality_of_life_slider", value=st.session_state.get("quality_of_life_slider", 5))
        st.text_area("3. What brings you joy or feels challenging in life?", key="life_satisfaction_input", value=st.session_state.get("life_satisfaction_input", ""))
        
        # --- CORRECTION: Re-added all conditional widgets to complete the form ---
        if show_sleep:
            st.divider()
            st.header("Sleep Details")
            sleep_opts = ["Very unsatisfied", "Unsatisfied", "Neutral", "Satisfied", "Very satisfied"]
            st.radio("Satisfaction with sleep?", sleep_opts, index=sleep_opts.index(st.session_state.get("sleep_quality_radio", "Neutral")), key="sleep_quality_radio", horizontal=True)
            st.multiselect("Sleep difficulties experienced:", ["Trouble falling asleep", "Waking up frequently", "Waking too early", "Sleeping too much", "Nightmares", "Restless sleep"], key="sleep_disturbances_multi", default=st.session_state.get("sleep_disturbances_multi", []))
            st.radio("Do you have a regular bedtime?", ["Yes", "No"], key="regular_bedtime_radio", index=["Yes", "No"].index(st.session_state.get("regular_bedtime_radio", "No")), horizontal=True)
            if st.session_state.get("regular_bedtime_radio") == "Yes":
                st.text_area("Describe your typical bedtime routine:", key="bedtime_routine_input", value=st.session_state.get("bedtime_routine_input", ""))

        if show_energy:
            st.divider()
            st.header("Energy & Functioning")
            energy_opts = ["Never", "Rarely", "Sometimes", "Often", "Nearly daily"]
            st.radio("How often do you feel tired or drained?", energy_opts, index=energy_opts.index(st.session_state.get("energy_level_radio", "Sometimes")), key="energy_level_radio", horizontal=True)
            st.text_area("Describe any patterns in your energy levels:", key="energy_fluctuation_input", value=st.session_state.get("energy_fluctuation_input", ""))

        if show_appetite:
            st.divider()
            st.header("Appetite & Physical Health")
            appetite_opts = ["Increased", "Decreased", "Fluctuating", "No change"]
            st.radio("Changes in appetite?", appetite_opts, index=appetite_opts.index(st.session_state.get("eating_patterns_radio", "No change")), key="eating_patterns_radio", horizontal=True)
            if st.session_state.get("eating_patterns_radio") != "No change":
                st.text_area("Describe the appetite changes:", key="eating_details_input", value=st.session_state.get("eating_details_input", ""))
            
            weight_opts = ["Significant gain", "Significant loss", "Minor changes", "No change"]
            st.radio("Unintentional weight changes recently?", weight_opts, index=weight_opts.index(st.session_state.get("weight_changes_radio", "No change")), key="weight_changes_radio", horizontal=True)
            if st.session_state.get("weight_changes_radio") in ["Significant gain", "Significant loss"]:
                st.number_input("Approximate weight change (lbs):", min_value=1, max_value=100, key="weight_details_input", value=st.session_state.get("weight_details_input", 5))

        if show_concentration:
            st.divider()
            st.header("Cognitive Function")
            conc_opts = ["Never", "Rarely", "Sometimes", "Often", "Nearly daily"]
            st.radio("Issues with focus/concentration?", conc_opts, index=conc_opts.index(st.session_state.get("concentration_radio", "Sometimes")), key="concentration_radio", horizontal=True)
            if st.session_state.get("concentration_radio") in ["Sometimes", "Often", "Nearly daily"]:
                st.text_area("Describe when focus is difficult and its impact:", key="concentration_details_input", value=st.session_state.get("concentration_details_input", ""))
            
            mem_opts = ["Not at all", "Occasionally", "Frequently"]
            st.radio("Any memory issues?", mem_opts, index=mem_opts.index(st.session_state.get("memory_issues_radio", "Occasionally")), key="memory_issues_radio", horizontal=True)

        if show_suicidal:
            st.divider()
            st.warning("The following are safety-related questions. Please answer as honestly as you can.")
            plan_opts = ["No specific thoughts", "Yes, fleeting thoughts", "Yes, considered specific methods"]
            st.radio("Have you had thoughts about specific ways to harm yourself?", plan_opts, index=plan_opts.index(st.session_state.get("suicidal_plan_radio", "No specific thoughts")), key="suicidal_plan_radio", horizontal=True)
            
            urge_opts = ["No urge", "Slight urge", "Moderate urge", "Strong urge", "Overwhelming urge"]
            st.radio("How strong has the urge been to act on these thoughts?", urge_opts, index=urge_opts.index(st.session_state.get("suicidal_intent_radio", "No urge")), key="suicidal_intent_radio")
            
            st.text_area("What helps you feel safe or gives you reasons for living?", key="protective_factors_input", value=st.session_state.get("protective_factors_input", ""))
            st.error("**If you are in immediate danger, please seek help:**\n- **Call 112** (India) or your local emergency number.\n- **Crisis Text Line:** Text HOME to 741741.")
            
        submitted = st.form_submit_button("Save and Continue to Report")

    if submitted:
        # Securely build the payload from the defined list of keys
        payload = {key: st.session_state.get(key) for key in ASSESSMENT_KEYS if st.session_state.get(key) is not None}
        
        if save_extra_assessment_data(st.session_state.user_id, payload):
            st.success("Your assessment has been saved successfully.")
            st.session_state.extra_completed = True
            st.session_state.page = "Report & Suggestions"
            st.rerun()
        else:
            st.error("There was a problem saving your assessment. Please try again.")